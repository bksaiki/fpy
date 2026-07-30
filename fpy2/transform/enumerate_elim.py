"""
Rewrite ``enumerate(...)`` iterables into indexed loops over the source
vector, so no backend has to materialize an intermediate list of
``(index, element)`` tuples.

1. **For-loop over enumerate.**

   .. code-block:: python

      for i, x in enumerate(xs):   ->   _src0 = xs
          BODY                          for i in range(len(_src0)):
                                            x = _src0[i]
                                            BODY

   The index slot gets no assignment: ``enumerate``'s index and
   ``range(len(...))``'s counter are the same value, so the user's own name
   becomes the loop variable.  The ``_src0`` temp evaluates the iterable
   exactly once and keeps the loop bound out of reach of a body that rebinds
   the source name.  A discarded element slot emits no assignment, but its
   source is still bound so any side-effect fires once.

2. **List comp over enumerate**, when the source is an access path.

   .. code-block:: python

      [elt for i, x in enumerate(xs)]   ->   [elt[x -> xs[i]] for i in range(len(xs))]

   The substitution is scope-aware: a nested comprehension that re-binds
   ``x`` shadows it, and those uses are left alone.

   A comprehension is an expression with no statement slot to host the
   ``_src0 = ...`` binding, so the source is inlined and re-evaluated per
   iteration.  That is only sound for an ``is_access_path`` — pure and O(1);
   anything else is left for the backend to materialize.

Composing with ``zip``
----------------------

``enumerate(zip(...))`` lowered naively builds *two* lists: the ``zip``'s
tuples and the ``enumerate``'s ``(index, tuple)`` pairs.  Indexing the
``zip``'s own arguments removes both at once:

.. code-block:: python

   for i, (a, b) in enumerate(zip(xs, ys)):   ->   _src0 = xs
       BODY                                        _src1 = ys
                                                   for i in range(len(_src0)):
                                                       a = _src0[i]
                                                       b = _src1[i]
                                                       BODY

This has to happen here rather than in :class:`fpy2.transform.ZipElim`, which
matches a ``zip`` in *iterable* position: after a plain ``enumerate`` rewrite
the ``zip`` sits on the right-hand side of ``_src0 = zip(xs, ys)``, out of its
reach.  Taking the bound from the first argument matches ``ZipElim`` and is
faithful because FPy's ``zip`` is strict.

A slot bound to the *whole* tuple gets it rebuilt per iteration instead —
``p = (_src0[i], _src1[i])``, one stack tuple rather than a list of them — and
the comp path substitutes ``p -> (xs[i], ys[i])``.  Neither inlines the
``zip`` itself, only one access path per source plus a tuple construction, so
both stay pure and O(1) per element.

The one shape left materialized is an element slot whose arity *disagrees*
with the ``zip``'s.  That program is already ill-typed, and keeping the
``zip`` keeps its diagnostic rather than trading it for an arity-mismatched
destructure.

A slot may also be a nested ``TupleBinding`` (``for i, (a, b) in
enumerate(pairs)``).  The for-loop path lowers it to a destructuring assign of
any arity; the comp path reaches its leaves by ``fst``/``snd``, which is
pair-only — see
:func:`fpy2.transform.iter_elim.comp_binding_is_pairs`.

Ordering: run this *before* :class:`fpy2.transform.ForUnpack`, which rewrites
``for (i, x) in iter:`` into ``for t in iter: i, x = t``, whose ``NamedId``
target defeats the guard below.  See :mod:`fpy2.transform.iter_elim` for the
machinery shared with :class:`fpy2.transform.ZipElim`.
"""

import dataclasses
from typing import Any

from ..analysis import DefineUse, DefineUseAnalysis, SyntaxCheck
from ..ast.fpyast import (
    Assign,
    Enumerate,
    Expr,
    ForStmt,
    FuncDef,
    Len,
    ListComp,
    ListRef,
    NamedId,
    Range1,
    Stmt,
    StmtBlock,
    TupleBinding,
    TupleExpr,
    UnderscoreId,
    Var,
    Zip,
)
from ..ast.visitor import DefaultTransformVisitor
from ..utils import Gensym, Id
from .iter_elim import (
    Ctx,
    SubstNames,
    clone,
    comp_binding_is_pairs,
    destructure_subst,
    index_access,
    is_access_path,
)

# An element binding slot: a name, a discard, or a nested destructure.
_Slot = Id | TupleBinding


def _split_target(
    target: Id | TupleBinding, iterable: Expr,
) -> tuple[Id, _Slot] | None:
    """Split ``for i, x in enumerate(src)``'s target into its index and
    element slots, or ``None`` if *target* / *iterable* aren't that pattern.

    The index slot must be a plain name or a discard: ``enumerate`` yields an
    integer there, so a destructuring slot would be ill-typed.
    """
    if not isinstance(iterable, Enumerate):
        return None
    if not isinstance(target, TupleBinding) or len(target.elts) != 2:
        return None
    idx, elt = target.elts
    if not isinstance(idx, (NamedId, UnderscoreId)):
        return None
    if not isinstance(elt, (NamedId, UnderscoreId, TupleBinding)):
        return None
    return idx, elt


@dataclasses.dataclass
class _Plan:
    """Which sources to index, and which bindings read them.

    ``args[0]`` also supplies the loop bound.  With ``tupled``, one slot reads
    the whole element — rebuilt as ``(args[0][i], ..., args[n][i])``; without
    it, each slot reads its own ``args[k][i]``.
    """
    args: list[Expr]
    slots: list[_Slot]
    tupled: bool

    def __post_init__(self):
        assert self.args, 'need a source for the bound'
        assert len(self.slots) == (1 if self.tupled else len(self.args))


def _plan(src: Expr, elt: _Slot) -> _Plan:
    """How to reach the element bound by *elt* when iterating ``enumerate(src)``.

    A ``zip`` source decomposes into its own arguments, so neither
    intermediate list is built.  Anything else is indexed whole.
    """
    # ``zip()`` has no source to take the bound from.
    if isinstance(src, Zip) and src.args:
        if (
            isinstance(elt, TupleBinding)
            and len(elt.elts) == len(src.args)
            and all(
                isinstance(e, (NamedId, UnderscoreId, TupleBinding))
                for e in elt.elts
            )
        ):
            return _Plan(list(src.args), list(elt.elts), tupled=False)
        if isinstance(elt, (NamedId, UnderscoreId)):
            return _Plan(list(src.args), [elt], tupled=True)
        # A `TupleBinding` of mismatched arity falls through: see the module
        # docstring on why the `zip` is left materialized.
    return _Plan([src], [elt], tupled=False)


class _EnumerateElimInstance(DefaultTransformVisitor):
    """Single-use visitor that drives the rewrite."""

    def __init__(self, func: FuncDef, def_use: DefineUseAnalysis):
        super().__init__()
        self.func = func
        self.gensym = Gensym(reserved=def_use.names())

    def apply(self) -> FuncDef:
        return self._visit_function(self.func, None)

    def _index_name(self, idx: Id) -> NamedId:
        """The counter standing in for the index slot: a named slot *is* the
        counter, a discarded one gets a fresh name nothing reads."""
        if isinstance(idx, NamedId):
            return idx
        return self.gensym.fresh('_i')

    # ------------------------------------------------------------------
    # Block walk with stmt → stmts expansion

    def _visit_block(self, block: StmtBlock, ctx: Any):
        # A local Ctx per block: preambles belong to the block introducing them.
        block_ctx = Ctx.default()
        for stmt in block.stmts:
            new_stmt, _ = self._visit_statement(stmt, block_ctx)
            block_ctx.stmts.append(new_stmt)
        return StmtBlock(block_ctx.stmts), ctx

    # ------------------------------------------------------------------
    # For loops

    def _visit_for(self, stmt: ForStmt, ctx: Ctx):
        split = _split_target(stmt.target, stmt.iterable)
        if split is None:
            return super()._visit_for(stmt, ctx)
        # Recursively rewrite the body first, in case it contains nested
        # enumerate patterns.
        body, _ = self._visit_block(stmt.body, ctx)
        return self._rewrite_for(stmt, split, body, ctx), ctx

    def _rewrite_for(
        self,
        stmt: ForStmt,
        split: tuple[Id, _Slot],
        new_body: StmtBlock,
        ctx: Ctx,
    ) -> ForStmt:
        assert isinstance(stmt.iterable, Enumerate)
        idx_slot, elt_slot = split
        plan = _plan(stmt.iterable.arg, elt_slot)

        # Bind each source to a fresh `_srcK` before the loop — a discarded
        # slot's source too, so any side-effect in it still fires once.
        src_names: list[NamedId] = []
        for arg in plan.args:
            src = self.gensym.fresh('_src')
            ctx.stmts.append(Assign(src, None, arg, None))
            src_names.append(src)

        idx = self._index_name(idx_slot)
        # A tupled plan feeds every source into one slot; otherwise each slot
        # reads its own.  `plan.tupled` is the discriminator, not the source
        # count: `zip(xs)` is tupled over one source and still yields 1-tuples.
        groups = [src_names] if plan.tupled else [[s] for s in src_names]
        per_iter: list[Stmt] = []
        for slot, srcs in zip(plan.slots, groups):
            if isinstance(slot, UnderscoreId):
                continue
            refs = [ListRef(Var(s, None), Var(idx, None), None) for s in srcs]
            # A nested slot becomes `(a, b) = src[i]` — the backends already
            # destructure, so a statement context needs no `fst`/`snd`.
            value = TupleExpr(refs, None) if plan.tupled else refs[0]
            per_iter.append(Assign(slot, None, value, None))

        return ForStmt(
            idx,
            Range1(None, Len(None, Var(src_names[0], None), None), None),
            StmtBlock(per_iter + list(new_body.stmts)),
            stmt.loc,
        )

    # ------------------------------------------------------------------
    # List comprehensions

    def _visit_list_comp(self, e: ListComp, ctx: Any):
        new_targets: list[Id | TupleBinding] = []
        new_iterables: list[Expr] = []
        subst: dict[NamedId, Expr] = {}

        for target, iterable in zip(e.targets, e.iterables):
            new_iter = self._visit_expr(iterable, ctx)
            # A later stage's iterable may reference an earlier stage's target
            # (`[... for i, x in enumerate(xs) for y in x]`), whose name no
            # longer exists once that stage is rewritten.
            if subst:
                new_iter = SubstNames(subst)._visit_expr(new_iter, ctx)
            rewritten = self._rewrite_comp_stage(target, new_iter, subst)
            if rewritten is None:
                new_targets.append(self._visit_binding(target, ctx))
                new_iterables.append(new_iter)
            else:
                new_target, new_iterable = rewritten
                new_targets.append(new_target)
                new_iterables.append(new_iterable)

        # Walk `elt` normally first, so nested enumerate patterns inside it are
        # rewritten too, then substitute.
        new_elt = self._visit_expr(e.elt, ctx)
        if subst:
            new_elt = SubstNames(subst)._visit_expr(new_elt, ctx)
        return ListComp(new_targets, new_iterables, new_elt, e.loc)

    def _rewrite_comp_stage(
        self,
        target: Id | TupleBinding,
        iterable: Expr,
        subst: dict[NamedId, Expr],
    ) -> tuple[Id, Expr] | None:
        """The rewritten ``(target, iterable)`` for one comprehension stage,
        extending *subst* with the accessors its element slot needs; ``None``
        to leave the stage as it is."""
        split = _split_target(target, iterable)
        if split is None:
            return None
        assert isinstance(iterable, Enumerate)
        idx_slot, elt_slot = split
        plan = _plan(iterable.arg, elt_slot)

        # Sources are inlined and re-evaluated per iteration, so each must be
        # pure and O(1).  This is also what rules out a `zip` source `_plan`
        # left whole: inlining it would rebuild the list per element.
        if not all(is_access_path(a) for a in plan.args):
            return None

        idx = self._index_name(idx_slot)
        if plan.tupled:
            # A whole-element slot is a name or a discard, so no `fst`/`snd`
            # chain is involved and `comp_binding_is_pairs` has nothing to say.
            slot = plan.slots[0]
            if isinstance(slot, NamedId):
                subst[slot] = TupleExpr(
                    [index_access(arg, idx)() for arg in plan.args], None,
                )
        else:
            # `fst`/`snd` are pair-only, so a destructuring slot of arity != 2
            # can't be reached; leave the stage rather than emit an ill-typed
            # `snd`-of-non-pair.
            if not all(comp_binding_is_pairs(slot) for slot in plan.slots):
                return None
            for slot, arg in zip(plan.slots, plan.args):
                destructure_subst(slot, index_access(arg, idx), subst)
        return idx, Range1(None, Len(None, clone(plan.args[0]), None), None)


class EnumerateElim:
    """Rewrite ``enumerate(...)`` iterables into indexed loops over the
    source vector, including the ``enumerate(zip(...))`` case where both
    intermediate lists collapse into direct indexing.  See the module
    docstring for the patterns recognized and the ordering constraint with
    :class:`fpy2.transform.ForUnpack`."""

    @staticmethod
    def apply(func: FuncDef) -> FuncDef:
        """Apply the transformation to a :class:`FuncDef`.  Returns a new
        ``FuncDef``; the input is not mutated."""
        if not isinstance(func, FuncDef):
            raise TypeError(f"expected a 'FuncDef', got `{func}`")
        def_use = DefineUse.analyze(func)
        out = _EnumerateElimInstance(func, def_use).apply()
        SyntaxCheck.check(out, ignore_unknown=True)
        return out
