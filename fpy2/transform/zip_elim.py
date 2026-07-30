"""
Rewrite ``zip(...)`` iterables into indexed loops over the source
vectors, removing the need for any backend to materialize an
intermediate list of tuples.

Two positions are recognized — a ``for``-loop iterable and a comprehension
iterable:

1. **For-loop over zip.**

   .. code-block:: python

      for a, b in zip(xs, ys):
          BODY

   is rewritten to::

      _src0 = xs
      _src1 = ys
      for _i in range(len(_src0)):
          a = _src0[_i]
          b = _src1[_i]
          BODY

   The per-source temporaries preserve "evaluate each iterable
   exactly once" semantics, identical to a faithful ``zip``
   implementation.  Underscore-binding targets emit no per-iteration
   assignment but their source is still bound to a temp so its
   side-effects fire exactly once.

2. **List comp over zip**, when every argument is an access path.

   .. code-block:: python

      [elt for a, b in zip(xs, ys)]

   is rewritten to::

      [elt[a -> xs[_i], b -> ys[_i]] for _i in range(len(xs))]

   The substitution is scope-aware: if ``elt`` contains a nested
   comprehension or other construct that re-binds ``a`` or ``b``,
   the shadowed uses are not rewritten.

   Restriction: every ``zip`` argument must be an
   :func:`fpy2.transform.iter_elim.is_access_path` — a ``Var`` or a pure,
   O(1) index/projection chain over one.  A list comprehension is an
   expression with no statement-level "preamble" to host the
   ``_srcK = ...`` bindings, so arguments are inlined and re-evaluated
   per iteration, which is only sound when they are pure and cheap.
   Comps with any other argument are left alone.

Whole-tuple targets
-------------------

A target need not destructure.  Bound to one name, the tuple is rebuilt per
iteration from the indexed sources — one stack tuple rather than an n-element
list of them::

   for p in zip(xs, ys):   ->   _src0 = xs
       BODY                     _src1 = ys
                                for _i in range(len(_src0)):
                                    p = (_src0[_i], _src1[_i])
                                    BODY

and the comp path substitutes ``p -> (xs[_i], ys[_i])``.  A discarded target
(``for _ in zip(xs, ys)``) builds nothing at all — its sources stay bound only
for their side-effects and their length.

Note ``zip(xs)`` yields *1-tuples*, so the decision to build a tuple keys on
the plan rather than on the number of sources.

Nested binding slots
--------------------

A binding slot may itself be a nested ``TupleBinding`` (e.g.
``for (a, b), c in zip(pairs, xs)``).  In the for-loop path the nested
slot lowers to a destructuring assignment ``(a, b) = _srcK[_i]`` (any arity;
no ``fst``/``snd``).  In the comp path — which has no statement context —
each leaf name is reached by an ``fst``/``snd`` chain over ``argK[_i]``
(for a pair ``t``, ``fst(t)`` / ``snd(t)``).  Because ``fst``/``snd`` are
pair-only, the comp path rewrites a nested slot only when it (and any
binding nested within it) is a pair; a comp with a nested slot of arity != 2
is left unchanged for the backend to materialize (see
:func:`fpy2.transform.iter_elim.comp_binding_is_pairs`).

Patterns that don't match the guards are left unchanged: a range iterable, a
destructuring target whose arity disagrees with the ``zip``'s (already
ill-typed, so keeping the ``zip`` keeps its diagnostic), and non-access-path
arguments in the comp path.

Ordering note: run :class:`ZipElim` *before*
:class:`fpy2.transform.ForUnpack`.  ``ForUnpack`` rewrites
``for (a, b) in iter:`` into ``for t in iter: a, b = t``, which
turns the ``ForStmt``'s target into a :class:`NamedId` and thereby
defeats this transform's guard.

See :mod:`fpy2.transform.iter_elim` for the machinery this shares with
:class:`fpy2.transform.EnumerateElim`, which handles the analogous
``enumerate(...)`` patterns.
"""

from typing import Any

from ..analysis import DefineUse, DefineUseAnalysis, SyntaxCheck
from ..ast.fpyast import (
    Assign,
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
    Plan,
    SubstNames,
    clone,
    comp_binding_is_pairs,
    destructure_subst,
    index_access,
    is_access_path,
    plan_for_zip,
)


def _plan(target: Id | TupleBinding, iterable: Expr) -> Plan | None:
    """How *target* reads one element of *iterable*, or ``None`` when this
    transform does not apply.

    Fires iff *iterable* is a :class:`Zip` whose element *target* can read by
    indexing the ``zip``'s own arguments — either destructuring them
    positionally, or binding the whole tuple to one name (rebuilt per
    iteration).  See :func:`fpy2.transform.iter_elim.plan_for_zip`.
    """
    if not isinstance(iterable, Zip):
        return None
    return plan_for_zip(list(iterable.args), target)


class _ZipElimInstance(DefaultTransformVisitor):
    """Single-use visitor that drives the rewrite."""

    def __init__(self, func: FuncDef, def_use: DefineUseAnalysis):
        super().__init__()
        self.func = func
        self.gensym = Gensym(reserved=def_use.names())

    def apply(self) -> FuncDef:
        return self._visit_function(self.func, None)

    # ------------------------------------------------------------------
    # Block walk with stmt → stmts expansion

    def _visit_block(self, block: StmtBlock, ctx: Any):
        # Local Ctx so each block has its own preamble buffer; the
        # outer caller's ctx (if any) is irrelevant — preambles
        # always belong to the block introducing them.
        block_ctx = Ctx.default()
        for stmt in block.stmts:
            new_stmt, _ = self._visit_statement(stmt, block_ctx)
            block_ctx.stmts.append(new_stmt)
        return StmtBlock(block_ctx.stmts), ctx

    # ------------------------------------------------------------------
    # For loops

    def _visit_for(self, stmt: ForStmt, ctx: Ctx):
        plan = _plan(stmt.target, stmt.iterable)
        if plan is None:
            return super()._visit_for(stmt, ctx)
        # Recursively rewrite the body first, in case it contains
        # nested zip patterns.
        body, _ = self._visit_block(stmt.body, ctx)
        return self._rewrite_for(stmt, plan, body, ctx), ctx

    def _rewrite_for(
        self, stmt: ForStmt, plan: Plan, new_body: StmtBlock, ctx: Ctx,
    ) -> ForStmt:
        # Bind each zip arg to a fresh ``_srcK`` before the loop.
        # Even ``UnderscoreId`` binding slots get a temp for the
        # source so any side-effect in the arg fires exactly once.
        src_names: list[NamedId] = []
        for arg in plan.args:
            src = self.gensym.fresh('_src')
            ctx.stmts.append(Assign(src, None, arg, None))
            src_names.append(src)

        idx = self.gensym.fresh('_i')
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
            # A later stage's iterable may reference an earlier stage's
            # target (``[... for a, b in zip(xs, ys) for c in a]``), whose
            # name no longer exists once that stage is rewritten.
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

        # Walk `elt` normally first, so nested zip patterns inside it are
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
        extending *subst* with the accessors its slots need; ``None`` to leave
        the stage as it is."""
        plan = _plan(target, iterable)
        if plan is None:
            return None
        # Sources are inlined and re-evaluated per iteration (a comp has no
        # preamble to bind them once), so each must be pure and O(1).
        if not all(is_access_path(a) for a in plan.args):
            return None

        idx = self.gensym.fresh('_i')
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


class ZipElim:
    """Rewrite ``zip(...)`` iterables into indexed loops over the
    source vectors.  See the module docstring for the patterns
    recognized and the ordering constraint with
    :class:`fpy2.transform.ForUnpack`."""

    @staticmethod
    def apply(func: FuncDef) -> FuncDef:
        """Apply the transformation to a :class:`FuncDef`.  Returns
        a new ``FuncDef``; the input is not mutated."""
        if not isinstance(func, FuncDef):
            raise TypeError(f"expected a 'FuncDef', got `{func}`")
        def_use = DefineUse.analyze(func)
        out = _ZipElimInstance(func, def_use).apply()
        SyntaxCheck.check(out, ignore_unknown=True)
        return out
