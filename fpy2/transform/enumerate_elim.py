"""
Rewrite ``enumerate(...)`` iterables into indexed loops over the source
vector, removing the need for any backend to materialize an intermediate
list of ``(index, element)`` tuples.

Two patterns are recognized:

1. **For-loop over enumerate.**

   .. code-block:: python

      for i, x in enumerate(xs):
          BODY

   is rewritten to::

      _src0 = xs
      for i in range(len(_src0)):
          x = _src0[i]
          BODY

   The index slot needs no per-iteration assignment: ``enumerate`` yields
   exactly the counter ``range(len(...))`` already produces, so the user's
   own name becomes the loop variable.  The source temporary preserves
   "evaluate the iterable exactly once" semantics and keeps the loop bound
   from being re-read through a name the body rebinds.  An underscore
   element slot emits no assignment, but its source is still bound to a temp
   so any side-effect in it fires exactly once.

2. **List comp over enumerate**, when the source is an access path.

   .. code-block:: python

      [elt for i, x in enumerate(xs)]

   is rewritten to::

      [elt[x -> xs[i]] for i in range(len(xs))]

   The substitution is scope-aware: if ``elt`` contains a nested
   comprehension that re-binds ``x``, the shadowed uses are not rewritten.

   Restriction: the ``enumerate`` argument must be an access path (a ``Var``
   or a pure O(1) projection/index chain over one).  A list comprehension is
   an expression with no statement-level "preamble" to host the
   ``_src0 = ...`` binding, so the source is inlined and re-evaluated once
   per iteration — sound only when that is free of side effects and O(1).
   Comps over a non-access-path source are left alone for the backend to
   materialize.

Composing with ``zip``
----------------------

``enumerate(zip(...))`` is the case worth special-casing: lowered naively it
builds *two* intermediate lists, the ``zip``'s list of tuples and the
``enumerate``'s list of ``(index, tuple)`` pairs.  When the element slot
destructures the ``zip`` positionally, both disappear at once — the transform
indexes the ``zip``'s own arguments directly:

.. code-block:: python

   for i, (a, b) in enumerate(zip(xs, ys)):
       BODY

becomes::

   _src0 = xs
   _src1 = ys
   for i in range(len(_src0)):
       a = _src0[i]
       b = _src1[i]
       BODY

This has to happen here rather than being left to
:class:`fpy2.transform.ZipElim`: that transform matches a ``zip`` in an
*iterable* position, and after a plain ``enumerate`` rewrite the ``zip`` sits
on the right-hand side of ``_src0 = zip(xs, ys)``, where nothing reaches it.
Taking the length from the first ``zip`` argument matches ``ZipElim`` and is
faithful because FPy's ``zip`` is strict — it requires every input to have
the same length.

When the element slot does *not* destructure the ``zip`` positionally — it is
a single name bound to the whole tuple, or a binding of mismatched arity —
only the ``enumerate`` is eliminated (``_src0 = zip(xs, ys)``) and the
``zip``'s list is still materialized.  That is no worse than before:
``ZipElim`` does not fire on such a pattern either.  The comp path skips the
fallback entirely, since a ``zip`` is not an access path and inlining it
would make every iteration rebuild the whole list.

Nested binding slots
--------------------

An element slot may itself be a nested ``TupleBinding``, e.g.
``for i, (a, b) in enumerate(pairs)``.  In the for-loop path it lowers to a
destructuring assignment ``(a, b) = _src0[i]`` of any arity.  In the comp
path — which has no statement context — each leaf is reached by an
``fst``/``snd`` chain, which is pair-only, so such a slot is rewritten only
when it (and anything nested within it) is a pair; see
:func:`fpy2.transform.iter_elim.comp_binding_is_pairs`.

Ordering note: run :class:`EnumerateElim` *before*
:class:`fpy2.transform.ForUnpack`, for the same reason as ``ZipElim`` —
``ForUnpack`` rewrites ``for (i, x) in iter:`` into
``for t in iter: i, x = t``, whose :class:`NamedId` target defeats this
transform's guard.

See :mod:`fpy2.transform.iter_elim` for the machinery this shares with
:class:`fpy2.transform.ZipElim`.
"""

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


def _sources(src: Expr, elt: _Slot) -> tuple[list[Expr], list[_Slot]]:
    """The per-slot sources to index, paired with the slots reading them.

    For ``enumerate(zip(a, b))`` whose element slot destructures the ``zip``
    positionally, these are the ``zip``'s own arguments — so neither
    intermediate list is built.  Otherwise it is the ``enumerate`` argument
    itself, read by the element slot as a whole.
    """
    if (
        isinstance(src, Zip)
        # ``zip()`` has no source to take the length from.
        and src.args
        and isinstance(elt, TupleBinding)
        and len(elt.elts) == len(src.args)
        and all(
            isinstance(e, (NamedId, UnderscoreId, TupleBinding))
            for e in elt.elts
        )
    ):
        return list(src.args), list(elt.elts)
    return [src], [elt]


class _EnumerateElimInstance(DefaultTransformVisitor):
    """Single-use visitor that drives the rewrite."""

    func: FuncDef
    gensym: Gensym

    def __init__(self, func: FuncDef, def_use: DefineUseAnalysis):
        super().__init__()
        self.func = func
        self.gensym = Gensym(reserved=def_use.names())

    def apply(self) -> FuncDef:
        return self._visit_function(self.func, None)

    def _index_name(self, idx: Id) -> NamedId:
        """The counter standing in for the index slot.  A named slot *is* the
        counter — ``enumerate``'s index and ``range(len(...))``'s counter are
        the same value — so no copy is needed; a discarded slot gets a fresh
        name nothing reads."""
        if isinstance(idx, NamedId):
            return idx
        return self.gensym.fresh('_i')

    # ------------------------------------------------------------------
    # Block walk with stmt → stmts expansion

    def _visit_block(self, block: StmtBlock, ctx: Any):
        # Local Ctx so each block has its own preamble buffer; the outer
        # caller's ctx (if any) is irrelevant — preambles always belong to
        # the block introducing them.
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
        args, slots = _sources(stmt.iterable.arg, elt_slot)

        # Bind each source to a fresh ``_srcK`` before the loop.  Even a
        # discarded slot's source gets a temp, so any side-effect in it fires
        # exactly once.
        src_names: list[NamedId] = []
        for arg in args:
            src = self.gensym.fresh('_src')
            ctx.stmts.append(Assign(src, None, arg, None))
            src_names.append(src)

        idx = self._index_name(idx_slot)
        # One per-iteration assignment per non-discarded slot.
        per_iter: list[Stmt] = []
        for slot, src in zip(slots, src_names):
            match slot:
                case UnderscoreId():
                    continue
                case NamedId() | TupleBinding():
                    # ``NamedId`` -> ``x = src[i]``; a nested ``TupleBinding``
                    # -> ``(a, b) = src[i]``, whose destructuring the backends
                    # already lower (a statement context needs no
                    # ``fst``/``snd``).
                    per_iter.append(
                        Assign(
                            slot, None,
                            ListRef(Var(src, None), Var(idx, None), None),
                            None,
                        )
                    )
                case _:
                    # Ruled out by `_split_target` / `_sources`, but stay
                    # defensive.
                    raise RuntimeError(
                        'unexpected binding element in enumerate target: '
                        f'{slot!r}'
                    )

        # Iterable: ``range(len(_src0))``.
        size_expr = Len(None, Var(src_names[0], None), None)
        return ForStmt(
            idx,
            Range1(None, size_expr, None),
            StmtBlock(per_iter + list(new_body.stmts)),
            stmt.loc,
        )

    # ------------------------------------------------------------------
    # List comprehensions

    def _visit_list_comp(self, e: ListComp, ctx: Any):
        # Walk the comp's (target, iterable) pairs, rewriting every enumerate
        # stage whose sources are access paths.  Non-matching stages pass
        # through.  Accumulated substitutions apply to the ``elt`` and to any
        # later stage's iterable.
        new_targets: list[Id | TupleBinding] = []
        new_iterables: list[Expr] = []
        subst: dict[NamedId, Expr] = {}

        for target, iterable in zip(e.targets, e.iterables):
            new_iter = self._visit_expr(iterable, ctx)
            # A later stage's iterable may reference an earlier stage's target
            # (``[... for i, x in enumerate(xs) for y in x]``), whose name no
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

        # Substitute names in ``elt`` after walking it normally (so nested
        # enumerate patterns inside the elt are also rewritten).
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
        args, slots = _sources(iterable.arg, elt_slot)

        # Every source is inlined into the comp body and re-evaluated per
        # iteration (no preamble to bind it once), so it must be pure and
        # O(1).  This also rules out the ``enumerate(zip(...))`` fallback,
        # whose single source is the ``zip`` itself.
        if not all(is_access_path(a) for a in args):
            return None
        # ``fst``/``snd`` are pair-only, so a destructuring slot of arity != 2
        # can't be reached by the comp path's accessor chains; leave the stage
        # alone rather than emit an ill-typed ``snd``-of-non-pair.
        if not all(comp_binding_is_pairs(slot) for slot in slots):
            return None

        idx = self._index_name(idx_slot)
        # Substitute each element slot with an accessor into its source:
        # ``name -> arg[idx]`` for a plain slot, and the ``fst``/``snd`` chain
        # for a nested tuple binding (whose element is itself a tuple).
        for slot, arg in zip(slots, args):
            destructure_subst(slot, index_access(arg, idx), subst)
        # New target/iterable: ``i in range(len(arg0))``.
        len_expr = Len(None, clone(args[0]), None)
        return idx, Range1(None, len_expr, None)


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
