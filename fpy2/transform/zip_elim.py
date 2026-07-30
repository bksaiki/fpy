"""
Rewrite ``zip(...)`` iterables into indexed loops over the source
vectors, removing the need for any backend to materialize an
intermediate list of tuples.

Two patterns are recognized:

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

2. **List comp over zip** *with* ``Var`` arguments only.

   .. code-block:: python

      [elt for a, b in zip(xs, ys)]

   is rewritten to::

      [elt[a -> xs[_i], b -> ys[_i]] for _i in range(len(xs))]

   The substitution is scope-aware: if ``elt`` contains a nested
   comprehension or other construct that re-binds ``a`` or ``b``,
   the shadowed uses are not rewritten.

   Restriction: every ``zip`` argument must be a :class:`Var`.  A
   list comprehension is an expression and has no statement-level
   "preamble" to host the ``_srcK = ...`` bindings, so we can't
   safely cache a non-pure ``zip`` argument across iterations.
   The transform leaves non-``Var``-argument zip comps alone; the
   cpp backend's emit-time fast path still optimizes them.

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

Patterns that don't match the guards (range iterables, mismatched
arity, non-``Var`` list-comp zip args) are left unchanged.

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


def _is_zip_tuple_binding(target: Id | TupleBinding, iterable: Expr) -> bool:
    """Predicate for the for-loop / comp fast path.

    Fires iff *iterable* is a :class:`Zip` and *target* is a
    :class:`TupleBinding` of matching arity.  Each binding slot may be a
    :class:`NamedId`, an :class:`UnderscoreId`, or a nested
    :class:`TupleBinding` (the per-iteration element of a nested slot is
    itself a tuple, destructured via the ``fst``/``snd`` accessors in the
    comp path and via a destructuring assignment in the for-loop path).
    """
    if not isinstance(iterable, Zip):
        return False
    if not iterable.args:
        # ``zip()`` has no source to take the length from.
        return False
    if not isinstance(target, TupleBinding):
        return False
    if len(iterable.args) != len(target.elts):
        return False
    return all(
        isinstance(e, (NamedId, UnderscoreId, TupleBinding))
        for e in target.elts
    )


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
        if not _is_zip_tuple_binding(stmt.target, stmt.iterable):
            return super()._visit_for(stmt, ctx)
        # Recursively rewrite the body first, in case it contains
        # nested zip patterns.
        body, _ = self._visit_block(stmt.body, ctx)
        return self._rewrite_for(stmt, body, ctx), ctx

    def _rewrite_for(
        self, stmt: ForStmt, new_body: StmtBlock, ctx: Ctx,
    ) -> ForStmt:
        assert isinstance(stmt.iterable, Zip)
        assert isinstance(stmt.target, TupleBinding)

        # Bind each zip arg to a fresh ``_srcK`` before the loop.
        # Even ``UnderscoreId`` binding slots get a temp for the
        # source so any side-effect in the arg fires exactly once.
        src_names: list[NamedId] = []
        for arg in stmt.iterable.args:
            src = self.gensym.fresh('_src')
            ctx.stmts.append(Assign(src, None, arg, None))
            src_names.append(src)

        idx = self.gensym.fresh('_i')
        # Build the per-iteration assignments: one per non-underscore
        # binding slot.  Underscore slots are skipped; their source
        # remains bound for side-effect ordering but isn't read.
        per_iter: list[Stmt] = []
        for elt, src in zip(stmt.target.elts, src_names):
            match elt:
                case UnderscoreId():
                    continue
                case NamedId() | TupleBinding():
                    # ``NamedId`` -> ``a = src[i]``; a nested
                    # ``TupleBinding`` -> ``(a, b) = src[i]``, whose
                    # destructuring the backends already lower (the
                    # statement context needs no ``fst``/``snd``).
                    per_iter.append(
                        Assign(
                            elt, None,
                            ListRef(Var(src, None), Var(idx, None), None),
                            None,
                        )
                    )
                case _:
                    # Should be ruled out by the guard, but stay
                    # defensive.
                    raise RuntimeError(
                        f'unexpected binding element in zip target: {elt!r}'
                    )

        # New body: per-iteration assigns, then the original body.
        new_body_stmts: list[Stmt] = per_iter + list(new_body.stmts)
        # Iterable: ``range(len(_src0))``.
        size_expr = Len(
            None,
            Var(src_names[0], None),
            None,
        )
        range_expr = Range1(
            None,
            size_expr,
            None,
        )
        return ForStmt(
            idx,
            range_expr,
            StmtBlock(new_body_stmts),
            stmt.loc,
        )

    # ------------------------------------------------------------------
    # List comprehensions

    def _visit_list_comp(self, e: ListComp, ctx: Any):
        # Walk the comp's (target, iterable) pairs, rewriting any
        # zip-tuple-binding stage whose zip arguments are all
        # ``Var``s.  Non-matching stages are passed through.  The
        # ``elt`` expression has substitutions applied for every
        # rewritten stage.
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
            if (
                _is_zip_tuple_binding(target, new_iter)
                and isinstance(new_iter, Zip)
                and isinstance(target, TupleBinding)
                # Each arg is inlined into the comp body (re-evaluated per
                # iteration, since a comp has no preamble to bind it once), so it
                # must be a pure, re-evaluable access path.
                and all(is_access_path(a) for a in new_iter.args)
                # `fst`/`snd` are pair-only, so a nested slot of arity != 2 can't
                # be reached by the comp path's accessor chains; leave such a
                # `zip` in place (the backends materialize it) rather than emit
                # ill-typed `snd`-of-non-pair.
                and all(comp_binding_is_pairs(elt) for elt in target.elts)
            ):
                idx = self.gensym.fresh('_i')
                # Substitute each binding slot with an accessor into its
                # source: ``name -> arg[idx]`` for a plain slot, and the
                # ``fst``/``snd`` chain for a nested tuple binding (whose
                # per-iteration element is itself a tuple).
                for binding, arg in zip(target.elts, new_iter.args):
                    destructure_subst(
                        binding, index_access(arg, idx), subst,
                    )
                # New target/iterable: ``_i in range(len(arg0))``.
                len_expr = Len(
                    None,
                    clone(new_iter.args[0]),
                    None,
                )
                new_targets.append(idx)
                new_iterables.append(
                    Range1(None, len_expr, None)
                )
            else:
                new_targets.append(self._visit_binding(target, ctx))
                new_iterables.append(new_iter)

        # Substitute names in ``elt`` after walking it normally
        # (so nested zip patterns inside the elt are also rewritten).
        new_elt = self._visit_expr(e.elt, ctx)
        if subst:
            new_elt = SubstNames(subst)._visit_expr(new_elt, ctx)
        return ListComp(new_targets, new_iterables, new_elt, e.loc)


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
