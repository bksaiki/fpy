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

Patterns that don't match the guards (range iterables, mismatched
arity, nested ``TupleBinding`` elements, non-``Var`` list-comp zip
args) are left unchanged.

Ordering note: run :class:`ZipElim` *before*
:class:`fpy2.transform.ForUnpack`.  ``ForUnpack`` rewrites
``for (a, b) in iter:`` into ``for t in iter: a, b = t``, which
turns the ``ForStmt``'s target into a :class:`NamedId` and thereby
defeats this transform's guard.
"""

import dataclasses

from typing import Any

from ..analysis import DefineUse, DefineUseAnalysis, SyntaxCheck
from ..ast.fpyast import (
    Assign, Expr, ForStmt, FuncDef, Len, ListComp, ListRef, NamedId,
    Range1, Stmt, StmtBlock, TupleBinding, UnderscoreId, Var, Zip,
)
from ..ast.visitor import DefaultTransformVisitor
from ..utils import Gensym, Id


@dataclasses.dataclass
class _Ctx:
    """Block-walk accumulator.  When :meth:`_visit_for` decides to
    rewrite a for-loop, it appends the ``_srcK = ...`` preamble
    assignments to ``stmts`` and returns the rewritten ``ForStmt``;
    :meth:`_visit_block` then appends that, producing one-to-many
    statement expansion without a custom statement visitor."""
    stmts: list[Stmt]

    @staticmethod
    def default() -> '_Ctx':
        return _Ctx(stmts=[])


def _is_zip_tuple_binding(target: Id | TupleBinding, iterable: Expr) -> bool:
    """Predicate for the for-loop / comp fast path.

    Fires iff *iterable* is a :class:`Zip`, *target* is a
    :class:`TupleBinding` of matching arity, and every element of
    the binding is a :class:`NamedId` or :class:`UnderscoreId`.
    Nested tuple bindings are out of scope — they'd require
    recursive destructuring of each per-iteration element, which
    isn't worth the complication.
    """
    if not isinstance(iterable, Zip):
        return False
    if not isinstance(target, TupleBinding):
        return False
    if len(iterable.args) != len(target.elts):
        return False
    return all(
        isinstance(e, (NamedId, UnderscoreId))
        for e in target.elts
    )


class _SubstNames(DefaultTransformVisitor):
    """Replace every :class:`Var` reference to a name in *subst*
    with the corresponding expression.  Scope-aware: comprehension
    targets that shadow a substituted name disable the substitution
    inside that comprehension's ``elt`` (the inner uses bind to the
    shadowing iteration variable, not to the outer one).
    """

    def __init__(self, subst: dict[NamedId, Expr]):
        super().__init__()
        # Active substitutions, shadowed lexically.  We mutate in
        # place via push/pop because ``DefaultTransformVisitor`` is
        # purely top-down and gives us no return-trip hook for
        # popping; the surrounding ``_visit_list_comp`` override
        # restores after recursing.
        self._subst = dict(subst)

    def _visit_var(self, e: Var, ctx: Any):
        # The substitution targets are ``NamedId``s; FPy's parser
        # synthesizes a ``Var`` whose ``.name`` is the same
        # ``NamedId`` object.  Equality is structural (``NamedId``
        # implements ``__eq__`` over (base, count)).
        replacement = self._subst.get(e.name)
        if replacement is not None:
            return replacement
        return super()._visit_var(e, ctx)

    def _visit_list_comp(self, e: ListComp, ctx: Any):
        # A target NamedId inside this comp shadows any outer
        # substitution for the same name.  Save the shadowed entries,
        # disable them, recurse, then restore.
        shadowed: dict[NamedId, Expr] = {}
        for target in e.targets:
            for name in self._binding_names(target):
                if name in self._subst:
                    shadowed[name] = self._subst.pop(name)
        try:
            return super()._visit_list_comp(e, ctx)
        finally:
            self._subst.update(shadowed)

    @staticmethod
    def _binding_names(target) -> list[NamedId]:
        """Flatten a comprehension target into the named identifiers
        it binds.  Underscore slots and nested bindings contribute
        zero or more names."""
        match target:
            case NamedId():
                return [target]
            case UnderscoreId():
                return []
            case TupleBinding():
                out: list[NamedId] = []
                for elt in target.elts:
                    out.extend(_SubstNames._binding_names(elt))
                return out
            case _:
                return []


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
        # Local _Ctx so each block has its own preamble buffer; the
        # outer caller's ctx (if any) is irrelevant — preambles
        # always belong to the block introducing them.
        block_ctx = _Ctx.default()
        for stmt in block.stmts:
            new_stmt, _ = self._visit_statement(stmt, block_ctx)
            block_ctx.stmts.append(new_stmt)
        return StmtBlock(block_ctx.stmts), ctx

    # ------------------------------------------------------------------
    # For loops

    def _visit_for(self, stmt: ForStmt, ctx: _Ctx):
        if not _is_zip_tuple_binding(stmt.target, stmt.iterable):
            return super()._visit_for(stmt, ctx)
        # Recursively rewrite the body first, in case it contains
        # nested zip patterns.
        body, _ = self._visit_block(stmt.body, ctx)
        return self._rewrite_for(stmt, body, ctx), ctx

    def _rewrite_for(
        self, stmt: ForStmt, new_body: StmtBlock, ctx: _Ctx,
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
                case NamedId():
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
            Var(NamedId('len'), None),
            Var(src_names[0], None),
            None,
        )
        range_expr = Range1(
            Var(NamedId('range'), None),
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
            if (
                _is_zip_tuple_binding(target, new_iter)
                and isinstance(new_iter, Zip)
                and isinstance(target, TupleBinding)
                and all(isinstance(a, Var) for a in new_iter.args)
            ):
                idx = self.gensym.fresh('_i')
                # Build a substitution for each named binding slot:
                # ``name -> arg[idx]``.  Underscore slots contribute
                # no substitution.
                for binding, arg in zip(target.elts, new_iter.args):
                    match binding:
                        case NamedId():
                            assert isinstance(arg, Var)
                            subst[binding] = ListRef(
                                Var(arg.name, None),
                                Var(idx, None),
                                None,
                            )
                        case UnderscoreId():
                            continue
                        case _:
                            raise RuntimeError(
                                f'unexpected binding element in zip '
                                f'target: {binding!r}'
                            )
                # New target/iterable: ``_i in range(len(arg0))``.
                assert isinstance(new_iter.args[0], Var)
                len_expr = Len(
                    Var(NamedId('len'), None),
                    Var(new_iter.args[0].name, None),
                    None,
                )
                new_targets.append(idx)
                new_iterables.append(
                    Range1(Var(NamedId('range'), None), len_expr, None)
                )
            else:
                new_targets.append(self._visit_binding(target, ctx))
                new_iterables.append(new_iter)

        # Substitute names in ``elt`` after walking it normally
        # (so nested zip patterns inside the elt are also rewritten).
        new_elt = self._visit_expr(e.elt, ctx)
        if subst:
            new_elt = _SubstNames(subst)._visit_expr(new_elt, ctx)
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
