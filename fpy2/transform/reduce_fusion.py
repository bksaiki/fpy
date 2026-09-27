"""
Fuse ``any`` / ``all`` over a list comprehension into a single loop,
eliminating the intermediate ``list[bool]``::

    # Before
    r = any([<elt> for <t> in <iter>])

    # After
    acc = False
    for <t> in <iter>:
        b = <elt>
        acc = acc or b
    r = acc

(``all`` seeds ``True`` and combines with ``and``.)

Unfused, the comprehension materializes the whole list before the reduction
scans it.  Whether skipping that is a *win* depends on the list's
representation, and where the length is proven it is a loss -- see "When
``ReduceFusion`` pays, and when it costs" in ``docs/todos/backend-cpp.md`` for
the measurements and why the pass runs anyway.

``b`` is bound rather than inlined into ``acc or <elt>``, and the bind is
load-bearing: FPy's ``or`` short-circuits, so an inlined element would stop
being evaluated once ``acc`` is ``True``.  Element expressions are not total
(an out-of-bounds ``xs[i]``, a stuck ``fp.cast``), so skipping them is
observable.  Binding forces every element, matching both the unfused program
and CPython — whose ``any`` short-circuits the *iterable*, but is handed an
already-built list.

Only the boolean reductions are fused, and the others turn out not to want it:
fusing ``Sum`` / ``AMin`` / ``AMax`` measures neutral-to-worse, and is *also*
blocked on the cpp emitter accepting an implicit narrowing inside
``std::accumulate`` that it rejects at an ordinary assignment.  Both are in
``docs/todos/backend-cpp.md``.  Multi-stage comprehensions
(``[e for a in xs for b in ys]``) would need nested loops and are left alone.

A reduction is fused only where :class:`~fpy2.analysis.Hoistability` says a
statement may go before its own -- not in a ternary arm, an ``and``/``or`` tail
or a ``while`` condition, nor to the right of an operand the loop would
overtake.  Run :class:`~fpy2.transform.Hoistable` first to make that everywhere.
"""

import dataclasses
from typing import Any

from ..analysis import DefineUse, DefineUseAnalysis, Hoistability, SyntaxCheck
from ..ast.fpyast import (
    AllOf,
    And,
    AnyOf,
    Assign,
    BoolVal,
    Expr,
    ForStmt,
    FuncDef,
    ListComp,
    Or,
    Stmt,
    StmtBlock,
    Var,
)
from ..ast.visitor import DefaultTransformVisitor
from ..utils import Gensym


@dataclasses.dataclass
class _Ctx:
    """Block-walk accumulator: :meth:`_fuse` appends the seed and loop here,
    and :meth:`_visit_block` emits them before the enclosing statement."""
    stmts: list[Stmt]

    @staticmethod
    def default() -> '_Ctx':
        return _Ctx(stmts=[])


class _ReduceFusionInstance(DefaultTransformVisitor):
    """Drives the rewrite.  Single-use — one instance per
    :meth:`ReduceFusion.apply` call."""

    func: FuncDef
    gensym: Gensym
    slots: set[Expr]
    """where a statement may go before the expression's own"""

    def __init__(self, func: FuncDef, def_use: DefineUseAnalysis):
        self.func = func
        self.gensym = Gensym(reserved=def_use.names())
        self.slots = Hoistability.analyze(func).slots

    def apply(self) -> FuncDef:
        return self._visit_function(self.func, None)

    # ------------------------------------------------------------------
    # Block walk — the ``_Ctx``-accumulator pattern from ``ZipElim``.

    def _visit_block(self, block: StmtBlock, ctx: Any) -> tuple[StmtBlock, Any]:
        block_ctx = _Ctx.default()
        for stmt in block.stmts:
            new_stmt, _ = self._visit_statement(stmt, block_ctx)
            block_ctx.stmts.append(new_stmt)
        return StmtBlock(block_ctx.stmts), ctx

    # ------------------------------------------------------------------
    # Expression rewriting

    def _visit_expr(self, e: Expr, ctx: Any) -> Expr:
        if (
            e in self.slots
            and isinstance(e, (AnyOf, AllOf))
            and isinstance(e.arg, ListComp)
            # multi-stage comps would need nested loops; leave them alone
            and len(e.arg.targets) == 1
        ):
            return self._fuse(e, e.arg, ctx)
        return super()._visit_expr(e, ctx)

    def _fuse(self, e: 'AnyOf | AllOf', comp: ListComp, ctx: _Ctx) -> Expr:
        """Emit the seed + loop into *ctx* and return ``Var(acc)``."""
        is_any = isinstance(e, AnyOf)
        acc = self.gensym.fresh('acc')
        elt = self.gensym.fresh('b')

        # nothing inside is fused: a comprehension's iterable and element are
        # sealed
        iterable = self._visit_expr(comp.iterables[0], ctx)
        target = self._visit_binding(comp.targets[0], ctx)
        elt_expr = self._visit_expr(comp.elt, ctx)

        op = Or if is_any else And
        combine = op([Var(acc, e.loc), Var(elt, e.loc)], e.loc)
        body = StmtBlock([
            # binding `b` preserves the unfused evaluation count -- see the
            # module docstring; folding `elt` inline would short-circuit it
            Assign(elt, None, elt_expr, e.loc),
            Assign(acc, None, combine, e.loc),
        ])

        ctx.stmts.append(Assign(acc, None, BoolVal(not is_any, e.loc), e.loc))
        ctx.stmts.append(ForStmt(target, iterable, body, e.loc))
        return Var(acc, e.loc)


class ReduceFusion:
    """Fuse ``any`` / ``all`` over a list comprehension into a single loop,
    eliminating the intermediate ``list[bool]``.  See the module docstring
    for the rewrite shape and why the element is bound to a temp."""

    @staticmethod
    def apply(func: FuncDef) -> FuncDef:
        """Apply the transformation to a :class:`FuncDef`.  Returns a new
        ``FuncDef``; the input is not mutated."""
        if not isinstance(func, FuncDef):
            raise TypeError(f"expected a 'FuncDef', got `{func}`")
        def_use = DefineUse.analyze(func)
        out = _ReduceFusionInstance(func, def_use).apply()
        SyntaxCheck.check(out, ignore_unknown=True)
        return out
