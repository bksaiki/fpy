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
scans it.  In C++ that is a ``std::vector<bool>`` — the bit-packed
specialization, so a heap allocation plus per-element bit twiddling — which
``-O2`` does not elide; the fused form measures 2-4x faster.

``b`` is bound rather than inlined into ``acc or <elt>``, and the bind is
load-bearing: FPy's ``or`` short-circuits, so an inlined element would stop
being evaluated once ``acc`` is ``True``.  Element expressions are not total
(an out-of-bounds ``xs[i]``, a stuck ``fp.cast``), so skipping them is
observable.  Binding forces every element, matching both the unfused program
and CPython — whose ``any`` short-circuits the *iterable*, but is handed an
already-built list.

Only the boolean reductions are fused.  ``Sum`` / ``AMin`` / ``AMax`` pay the
same allocation cost, but fusing them is blocked on the cpp emitter accepting
an implicit narrowing inside ``std::accumulate`` that it rejects at an
ordinary assignment; tracked under "Open TODOs" in ``docs/todos/backend-cpp.md``,
where the blocker lives.  Multi-stage comprehensions
(``[e for a in xs for b in ys]``) would need nested loops and are left alone.
"""

import dataclasses
from typing import Any

from ..analysis import DefineUse, DefineUseAnalysis, SyntaxCheck
from ..ast.fpyast import (
    AllOf,
    And,
    AnyOf,
    Assign,
    BoolVal,
    Expr,
    ForStmt,
    FuncDef,
    IfExpr,
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
    and :meth:`_visit_block` emits them before the enclosing statement.  A
    ``ctx`` of ``None`` instead of a ``_Ctx`` marks a position with no
    statement slot to hoist into, suppressing fusion there."""
    stmts: list[Stmt]

    @staticmethod
    def default() -> '_Ctx':
        return _Ctx(stmts=[])


class _ReduceFusionInstance(DefaultTransformVisitor):
    """Drives the rewrite.  Single-use — one instance per
    :meth:`ReduceFusion.apply` call."""

    func: FuncDef
    gensym: Gensym

    def __init__(self, func: FuncDef, def_use: DefineUseAnalysis):
        self.func = func
        self.gensym = Gensym(reserved=def_use.names())

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
            isinstance(ctx, _Ctx)
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

        # The iterable is evaluated once, before the loop, so it keeps `ctx`
        # and a fusable reduction inside it hoists to this block too.  The
        # element sees the loop target, so it gets no statement slot.
        iterable = self._visit_expr(comp.iterables[0], ctx)
        target = self._visit_binding(comp.targets[0], ctx)
        elt_expr = self._visit_expr(comp.elt, None)

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

    # ------------------------------------------------------------------
    # Positions with no statement-level slot: suppress fusion.

    def _visit_list_comp(self, e: ListComp, ctx: Any) -> ListComp:
        # The elt sees the loop targets and successive iterables reference
        # earlier targets, so nothing inside can be hoisted to the enclosing
        # block.  (Mirrors ``RoundElim._visit_list_comp``.)
        targets = [self._visit_binding(t, ctx) for t in e.targets]
        iterables = [self._visit_expr(i, None) for i in e.iterables]
        elt = self._visit_expr(e.elt, None)
        return ListComp(targets, iterables, elt, e.loc)

    def _visit_if_expr(self, e: IfExpr, ctx: Any) -> IfExpr:
        # The branches are conditional; hoisting a loop out of one would run
        # it unconditionally, which is observable when the element expression
        # can fault.  The cond is unconditional, so it keeps ``ctx``.
        cond = self._visit_expr(e.cond, ctx)
        ift = self._visit_expr(e.ift, None)
        iff = self._visit_expr(e.iff, None)
        return IfExpr(cond, ift, iff, e.loc)


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
