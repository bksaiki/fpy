"""
Context inference.
"""

from dataclasses import dataclass
from typing import TypeAlias

from ..ast import *
from ..fpc_context import FPCoreContext
from ..number import Context
from ..utils import DEFAULT, DefaultOr
from .define_use import DefineUse, DefineUseAnalysis, Definition, DefSite

_Context: TypeAlias = DefaultOr[Context | None]


@dataclass(frozen=True)
class ContextAnalysis:
    ret_ctx: _Context
    by_def: dict[Definition, _Context]
    by_expr: dict[Expr, _Context]


class _ContextInferInstance(DefaultVisitor):
    """
    Context inference instance.

    This visitor traverses the function and infers rounding contexts
    for each definition site.
    """

    func: FuncDef
    def_use: DefineUseAnalysis
    by_def: dict[Definition, _Context]
    by_expr: dict[Expr, _Context]
    ret_ctx: _Context

    def __init__(self, func: FuncDef, def_use: DefineUseAnalysis):
        self.func = func
        self.def_use = def_use
        self.by_def = {}
        self.by_expr = {}
        self.ret_ctx = None

    def infer(self):
        self._visit_function(self.func, None)
        return ContextAnalysis(self.ret_ctx, self.by_def, self.by_expr)

    def _visit_binding(self, site: DefSite, target: Id | TupleBinding, ctx: _Context):
        match target:
            case NamedId():
                d = self.def_use.find_def_from_site(target, site)
                self.by_def[d] = ctx
            case UnderscoreId():
                pass
            case TupleBinding():
                for elt in target.elts:
                    self._visit_binding(site, elt, ctx)
            case _:
                raise RuntimeError(f'unreachable: {target}')

    def _visit_assign(self, stmt: Assign, ctx: _Context):
        self._visit_expr(stmt.expr, ctx)
        self._visit_binding(stmt, stmt.binding, ctx)
        return ctx

    def _visit_indexed_assign(self, stmt: IndexedAssign, ctx: _Context):
        for s in stmt.slices:
            self._visit_expr(s, ctx)
        self._visit_expr(stmt.expr, ctx)
        return ctx

    def _visit_if1(self, stmt: If1Stmt, ctx: _Context):
        self._visit_expr(stmt.cond, ctx)
        self._visit_block(stmt.body, ctx)
        return ctx

    def _visit_if(self, stmt: IfStmt, ctx: _Context):
        self._visit_expr(stmt.cond, ctx)
        self._visit_block(stmt.ift, ctx)
        self._visit_block(stmt.iff, ctx)
        return ctx

    def _visit_while(self, stmt: WhileStmt, ctx: _Context):
        self._visit_expr(stmt.cond, ctx)
        self._visit_block(stmt.body, ctx)
        return ctx

    def _visit_for(self, stmt: ForStmt, ctx: _Context):
        self._visit_expr(stmt.iterable, ctx)
        self._visit_binding(stmt, stmt.target, ctx)
        self._visit_block(stmt.body, ctx)
        return ctx

    def _visit_context(self, stmt: ContextStmt, ctx: _Context):
        match stmt.ctx:
            case ForeignVal():
                if isinstance(stmt.ctx.val, Context):
                    body_ctx = stmt.ctx.val
                else:
                    body_ctx = None
            case Var() | ForeignAttribute() | ContextExpr():
                body_ctx = None
            case _:
                raise RuntimeError(f'unreachable: {stmt.ctx}')

        self._visit_block(stmt.body, body_ctx)
        return ctx

    def _visit_assert(self, stmt: AssertStmt, ctx: _Context):
        self._visit_expr(stmt.test, ctx)
        return ctx

    def _visit_effect(self, stmt: EffectStmt, ctx: _Context):
        self._visit_expr(stmt.expr, ctx)
        return ctx

    def _visit_return(self, stmt: ReturnStmt, ctx: _Context):
        self._visit_expr(stmt.expr, ctx)
        self.ret_ctx = ctx
        return ctx

    def _visit_block(self, block: StmtBlock, ctx: _Context):
        for stmt in block.stmts:
            ctx = self._visit_statement(stmt, ctx)

    def _visit_function(self, func: FuncDef, ctx: None):
        # function can have an overriding context
        match func.ctx:
            case None:
                body_ctx: _Context = DEFAULT
            case FPCoreContext():
                body_ctx = None
            case _:
                body_ctx = func.ctx

        self._visit_block(func.body, body_ctx)
        return self.ret_ctx

    def _visit_expr(self, expr: Expr, ctx: _Context):
        self.by_expr[expr] = ctx
        super()._visit_expr(expr, ctx)


class ContextInfer:
    """
    Context inference.

    Like type checking but for rounding contexts.
    Most rounding contexts are statically known, so
    we can assign every statement (or expression) a rounding context
    if it can be determined.
    """

    @staticmethod
    def infer(func: FuncDef):
        """
        Performs rounding context inference.

        Produces a map from definition sites to their rounding contexts.
        """
        if not isinstance(func, FuncDef):
            raise TypeError(f'expected a \'FuncDef\', got {func}')

        def_use = DefineUse.analyze(func)
        inst = _ContextInferInstance(func, def_use)
        return inst.infer()
