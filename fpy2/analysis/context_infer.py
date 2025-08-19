"""
Context inference.
"""

from dataclasses import dataclass
from typing import TypeAlias, Iterable

from ..ast import *
from ..fpc_context import FPCoreContext
from ..number import Context
from ..utils import DEFAULT, DefaultOr
from .define_use import DefineUse, DefineUseAnalysis, Definition, DefSite


_BaseContext: TypeAlias = DefaultOr[Context | None]

class TupleContext:
    """Tracks the contexts of tuples"""

    elts: tuple[_BaseContext | 'TupleContext', ...]

    def __init__(self, elts: Iterable[_BaseContext | 'TupleContext']):
        self.elts = tuple(elts)

    def __eq__(self, other):
        return isinstance(other, TupleContext) and self.elts == other.elts

    def __hash__(self):
        return hash(self.elts)

_Context: TypeAlias = DefaultOr[Context | TupleContext | None]

@dataclass(frozen=True)
class ContextAnalysis:
    ret_ctx: _Context
    by_def: dict[Definition, _Context]
    by_expr: dict[Expr, _Context]


class _ContextInferInstance(Visitor):
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

    def _merge(self, a_ctx: _Context, b_ctx: _Context) -> _Context:
        match a_ctx, b_ctx:
            case _, _ if a_ctx is DEFAULT and b_ctx is DEFAULT:
                return DEFAULT
            case Context(), Context():
                if a_ctx.is_equiv(b_ctx):
                    return a_ctx
                else:
                    return None
            case _:
                return None

    def _visit_var(self, e: Var, ctx: _Context):
        d = self.def_use.find_def_from_use(e)
        return self.by_def[d]

    def _visit_bool(self, e: BoolVal, ctx: _Context):
        return ctx

    def _visit_foreign(self, e: ForeignVal, ctx: _Context):
        return None

    def _visit_decnum(self, e: Decnum, ctx: _Context):
        return ctx

    def _visit_hexnum(self, e: Hexnum, ctx: _Context):
        return ctx

    def _visit_integer(self, e: Integer, ctx: _Context):
        return ctx

    def _visit_rational(self, e: Rational, ctx: _Context):
        return ctx

    def _visit_digits(self, e: Digits, ctx: _Context):
        return ctx

    def _visit_nullaryop(self, e: NullaryOp, ctx: _Context):
        return ctx

    def _visit_unaryop(self, e: UnaryOp, ctx: _Context):
        self._visit_expr(e.arg, ctx)
        return ctx

    def _visit_binaryop(self, e: BinaryOp, ctx: _Context):
        self._visit_expr(e.first, ctx)
        self._visit_expr(e.second, ctx)
        return ctx

    def _visit_ternaryop(self, e: TernaryOp, ctx: _Context):
        self._visit_expr(e.first, ctx)
        self._visit_expr(e.second, ctx)
        self._visit_expr(e.third, ctx)
        return ctx

    def _visit_naryop(self, e: NaryOp, ctx: _Context):
        for arg in e.args:
            self._visit_expr(arg, ctx)
        return ctx

    def _visit_compare(self, e: Compare, ctx: _Context):
        for arg in e.args:
            self._visit_expr(arg, ctx)
        return ctx

    def _visit_call(self, e: Call, ctx: _Context):
        raise NotImplementedError

    def _visit_tuple_expr(self, e: TupleExpr, ctx: _Context):
        return TupleContext(self._visit_expr(arg, ctx) for arg in e.args)

    def _visit_list_expr(self, e: ListExpr, ctx: _Context):
        if len(e.args) == 0:
            return ctx
        else:
            elt_ctx = self._visit_expr(e.args[0], ctx)
            for arg in e.args[1:]:
                elt_ctx = self._merge(elt_ctx, self._visit_expr(arg, ctx))
            return elt_ctx

    def _visit_list_comp(self, e: ListComp, ctx: _Context):
        raise NotImplementedError

    def _visit_list_ref(self, e: ListRef, ctx: _Context):
        raise NotImplementedError

    def _visit_list_slice(self, e: ListSlice, ctx: _Context):
        raise NotImplementedError

    def _visit_list_set(self, e: ListSet, ctx: _Context):
        raise NotImplementedError

    def _visit_if_expr(self, e: IfExpr, ctx: _Context):
        self._visit_expr(e.cond, ctx)
        ift_ctx = self._visit_expr(e.ift, ctx)
        iff_ctx = self._visit_expr(e.iff, ctx)
        return self._merge(ift_ctx, iff_ctx)

    def _visit_context_expr(self, e: ContextExpr, ctx: _Context):
        return None

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

    def _visit_expr(self, expr: Expr, ctx: _Context) -> _Context:
        ret_ctx = super()._visit_expr(expr, ctx)
        self.by_expr[expr] = ret_ctx
        return ret_ctx


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
