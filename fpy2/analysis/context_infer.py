"""
Context inference.
"""

from typing import TypeAlias

from ..ast import *
from ..fpc_context import FPCoreContext
from ..number import Context
from ..utils import DEFAULT, DefaultOr
from .define_use import DefineUse, DefineUseAnalysis, Definition, DefineUnion, DefSite


_Context: TypeAlias = DefaultOr[Context | None]


class _ContextInferInstance(Visitor):
    """
    Context inference instance.

    This visitor traverses the function and infers rounding contexts
    for each definition site.
    """

    func: FuncDef
    def_use: DefineUseAnalysis
    contexts: dict[Definition, _Context]

    def __init__(self, func: FuncDef, def_use: DefineUseAnalysis):
        self.func = func
        self.def_use = def_use
        self.contexts = {}

    def infer(self):
        self._visit_function(self.func, None)
        return self.contexts

    def _visit_var(self, e: Var, ctx: _Context):
        raise NotImplementedError

    def _visit_bool(self, e: BoolVal, ctx: _Context):
        raise NotImplementedError

    def _visit_foreign(self, e: ForeignVal, ctx: _Context):
        raise NotImplementedError

    def _visit_decnum(self, e: Decnum, ctx: _Context):
        raise NotImplementedError

    def _visit_hexnum(self, e: Hexnum, ctx: _Context):
        raise NotImplementedError

    def _visit_integer(self, e: Integer, ctx: _Context):
        raise NotImplementedError

    def _visit_rational(self, e: Rational, ctx: _Context):
        raise NotImplementedError

    def _visit_digits(self, e: Digits, ctx: _Context):
        raise NotImplementedError

    def _visit_nullaryop(self, e: NullaryOp, ctx: _Context):
        raise NotImplementedError

    def _visit_unaryop(self, e: UnaryOp, ctx: _Context):
        raise NotImplementedError

    def _visit_binaryop(self, e: BinaryOp, ctx: _Context):
        raise NotImplementedError

    def _visit_ternaryop(self, e: TernaryOp, ctx: _Context):
        raise NotImplementedError

    def _visit_naryop(self, e: NaryOp, ctx: _Context):
        raise NotImplementedError

    def _visit_compare(self, e: Compare, ctx: _Context):
        raise NotImplementedError

    def _visit_call(self, e: Call, ctx: _Context):
        raise NotImplementedError

    def _visit_tuple_expr(self, e: TupleExpr, ctx: _Context):
        raise NotImplementedError

    def _visit_list_expr(self, e: ListExpr, ctx: _Context):
        raise NotImplementedError

    def _visit_list_comp(self, e: ListComp, ctx: _Context):
        raise NotImplementedError

    def _visit_list_ref(self, e: ListRef, ctx: _Context):
        raise NotImplementedError

    def _visit_list_slice(self, e: ListSlice, ctx: _Context):
        raise NotImplementedError

    def _visit_list_set(self, e: ListSet, ctx: _Context):
        raise NotImplementedError

    def _visit_if_expr(self, e: IfExpr, ctx: _Context):
        raise NotImplementedError

    def _visit_context_expr(self, e: ContextExpr, ctx: _Context):
        raise NotImplementedError

    def _visit_binding(self, site: DefSite, target: Id | TupleBinding, ctx: _Context):
        match target:
            case NamedId():
                d = self.def_use.find_def_from_site(target, site)
                self.contexts[d] = ctx
            case UnderscoreId():
                pass
            case TupleBinding():
                for elt in target.elts:
                    self._visit_binding(site, elt, ctx)
            case _:
                raise RuntimeError(f'unreachable: {target}')

    def _visit_assign(self, stmt: Assign, ctx: _Context):
        self._visit_binding(stmt, stmt.binding, ctx)
        return ctx

    def _visit_indexed_assign(self, stmt: IndexedAssign, ctx: _Context):
        return ctx

    def _visit_if1(self, stmt: If1Stmt, ctx: _Context):
        self._visit_block(stmt.body, ctx)
        return ctx

    def _visit_if(self, stmt: IfStmt, ctx: _Context):
        self._visit_block(stmt.ift, ctx)
        self._visit_block(stmt.iff, ctx)
        return ctx

    def _visit_while(self, stmt: WhileStmt, ctx: _Context):
        self._visit_block(stmt.body, ctx)
        return ctx

    def _visit_for(self, stmt: ForStmt, ctx: _Context):
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
        return ctx

    def _visit_effect(self, stmt: EffectStmt, ctx: _Context):
        return ctx

    def _visit_return(self, stmt: ReturnStmt, ctx: _Context):
        return ctx

    def _visit_block(self, block: StmtBlock, ctx: _Context):
        for stmt in block.stmts:
            ctx = self._visit_statement(stmt, ctx)

    def _visit_function(self, func: FuncDef, ctx: None):
        # function can have an overriding context
        match func.ctx:
            case None:
                self._visit_block(func.body, DEFAULT)
            case FPCoreContext():
                self._visit_block(func.body, None)
            case _:
                self._visit_block(func.body, func.ctx)


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
        info = inst.infer()
        print(func.name, info)
