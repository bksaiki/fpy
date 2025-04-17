"""Live variable analysis on the FPy IR"""

from typing import TypeAlias

from ..ir import *

_RetType: TypeAlias = set[NamedId]

class _LiveVars(ReduceVisitor):
    """Single-instance of the live variable analysis."""
    e: FunctionDef | Block | Stmt | Expr

    def __init__(self, e: FunctionDef | Block | Stmt | Expr):
        self.e = e

    def analyze(self):
        """Runs live-variable analysis on `self.e`."""
        match self.e:
            case _:
                raise RuntimeError('unreachable')

    def _visit_var(self, e: Var, ctx: None) -> _RetType:
        return { e.name }

    def _visit_bool(self, e: Bool, ctx: None) -> _RetType:
        return set()

    def _visit_decnum(self, e: Decnum, ctx: None) -> _RetType:
        return set()

    def _visit_hexnum(self, e: Hexnum, ctx: None) -> _RetType:
        return set()

    def _visit_integer(self, e: Integer, ctx: None) -> _RetType:
        return set()

    def _visit_rational(self, e: Rational, ctx: None) -> _RetType:
        return set()

    def _visit_constant(self, e: Constant, ctx: None) -> _RetType:
        return set()

    def _visit_digits(self, e: Digits, ctx: None) -> _RetType:
        return set()

    def _visit_unknown(self, e: UnknownCall, ctx: None) -> _RetType:
        fvs = set()
        for arg in e.children:
            fvs.update(self._visit_expr(arg, ctx))
        return fvs

    def _visit_nary_expr(self, e: NaryExpr, ctx: None) -> _RetType:
        fvs = set()
        for arg in e.children:
            fvs.update(self._visit_expr(arg, ctx))
        return fvs

    def _visit_compare(self, e: Compare, ctx: None) -> _RetType:
        raise NotImplementedError

    def _visit_tuple_expr(self, e: TupleExpr, ctx: None) -> _RetType:
        raise NotImplementedError

    def _visit_tuple_ref(self, e: TupleRef, ctx: None) -> _RetType:
        raise NotImplementedError

    def _visit_tuple_set(self, e: TupleSet, ctx: None) -> _RetType:
        raise NotImplementedError

    def _visit_comp_expr(self, e: CompExpr, ctx: None) -> _RetType:
        raise NotImplementedError

    def _visit_if_expr(self, e: IfExpr, ctx: None) -> _RetType:
        raise NotImplementedError

    def _visit_var_assign(self, stmt: VarAssign, ctx: None) -> _RetType:
        raise NotImplementedError

    def _visit_tuple_assign(self, stmt: TupleAssign, ctx: None) -> _RetType:
        raise NotImplementedError

    def _visit_ref_assign(self, stmt: RefAssign, ctx: None) -> _RetType:
        raise NotImplementedError

    def _visit_if1_stmt(self, stmt: If1Stmt, ctx: None) -> _RetType:
        raise NotImplementedError

    def _visit_if_stmt(self, stmt: IfStmt, ctx: None) -> _RetType:
        raise NotImplementedError

    def _visit_while_stmt(self, stmt: WhileStmt, ctx: None) -> _RetType:
        raise NotImplementedError

    def _visit_for_stmt(self, stmt: ForStmt, ctx: None) -> _RetType:
        raise NotImplementedError

    def _visit_context(self, stmt: ContextStmt, ctx: None) -> _RetType:
        raise NotImplementedError

    def _visit_assert(self, stmt: AssertStmt, ctx: None) -> _RetType:
        raise NotImplementedError

    def _visit_effect(self, stmt: EffectStmt, ctx: None) -> _RetType:
        raise NotImplementedError

    def _visit_return(self, stmt: Return, ctx: None) -> _RetType:
        raise NotImplementedError

    def _visit_phis(self, phis: list[PhiNode], lctx: None, rctx: None) -> _RetType:
        raise NotImplementedError

    def _visit_loop_phis(self, phis: list[PhiNode], lctx: None, rctx: None) -> _RetType:
        raise NotImplementedError

    def _visit_block(self, block: Block, ctx: None) -> _RetType:
        raise NotImplementedError

    def _visit_function(self, func: FunctionDef, ctx: None) -> _RetType:
        raise NotImplementedError
    
    # overriden for typing hint
    def _visit_expr(self, e: Expr, ctx: None) -> _RetType:
        return super()._visit_expr(e, ctx)


class LiveVars:
    """Live variable analysis for the FPy AST."""

    @staticmethod
    def analyze(e: FunctionDef | Block | Stmt | Expr):
        """Analyze the live variables in a function."""
        if not isinstance(e, (FunctionDef, Block, Stmt, Expr)):
            raise TypeError(f'Expected FunctionDef, Block, Stmt or Expr, got {type(e)}')
        return _LiveVars(e).analyze()
