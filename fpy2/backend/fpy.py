"""
Compilation from FPy IR to FPy AST.

Useful for source-to-source transformations.
"""

from ..ir import *

from ..frontend import fpyast as ast
from ..runtime import Function

from .backend import Backend

_nary_table = {
    And: ast.NaryOpKind.AND,
    Or: ast.NaryOpKind.OR,
}


class _FPyCompilerInstance(ReduceVisitor):
    """Compilation instance from FPy to FPCore"""
    func: FunctionDef

    def __init__(self, func: FunctionDef):
        self.func = func

    def compile(self) -> ast.FunctionDef:
        f = self._visit_function(self.func, None)
        assert isinstance(f, ast.FunctionDef), 'unexpected result type'
        return f

    def _visit_var(self, e: Var, ctx: None):
        return ast.Var(e.name, None)

    def _visit_decnum(self, e: Decnum, ctx: None):
        return ast.Decnum(e.val, None)

    def _visit_hexnum(self, e: Hexnum, ctx: None):
        return ast.Hexnum(e.val, None)

    def _visit_integer(self, e: Integer, ctx: None):
        return ast.Integer(e.val, None)

    def _visit_rational(self, e: Rational, ctx: None):
        return ast.Rational(e.p, e.q, None)

    def _visit_constant(self, e: Constant, ctx: None):
        return ast.Constant(e.val, None)

    def _visit_digits(self, e: Digits, ctx: None):
        return ast.Digits(e.m, e.e, e.b, None)

    def _visit_unknown(self, e: UnknownCall, ctx: None):
        args = [self._visit_expr(arg, None) for arg in e.children]
        return ast.Call(e.name, args, None)

    def _visit_nary_expr(self, e: NaryExpr, ctx: None):
        if type(e) not in _nary_table:
            raise NotImplementedError(f'unsupported expression {e}')
        kind = _nary_table[type(e)]
        args = [self._visit_expr(arg, None) for arg in e.children]
        return ast.NaryOp(kind, args, None)

    def _visit_compare(self, e: Compare, ctx: None):
        args = [self._visit_expr(arg, None) for arg in e.children]
        return ast.Compare(list(e.ops), args, None)

    def _visit_tuple_expr(self, e: TupleExpr, ctx: None):
        raise NotImplementedError

    def _visit_tuple_ref(self, e: TupleRef, ctx: None):
        raise NotImplementedError

    def _visit_tuple_set(self, e: TupleSet, ctx: None):
        raise NotImplementedError

    def _visit_comp_expr(self, e: CompExpr, ctx: None):
        raise NotImplementedError

    def _visit_if_expr(self, e: IfExpr, ctx: None):
        raise NotImplementedError

    def _visit_var_assign(self, stmt: VarAssign, ctx: None):
        raise NotImplementedError

    def _visit_tuple_assign(self, stmt: TupleAssign, ctx: None):
        raise NotImplementedError

    def _visit_ref_assign(self, stmt: RefAssign, ctx: None):
        raise NotImplementedError

    def _visit_if1_stmt(self, stmt: If1Stmt, ctx: None):
        raise NotImplementedError

    def _visit_if_stmt(self, stmt: IfStmt, ctx: None):
        raise NotImplementedError

    def _visit_while_stmt(self, stmt: WhileStmt, ctx: None):
        raise NotImplementedError

    def _visit_for_stmt(self, stmt: ForStmt, ctx: None):
        raise NotImplementedError

    def _visit_context(self, stmt: ContextStmt, ctx: None):
        raise NotImplementedError

    def _visit_assert(self, stmt: AssertStmt, ctx: None):
        raise NotImplementedError

    def _visit_return(self, stmt: Return, ctx: None):
        raise NotImplementedError

    def _visit_phis(self, phis: list[PhiNode], lctx: None, rctx: None):
        raise NotImplementedError

    def _visit_loop_phis(self, phis: list[PhiNode], lctx: None, rctx: None):
        raise NotImplementedError

    def _visit_block(self, block: Block, ctx: None):
        raise NotImplementedError

    def _visit_function(self, func: FunctionDef, ctx: None):
        raise NotImplementedError


class FPYCompiler(Backend):
    """Compiler from FPy IR to FPy"""

    def compile(self, func: Function):
        return _FPyCompilerInstance(func.ir).compile()

