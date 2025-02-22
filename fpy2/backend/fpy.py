"""
Compilation from FPy IR to FPy AST.

Useful for source-to-source transformations.
"""

from ..ir import *

from ..frontend import fpyast as ast
from ..frontend.codegen import (
    _unary_table,
    _binary_table,
    _ternary_table,
    _nary_table
)
from ..frontend.syntax_check import SyntaxCheck

from ..runtime import Function
from .backend import Backend

# reverse operator tables
_unary_rev_table = { v: k for k, v in _unary_table.items() }
_binary_rev_table = { v: k for k, v in _binary_table.items() }
_ternary_rev_table = { v: k for k, v in _ternary_table.items() }
_nary_rev_table = { v: k for k, v in _nary_table.items() }

class _FPyCompilerInstance(ReduceVisitor):
    """Compilation instance from FPy to FPCore"""
    func: FunctionDef
    env: dict[NamedId, NamedId]

    def __init__(self, func: FunctionDef):
        self.func = func
        self.env = {}

    def compile(self) -> ast.FunctionDef:
        return self._visit_function(self.func, None)

    def _visit_var(self, e: Var, ctx: None):
        name = self.env.get(e.name, e.name)
        return ast.Var(name, None)

    def _visit_bool(self, e: Bool, ctx: None):
        return ast.Bool(e.val, None)

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

    def _visit_unary_expr(self, e: UnaryExpr, ctx: None):
        cls = type(e)
        if cls not in _unary_rev_table:
            raise NotImplementedError(f'unsupported unary expression {e}')
        kind = _unary_rev_table[cls]
        arg = self._visit_expr(e.children[0], None)
        return ast.UnaryOp(kind, arg, None)

    def _visit_binary_expr(self, e: BinaryExpr, ctx: None):
        cls = type(e)
        if cls not in _binary_rev_table:
            raise NotImplementedError(f'unsupported binary expression {e}')
        kind = _binary_rev_table[cls]
        lhs = self._visit_expr(e.children[0], None)
        rhs = self._visit_expr(e.children[1], None)
        return ast.BinaryOp(kind, lhs, rhs, None)

    def _visit_ternary_expr(self, e: TernaryExpr, ctx: None):
        cls = type(e)
        if cls not in _ternary_rev_table:
            raise NotImplementedError(f'unsupported ternary expression {e}')
        kind = _ternary_rev_table[cls]
        arg0 = self._visit_expr(e.children[0], None)
        arg1 = self._visit_expr(e.children[1], None)
        arg2 = self._visit_expr(e.children[2], None)
        return ast.TernaryOp(kind, arg0, arg1, arg2, None)

    def _visit_nary_expr(self, e: NaryExpr, ctx: None):
        match e:
            case UnaryExpr():
                return self._visit_unary_expr(e, ctx)
            case BinaryExpr():
                return self._visit_binary_expr(e, ctx)
            case TernaryExpr():
                return self._visit_ternary_expr(e, ctx)
            case NaryExpr():
                cls = type(e)
                if cls not in _nary_rev_table:
                    raise NotImplementedError(f'unsupported N-ary expression {e}')
                kind = _nary_rev_table[cls]
                args = [self._visit_expr(arg, None) for arg in e.children]
                return ast.NaryOp(kind, args, None)
            case _:
                raise NotImplementedError(f'unsupported expression {e}')

    def _visit_compare(self, e: Compare, ctx: None):
        args = [self._visit_expr(arg, None) for arg in e.children]
        return ast.Compare(list(e.ops), args, None)

    def _visit_tuple_expr(self, e: TupleExpr, ctx: None):
        args = [self._visit_expr(arg, None) for arg in e.children]
        return ast.TupleExpr(args, None)

    def _visit_tuple_ref(self, e: TupleRef, ctx: None):
        slices = [self._visit_expr(s, None) for s in e.slices]
        value = self._visit_expr(e.value, None)
        return ast.RefExpr(value, slices, None)

    def _visit_tuple_set(self, e: TupleSet, ctx: None):
        raise NotImplementedError('do not call')

    def _visit_comp_expr(self, e: CompExpr, ctx: None):
        iters = [self._visit_expr(i, None) for i in e.iterables]
        elt = self._visit_expr(e.elt, None)
        return ast.CompExpr(e.vars, iters, elt, None)

    def _visit_if_expr(self, e: IfExpr, ctx: None):
        cond = self._visit_expr(e.cond, None)
        ift = self._visit_expr(e.ift, None)
        iff = self._visit_expr(e.iff, None)
        return ast.IfExpr(cond, ift, iff, None)

    def _visit_var_assign(self, stmt: VarAssign, ctx: None):
        # TODO: typing annotation
        e = self._visit_expr(stmt.expr, None)
        if isinstance(stmt.var, NamedId):
            if stmt.var in self.env:
                name = self.env[stmt.var]
            else:
                name = stmt.var
                self.env[stmt.var] = name
            return ast.VarAssign(name, e, None, None)
        else:
            return ast.VarAssign(stmt.var, e, None, None)

    def _visit_tuple_binding(self, vars: TupleBinding):
        new_vars: list[Id | ast.TupleBinding] = []
        for name in vars:
            if isinstance(name, NamedId):
                if name in self.env:
                    name = self.env[name]
                else:
                    self.env[name] = name
                new_vars.append(name)
            elif isinstance(name, Id):
                new_vars.append(name)
            elif isinstance(name, TupleBinding):
                new_vars.append(self._visit_tuple_binding(name))
            else:
                raise NotImplementedError('unexpected tuple identifier', name)
        return ast.TupleBinding(new_vars, None)

    def _visit_tuple_assign(self, stmt: TupleAssign, ctx: None):
        binding = self._visit_tuple_binding(stmt.binding)
        expr = self._visit_expr(stmt.expr, ctx)
        return ast.TupleAssign(binding, expr, None)

    def _visit_ref_assign(self, stmt: RefAssign, ctx: None):
        var = self.env.get(stmt.var, stmt.var)
        slices = [self._visit_expr(s, ctx) for s in stmt.slices]
        value = self._visit_expr(stmt.expr, ctx)
        return ast.RefAssign(var, slices, value, None)

    def _visit_if1_stmt(self, stmt: If1Stmt, ctx: None):
        # setting `phi.name` to be the canonical name
        for phi in stmt.phis:
            name = self.env.get(phi.lhs, phi.lhs)
            self.env[phi.rhs] = name
            self.env[phi.name] = name

        cond = self._visit_expr(stmt.cond, None)
        body = self._visit_block(stmt.body, None)
        return ast.IfStmt(cond, body, None, None)

    def _visit_if_stmt(self, stmt: IfStmt, ctx: None):
        # setting `phi.name` to be the canonical name
        for phi in stmt.phis:
            if phi.lhs in self.env:
                # `phi.lhs` is already in scope
                name = self.env.get(phi.lhs, phi.lhs)
                self.env[phi.rhs] = name
                self.env[phi.name] = name
            elif phi.rhs in self.env:
                # `phi.rhs` is already in scope
                name = self.env.get(phi.rhs, phi.rhs)
                self.env[phi.lhs] = name
                self.env[phi.name] = name
            else:
                # definitions on both paths
                name = self.env.get(phi.name, phi.name)
                self.env[phi.lhs] = name
                self.env[phi.rhs] = name

        cond = self._visit_expr(stmt.cond, None)
        ift = self._visit_block(stmt.ift, None)
        iff = self._visit_block(stmt.iff, None)
        return ast.IfStmt(cond, ift, iff, None)

    def _visit_while_stmt(self, stmt: WhileStmt, ctx: None):
        # setting `phi.name` to be the canonical name
        for phi in stmt.phis:
            name = self.env.get(phi.lhs, phi.lhs)
            self.env[phi.rhs] = name
            self.env[phi.name] = name

        cond = self._visit_expr(stmt.cond, None)
        body = self._visit_block(stmt.body, None)
        return ast.WhileStmt(cond, body, None)

    def _visit_for_stmt(self, stmt: ForStmt, ctx: None):
        # setting `phi.name` to be the canonical name
        for phi in stmt.phis:
            name = self.env.get(phi.lhs, phi.lhs)
            self.env[phi.rhs] = name
            self.env[phi.name] = name

        iterable = self._visit_expr(stmt.iterable, None)
        body = self._visit_block(stmt.body, None)
        return ast.ForStmt(stmt.var, iterable, body, None)

    def _visit_context(self, stmt: ContextStmt, ctx: None):
        body = self._visit_block(stmt.body, None)
        return ast.ContextStmt(stmt.name, dict(stmt.props), body, None)

    def _visit_assert(self, stmt: AssertStmt, ctx: None):
        e = self._visit_expr(stmt.test, None)
        return ast.AssertStmt(e, stmt.msg, None)

    def _visit_return(self, stmt: Return, ctx: None):
        e = self._visit_expr(stmt.expr, None)
        return ast.Return(e, None)

    def _visit_phis(self, phis: list[PhiNode], lctx: None, rctx: None):
        raise NotImplementedError('do not call')

    def _visit_loop_phis(self, phis: list[PhiNode], lctx: None, rctx: None):
        raise NotImplementedError('do not call')

    def _visit_block(self, block: Block, ctx: None):
        stmts = [self._visit_statement(s, None) for s in block.stmts]
        return ast.Block(stmts)

    def _visit_props(self, props: dict[str, Any]):
        new_props: dict[str, Any] = {}
        for k, v in props.items():
            if isinstance(v, Expr):
                new_props[k] = self._visit_expr(v, None)
            else:
                new_props[k] = v
        return new_props

    def _visit_function(self, func: FunctionDef, ctx: None):
        args: list[ast.Argument] = []
        for arg in func.args:
            # TODO: translate typing annotation
            if isinstance(arg.name, NamedId):
                self.env[arg.name] = arg.name
            args.append(ast.Argument(arg.name, None, None))

        body = self._visit_block(func.body, None)
        stx = ast.FunctionDef(func.name, args, body, None)
        stx.ctx = self._visit_props(func.ctx)
        return stx

    # override for typing hint:
    def _visit_expr(self, e: Expr, ctx: None) -> ast.Expr:
        return super()._visit_expr(e, None)
    
    # override for typing hint:
    def _visit_statement(self, stmt: Stmt, ctx: None) -> ast.Stmt:
        return super()._visit_statement(stmt, None)


class FPYCompiler(Backend):
    """Compiler from FPy IR to FPy"""

    def compile(self, func: Function):
        ast = _FPyCompilerInstance(func.ir).compile()
        SyntaxCheck.analyze(ast)
        return ast

