"""
Compilation from FPy IR to FPy AST.

Useful for source-to-source transformations.
"""

from ..ir import *

from ..ast import fpyast as ast
from ..ast.syntax_check import SyntaxCheck
from ..function import Function
from ..transform import UnSSA

from ..ir.codegen import (
    _unary_table,
    _binary_table,
    _ternary_table,
    _nary_table
)

from .backend import Backend


# reverse operator tables
_unary_rev_table = { v: k for k, v in _unary_table.items() }
_binary_rev_table = { v: k for k, v in _binary_table.items() }
_ternary_rev_table = { v: k for k, v in _ternary_table.items() }
_nary_rev_table = { v: k for k, v in _nary_table.items() }

class _FPyCompilerInstance(ReduceVisitor):
    """Compilation instance from FPy to FPCore"""
    func: FuncDef

    def __init__(self, func: FuncDef):
        self.func = func

    def compile(self) -> ast.FuncDef:
        return self._visit_function(self.func, None)

    def _visit_var(self, e: Var, ctx: None):
        return ast.Var(e.name, None)

    def _visit_bool(self, e: BoolVal, ctx: None):
        return ast.BoolVal(e.val, None)

    def _visit_foreign(self, e: ForeignVal, ctx: None):
        return ast.ForeignVal(e.val, None)

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
        return ast.TupleRef(value, slices, None)

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

    def _visit_simple_assign(self, stmt: SimpleAssign, ctx: None):
        # TODO: typing annotation
        e = self._visit_expr(stmt.expr, None)
        return ast.SimpleAssign(stmt.var, e, None, None)

    def _visit_tuple_binding(self, vars: TupleBinding):
        new_vars: list[Id | ast.TupleBinding] = []
        for name in vars:
            match name:
                case Id():
                    new_vars.append(name)
                case TupleBinding():
                    bind = self._visit_tuple_binding(name)
                    new_vars.append(bind)
                case _:
                    raise NotImplementedError('unexpected tuple identifier', name)
        return ast.TupleBinding(new_vars, None)

    def _visit_tuple_unpack(self, stmt: TupleUnpack, ctx: None):
        binding = self._visit_tuple_binding(stmt.binding)
        expr = self._visit_expr(stmt.expr, ctx)
        return ast.TupleUnpack(binding, expr, None)

    def _visit_index_assign(self, stmt: IndexAssign, ctx: None):
        slices = [self._visit_expr(s, ctx) for s in stmt.slices]
        value = self._visit_expr(stmt.expr, ctx)
        return ast.IndexAssign(stmt.var, slices, value, None)

    def _visit_if1(self, stmt: If1Stmt, ctx: None):
        # check that phis are empty
        if stmt.phis != []:
            raise ValueError(f'expected no phis in statement: {stmt}')

        cond = self._visit_expr(stmt.cond, None)
        body = self._visit_block(stmt.body, None)
        return ast.If1Stmt(cond, body, None)

    def _visit_if(self, stmt: IfStmt, ctx: None):
        # check that phis are empty
        if stmt.phis != []:
            raise ValueError(f'expected no phis in statement: {stmt}')

        cond = self._visit_expr(stmt.cond, None)
        ift = self._visit_block(stmt.ift, None)
        iff = self._visit_block(stmt.iff, None)
        return ast.IfStmt(cond, ift, iff, None)

    def _visit_while(self, stmt: WhileStmt, ctx: None):
        # check that phis are empty
        if stmt.phis != []:
            raise ValueError(f'expected no phis in statement: {stmt}')

        cond = self._visit_expr(stmt.cond, None)
        body = self._visit_block(stmt.body, None)
        return ast.WhileStmt(cond, body, None)

    def _visit_for(self, stmt: ForStmt, ctx: None):
        # check that phis are empty
        if stmt.phis != []:
            raise ValueError(f'expected no phis in statement: {stmt}')

        iterable = self._visit_expr(stmt.iterable, None)
        body = self._visit_block(stmt.body, None)
        return ast.ForStmt(stmt.var, iterable, body, None)

    def _visit_context_expr(self, e: ContextExpr, ctx: Any):
        match e.ctor:
            case Var():
                ctor = self._visit_var(e.ctor, ctx)
            case ForeignAttribute():
                ctor = ast.ForeignAttribute(e.ctor.name, e.ctor.attrs, None)
            case _:
                raise RuntimeError('unreachable', e.ctor)

        args: list[ast.Expr | ast.ForeignAttribute] = []
        for arg in e.args:
            match arg:
                case ForeignAttribute():
                    args.append(ast.ForeignAttribute(arg.name, arg.attrs, None))
                case _:
                    args.append(self._visit_expr(arg, ctx))

        kwargs: list[tuple[str, ast.Expr | ast.ForeignAttribute]] = []
        for k, v in e.kwargs:
            match v:
                case ForeignAttribute():
                    kwargs.append((k, ast.ForeignAttribute(v.name, v.attrs, None)))
                case _:
                    kwargs.append((k, self._visit_expr(v, ctx)))

        # TODO: kwargs
        return ast.ContextExpr(ctor, args, kwargs, None)

    def _visit_context(self, stmt: ContextStmt, ctx: None):
        match stmt.ctx:
            case Var():
                context = self._visit_var(stmt.ctx, ctx)
            case ContextExpr():
                context = self._visit_context_expr(stmt.ctx, ctx)
            case ForeignVal():
                context = ast.ForeignVal(stmt.ctx.val, None)
            case _:
                raise RuntimeError('unreachable', stmt.ctx)
        body = self._visit_block(stmt.body, None)
        return ast.ContextStmt(stmt.name, context, body, None)

    def _visit_assert(self, stmt: AssertStmt, ctx: None):
        e = self._visit_expr(stmt.test, None)
        return ast.AssertStmt(e, stmt.msg, None)

    def _visit_effect(self, stmt: EffectStmt, ctx: None):
        e = self._visit_expr(stmt.expr, None)
        return ast.EffectStmt(e, None)

    def _visit_return(self, stmt: ReturnStmt, ctx: None):
        e = self._visit_expr(stmt.expr, None)
        return ast.ReturnStmt(e, None)

    def _visit_block(self, block: StmtBlock, ctx: None):
        stmts = [self._visit_statement(s, None) for s in block.stmts]
        return ast.StmtBlock(stmts)

    def _visit_props(self, props: dict[str, Any]):
        new_props: dict[str, Any] = {}
        for k, v in props.items():
            if isinstance(v, Expr):
                new_props[k] = self._visit_expr(v, None)
            else:
                new_props[k] = v
        return new_props

    def _visit_function(self, func: FuncDef, ctx: None):
        args: list[ast.Argument] = []
        for arg in func.args:
            # TODO: translate typing annotation
            args.append(ast.Argument(arg.name, None, None))

        body = self._visit_block(func.body, None)
        stx = ast.FuncDef(func.name, args, body, None)
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

    def compile(self, func: FuncDef):
        if not isinstance(func, FuncDef):
            raise TypeError(f'expected \'FuncDef\', got {func}')

        func = UnSSA.apply(func)
        ast = _FPyCompilerInstance(func).compile()
        free_vars = set([str(v) for v in func.free_vars])
        SyntaxCheck.analyze(ast, free_vars=free_vars)
        return ast

