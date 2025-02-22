"""Pretty printing of FPy IRs"""

from typing import Optional

from .ir import *
from .types import *
from .visitor import BaseVisitor

class _IndentCtx:
    _indent: str
    _level: int

    def __init__(self, indent: str, level: int):
        self._indent = indent
        self._level = level

    def __str__(self):
        return self._indent * self._level

    def indent(self):
        return _IndentCtx(self._indent, self._level + 1)

class _FormatterInstance(BaseVisitor):
    """Single-instance visitor for pretty printing FPy IRs"""
    ir: IR
    fmt: str

    def __init__(self, ir: IR):
        self.ir = ir
        self.fmt = ''

    def apply(self) -> str:
        ctx = _IndentCtx('    ', 0)
        match self.ir:
            case Expr():
                self.fmt = self._visit_expr(self.ir, ctx)
            case Stmt():
                self._visit_statement(self.ir, ctx)
            case Block():
                self._visit_block(self.ir, ctx)
            case FunctionDef():
                self._visit_function(self.ir, ctx)
            case _:
                raise NotImplementedError('unsupported IR node', self.ir)
        return self.fmt.strip()

    def _add_line(self, line: str, ctx: _IndentCtx):
        self.fmt += str(ctx) + line + '\n'

    def _format_type(self, ty: IRType) -> str:
        match ty:
            case AnyType():
                return 'Any'
            case RealType():
                return 'Real'
            case _:
                raise NotImplementedError('unexpected', ty)

    def _visit_var(self, e: Var, ctx: _IndentCtx):
        return str(e.name)

    def _visit_bool(self, e: Bool, ctx: Any):
        return str(e.val)

    def _visit_decnum(self, e: Decnum, ctx: _IndentCtx):
        return e.val

    def _visit_hexnum(self, e: Hexnum, ctx: _IndentCtx):
        return e.val

    def _visit_integer(self, e: Integer, ctx: _IndentCtx):
        return str(e.val)

    def _visit_rational(self, e: Rational, ctx: _IndentCtx):
        return f'{e.p}/{e.q}'

    def _visit_constant(self, e: Constant, ctx: _IndentCtx):
        return e.val

    def _visit_digits(self, e: Digits, ctx: _IndentCtx):
        return f'digits({e.m}, {e.e}, {e.b})'

    def _visit_unknown(self, e: UnknownCall, ctx: _IndentCtx):
        args = [self._visit_expr(arg, ctx) for arg in e.children]
        return f'{e.name}({", ".join(args)})'

    def _visit_nary_expr(self, e: NaryExpr, ctx: _IndentCtx):
        match e:
            case Add():
                lhs = self._visit_expr(e.children[0], ctx)
                rhs = self._visit_expr(e.children[1], ctx)
                return f'{lhs} + {rhs}'
            case Sub():
                lhs = self._visit_expr(e.children[0], ctx)
                rhs = self._visit_expr(e.children[1], ctx)
                return f'{lhs} - {rhs}'
            case Mul():
                lhs = self._visit_expr(e.children[0], ctx)
                rhs = self._visit_expr(e.children[1], ctx)
                return f'{lhs} * {rhs}'
            case Div():
                lhs = self._visit_expr(e.children[0], ctx)
                rhs = self._visit_expr(e.children[1], ctx)
                return f'{lhs} / {rhs}'
            case _:
                args = [self._visit_expr(arg, ctx) for arg in e.children]
                return f'{e.name}({", ".join(args)})'

    def _visit_compare(self, e: Compare, ctx: _IndentCtx):
        first = self._visit_expr(e.children[0], ctx)
        rest = [self._visit_expr(arg, ctx) for arg in e.children[1:]]
        s = ' '.join(f'{op.symbol()} {arg}' for op, arg in zip(e.ops, rest))
        return f'{first} {s}'

    def _visit_tuple_expr(self, e: TupleExpr, ctx: _IndentCtx):
        elts = [self._visit_expr(elt, ctx) for elt in e.children]
        return f'({", ".join(elts)})'

    def _visit_tuple_ref(self, e: TupleRef, ctx: _IndentCtx):
        value = self._visit_expr(e.value, ctx)
        slices = [self._visit_expr(slice, ctx) for slice in e.slices]
        ref_str = ''.join(f'[{slice}]' for slice in slices)
        return f'{value}{ref_str}'

    def _visit_tuple_set(self, e: TupleSet, ctx: _IndentCtx):
        array = self._visit_expr(e.array, ctx)
        slices = [self._visit_expr(s, ctx) for s in e.slices]
        value = self._visit_expr(e.value, ctx)
        arg_str = ' '.join([array] + slices + [value])
        return f'set({arg_str})'

    def _visit_comp_expr(self, e: CompExpr, ctx: _IndentCtx):
        elt = self._visit_expr(e.elt, ctx)
        iterables = [self._visit_expr(iterable, ctx) for iterable in e.iterables]
        s = ' '.join(f'for {str(var)} in {iterable}' for var, iterable in zip(e.vars, iterables))
        return f'[{elt} {s}]'

    def _visit_if_expr(self, e: IfExpr, ctx: _IndentCtx):
        cond = self._visit_expr(e.cond, ctx)
        ift = self._visit_expr(e.ift, ctx)
        iff = self._visit_expr(e.iff, ctx)
        return f'{ift} if {cond} else {iff}'

    def _visit_var_assign(self, stmt: VarAssign, ctx: _IndentCtx):
        val = self._visit_expr(stmt.expr, ctx)
        self._add_line(f'{str(stmt.var)}: {self._format_type(stmt.ty)} = {val}', ctx)

    def _visit_tuple_binding(self, vars: TupleBinding) -> str:
        ss: list[str] = []
        for var in vars:
            match var:
                case Id():
                    ss.append(str(var))
                case TupleBinding():
                    s = self._visit_tuple_binding(var)
                    ss.append(f'({s})')
                case _:
                    raise NotImplementedError('unreachable', var)
        return ', '.join(ss)

    def _visit_tuple_assign(self, stmt: TupleAssign, ctx: _IndentCtx):
        val = self._visit_expr(stmt.expr, ctx)
        vars = self._visit_tuple_binding(stmt.binding)
        self._add_line(f'{vars} = {val} : {self._format_type(stmt.ty)}', ctx)

    def _visit_ref_assign(self, stmt: RefAssign, ctx: _IndentCtx):
        slices = [self._visit_expr(s, ctx) for s in stmt.slices]
        expr = self._visit_expr(stmt.expr, ctx)
        ref_str = ''.join(f'[{slice}]' for slice in slices)
        self._add_line(f'{str(stmt.var)}{ref_str} = {expr}', ctx)

    def _visit_if1_stmt(self, stmt: If1Stmt, ctx: _IndentCtx):
        cond = self._visit_expr(stmt.cond, ctx)
        self._add_line(f'if {cond}:', ctx)
        self._visit_block(stmt.body, ctx.indent())
        self._visit_phis(stmt.phis, ctx, ctx)

    def _visit_if_stmt(self, stmt: IfStmt, ctx: _IndentCtx):
        cond = self._visit_expr(stmt.cond, ctx)
        self._add_line(f'if {cond}:', ctx)
        self._visit_block(stmt.ift, ctx.indent())
        self._add_line('else:', ctx)
        self._visit_block(stmt.iff, ctx.indent())
        self._visit_phis(stmt.phis, ctx, ctx)

    def _visit_while_stmt(self, stmt: WhileStmt, ctx: _IndentCtx):
        self._visit_loop_phis(stmt.phis, ctx, None)
        cond = self._visit_expr(stmt.cond, ctx)
        self._add_line(f'while {cond}:', ctx)
        self._visit_block(stmt.body, ctx.indent())

    def _visit_for_stmt(self, stmt: ForStmt, ctx: _IndentCtx):
        iterable = self._visit_expr(stmt.iterable, ctx)
        self._visit_loop_phis(stmt.phis, ctx, None)
        self._add_line(f'for {str(stmt.var)} in {iterable}:', ctx)
        self._visit_block(stmt.body, ctx.indent())

    def _visit_context(self, stmt: ContextStmt, ctx: _IndentCtx):
        # TODO: format data
        props = ', '.join(f'{k}={v}' for k, v in stmt.props.items())
        self._add_line(f'with Context({props}):', ctx)
        self._visit_block(stmt.body, ctx.indent())

    def _visit_assert(self, stmt: AssertStmt, ctx: _IndentCtx):
        test = self._visit_expr(stmt.test, ctx)
        if stmt.msg is None:
            self._add_line(f'assert {test}', ctx)
        else:
            self._add_line(f'assert {test}, {stmt.msg}', ctx)

    def _visit_return(self, stmt: Return, ctx: _IndentCtx):
        s = self._visit_expr(stmt.expr, ctx)
        self._add_line(f'return {s}', ctx)

    def _visit_phis(self, phis: list[PhiNode], lctx: _IndentCtx, rctx: _IndentCtx):
        for phi in phis:
            decl = f'{str(phi.name)}: {self._format_type(phi.ty)}'
            e = f'phi({str(phi.lhs)}, {str(phi.rhs)})'
            self._add_line(f'{decl} = {e}', lctx)

    def _visit_loop_phis(self, phis: list[PhiNode], lctx: _IndentCtx, rctx: Optional[_IndentCtx]):
        for phi in phis:
            decl = f'{str(phi.name)}: {self._format_type(phi.ty)}'
            e = f'phi({str(phi.lhs)}, {str(phi.rhs)})'
            self._add_line(f'{decl} = {e}', lctx)

    def _visit_block(self, block: Block, ctx: _IndentCtx):
        for stmt in block.stmts:
            self._visit_statement(stmt, ctx)

    def _format_decorator(self, props: dict[str, str], ctx: _IndentCtx):
        if len(props) == 0:
            self._add_line('@fpy_ir', ctx)
        elif len(props) == 1:
            k, *_ = tuple(props.keys())
            v = props[k]
            self._add_line(f'@fpy_ir({k}={v})', ctx)
        else:
            self._add_line('@fpy_ir(', ctx)
            for k, v in props.items():
                self._add_line(f'{k}={v},', ctx.indent())
            self._add_line(')', ctx)

    def _visit_function(self, func: FunctionDef, ctx: _IndentCtx):
        arg_strs: list[str] = []
        for arg in func.args:
            arg_str = f'{str(arg.name)}: {self._format_type(arg.ty)}'
            arg_strs.append(arg_str)

        arg_str = ', '.join(arg_strs)
        self._format_decorator(func.ctx, ctx)
        self._add_line(f'def {func.name}({arg_str}):', ctx)
        self._visit_block(func.body, ctx.indent())

    # override for typing hint
    def _visit_expr(self, e: Expr, ctx: _IndentCtx) -> str:
        return super()._visit_expr(e, ctx)
    
    # override for typing hint
    def _visit_statement(self, stmt: Stmt, ctx: _IndentCtx) -> None:
        return super()._visit_statement(stmt, ctx)


class Formatter(BaseFormatter):
    """"Pretty printer for FPy IRs"""

    def format(self, ir: IR) -> str:
        """Pretty print the given AST"""
        return _FormatterInstance(ir).apply()
