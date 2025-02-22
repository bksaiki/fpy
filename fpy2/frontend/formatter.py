"""Pretty printing of FPy ASTs"""

from pprint import pformat

from .fpyast import *
from .visitor import AstVisitor

_Ctx = int

class _FormatterInstance(AstVisitor):
    """Single-instance visitor for pretty printing FPy ASTs"""
    ast: Ast
    fmt: str

    def __init__(self, ast: Ast):
        self.ast = ast
        self.fmt = ''

    def apply(self) -> str:
        match self.ast:
            case Expr():
                self.fmt = self._visit_expr(self.ast, 0)
            case Stmt():
                self._visit_statement(self.ast, 0)
            case Block():
                self._visit_block(self.ast, 0)
            case FunctionDef():
                self._visit_function(self.ast, 0)
            case _:
                raise NotImplementedError('unsupported AST node', self.ast)
        return self.fmt.strip()

    def _add_line(self, line: str, indent: int):
        self.fmt += '    ' * indent + line + '\n'

    def _visit_var(self, e: Var, ctx: _Ctx) -> str:
        return str(e.name)

    def _visit_bool(self, e: Bool, ctx: _Ctx):
        return str(e.val)

    def _visit_decnum(self, e: Decnum, ctx: _Ctx):
        return e.val

    def _visit_hexnum(self, e: Hexnum, ctx: _Ctx):
        return f'hexfloat(\'{e.val}\')'

    def _visit_integer(self, e: Integer, ctx: _Ctx):
        return str(e.val)

    def _visit_rational(self, e: Rational, ctx: _Ctx):
        return f'rational({e.p}, {e.q})'

    def _visit_digits(self, e: Digits, ctx: _Ctx):
        return f'digits({e.m}, {e.e}, {e.b})'

    def _visit_constant(self, e: Constant, ctx: _Ctx):
        return e.val

    def _visit_unaryop(self, e: UnaryOp, ctx: _Ctx):
        arg = self._visit_expr(e.arg, ctx)
        match e.op:
            case UnaryOpKind.NEG:
                return f'-{arg}'
            case UnaryOpKind.NOT:
                return f'not {arg}'
            case _:
                return f'{str(e.op)}({arg})'

    def _visit_binaryop(self, e: BinaryOp, ctx: _Ctx):
        lhs = self._visit_expr(e.left, ctx)
        rhs = self._visit_expr(e.right, ctx)
        match e.op:
            case BinaryOpKind.ADD:
                return f'({lhs} + {rhs})'
            case BinaryOpKind.SUB:
                return f'({lhs} - {rhs})'
            case BinaryOpKind.MUL:
                return f'({lhs} * {rhs})'
            case BinaryOpKind.DIV:
                return f'({lhs} / {rhs})'
            case _:
                return f'{str(e.op)}({lhs}, {rhs})'

    def _visit_ternaryop(self, e: TernaryOp, ctx: _Ctx):
        arg0 = self._visit_expr(e.arg0, ctx)
        arg1 = self._visit_expr(e.arg1, ctx)
        arg2 = self._visit_expr(e.arg2, ctx)
        match e.op:
            case TernaryOpKind.FMA:
                return f'fma({arg0}, {arg1}, {arg2})'
            case _:
                return f'{str(e.op)}({arg0}, {arg1}, {arg2})'

    def _visit_naryop(self, e: NaryOp, ctx: _Ctx):
        args = [self._visit_expr(arg, ctx) for arg in e.args]
        match e.op:
            case NaryOpKind.AND:
                return ' and '.join(args)
            case NaryOpKind.OR:
                return ' or '.join(args)
            case _:
                raise NotImplementedError

    def _visit_compare(self, e: Compare, ctx: _Ctx):
        first = self._visit_expr(e.args[0], ctx)
        rest = [self._visit_expr(arg, ctx) for arg in e.args[1:]]
        s = ' '.join(f'{op.symbol()} {arg}' for op, arg in zip(e.ops, rest))
        return f'{first} {s}'

    def _visit_call(self, e: Call, ctx: _Ctx):
        args = [self._visit_expr(arg, ctx) for arg in e.args]
        arg_str = ', '.join(args)
        return f'{e.op}({arg_str})'

    def _visit_tuple_expr(self, e: TupleExpr, ctx: _Ctx):
        elts = [self._visit_expr(elt, ctx) for elt in e.args]
        return f'({", ".join(elts)})'

    def _visit_comp_expr(self, e: CompExpr, ctx: _Ctx):
        elt = self._visit_expr(e.elt, ctx)
        iterables = [self._visit_expr(iterable, ctx) for iterable in e.iterables]
        s = ' '.join(f'for {str(var)} in {iterable}' for var, iterable in zip(e.vars, iterables))
        return f'[{elt} {s}]'

    def _visit_ref_expr(self, e: RefExpr, ctx: _Ctx):
        value = self._visit_expr(e.value, ctx)
        slices = [self._visit_expr(slice, ctx) for slice in e.slices]
        ref_str = ''.join(f'[{slice}]' for slice in slices)
        return f'{value}{ref_str}'

    def _visit_if_expr(self, e: IfExpr, ctx: _Ctx):
        cond = self._visit_expr(e.cond, ctx)
        ift = self._visit_expr(e.ift, ctx)
        iff = self._visit_expr(e.iff, ctx)
        return f'({ift} if {cond} else {iff})'

    def _visit_var_assign(self, stmt: VarAssign, ctx: _Ctx):
        val = self._visit_expr(stmt.expr, ctx)
        self._add_line(f'{str(stmt.var)} = {val}', ctx)

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

    def _visit_tuple_assign(self, stmt: TupleAssign, ctx: _Ctx):
        val = self._visit_expr(stmt.expr, ctx)
        vars = self._visit_tuple_binding(stmt.binding)
        self._add_line(f'{vars} = {val}', ctx)

    def _visit_ref_assign(self, stmt: RefAssign, ctx: _Ctx):
        slices = [self._visit_expr(slice, ctx) for slice in stmt.slices]
        val = self._visit_expr(stmt.expr, ctx)
        ref_str = ''.join(f'[{slice}]' for slice in slices)
        self._add_line(f'{str(stmt.var)}{ref_str} = {val}', ctx)

    def _visit_if_stmt(self, stmt: IfStmt, ctx: _Ctx):
        cond = self._visit_expr(stmt.cond, ctx)
        self._add_line(f'if {cond}:', ctx)
        self._visit_block(stmt.ift, ctx + 1)
        if stmt.iff:
            self._add_line('else:', ctx)
            self._visit_block(stmt.iff, ctx + 1)

    def _visit_while_stmt(self, stmt: WhileStmt, ctx: _Ctx):
        cond = self._visit_expr(stmt.cond, ctx)
        self._add_line(f'while {cond}:', ctx)
        self._visit_block(stmt.body, ctx + 1)

    def _visit_for_stmt(self, stmt: ForStmt, ctx: _Ctx):
        iterable = self._visit_expr(stmt.iterable, ctx)
        self._add_line(f'for {str(stmt.var)} in {iterable}:', ctx)
        self._visit_block(stmt.body, ctx + 1)

    def _visit_context(self, stmt: ContextStmt, ctx: _Ctx):
        # TODO: format data
        props = ', '.join(f'{k}={pformat(v)}' for k, v in stmt.props.items())
        self._add_line(f'with Context({props}):', ctx)
        self._visit_block(stmt.body, ctx + 1)

    def _visit_assert(self, stmt: AssertStmt, ctx: _Ctx):
        test = self._visit_expr(stmt.test, ctx)
        if stmt.msg is None:
            self._add_line(f'assert {test}', ctx)
        else:
            self._add_line(f'assert {test}, {stmt.msg}', ctx)

    def _visit_return(self, stmt: Return, ctx: _Ctx):
        s = self._visit_expr(stmt.expr, ctx)
        self._add_line(f'return {s}', ctx)

    def _visit_block(self, block: Block, ctx: _Ctx):
        for stmt in block.stmts:
            self._visit_statement(stmt, ctx)

    def _format_data(self, data, arg_str: str):
        if isinstance(data, Expr):
            e = self._visit_expr(data, 0)
            return f'lambda {arg_str}: {e}'
        else:
            return pformat(data)

    def _format_decorator(self, props: dict[str, str], arg_str: str, ctx: _Ctx):
        if len(props) == 0:
            self._add_line('@fpy', ctx)
        elif len(props) == 1:
            k, *_ = tuple(props.keys())
            v = self._format_data(props[k], arg_str)
            self._add_line(f'@fpy({k}={v})', ctx)
        else:
            self._add_line('@fpy(', ctx)
            for k, data in props.items():
                v = self._format_data(data, arg_str)
                self._add_line(f'{k}={v},', ctx + 1)
            self._add_line(')', ctx)

    def _visit_function(self, func: FunctionDef, ctx: _Ctx):
        # TODO: type annotation
        arg_strs = [str(arg.name) for arg in func.args]
        arg_str = ', '.join(arg_strs)
        self._format_decorator(func.ctx, arg_str, ctx)
        self._add_line(f'def {func.name}({arg_str}):', ctx)
        self._visit_block(func.body, ctx + 1)

    # override for typing hint
    def _visit_expr(self, e: Expr, ctx: _Ctx) -> str:
        return super()._visit_expr(e, ctx)

    # override for typing hint
    def _visit_statement(self, stmt: Stmt, ctx: _Ctx) -> None:
        return super()._visit_statement(stmt, ctx)


class Formatter(BaseFormatter):
    """"Pretty printer for FPy AST"""

    def format(self, ast: Ast) -> str:
        """Pretty print the given AST"""
        return _FormatterInstance(ast).apply()


