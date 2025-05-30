"""Pretty printing of FPy ASTs"""

from pprint import pformat

from ..fpc_context import FPCoreContext
from .fpyast import *
from .visitor import Visitor

_Ctx = int

class _FormatterInstance(Visitor):
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
            case StmtBlock():
                self._visit_block(self.ast, 0)
            case FuncDef():
                self._visit_function(self.ast, 0)
            case _:
                raise NotImplementedError('unsupported AST node', self.ast)
        return self.fmt.strip()

    def _add_line(self, line: str, indent: int):
        self.fmt += '    ' * indent + line + '\n'

    def _visit_var(self, e: Var, ctx: _Ctx) -> str:
        return str(e.name)

    def _visit_bool(self, e: BoolVal, ctx: _Ctx):
        return str(e.val)

    def _visit_foreign(self, e: ForeignVal, ctx: _Ctx):
        match e.val:
            case FPCoreContext():
                # TODO: currently pruning unsupported attributes
                props: dict[str, Any] = {}
                for k, v in e.val.props.items():
                    if k == 'precision' or k == 'round':
                        props[k] = v
                return FPCoreContext(**props)
            case _:
                return repr(e.val)

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
        match e:
            case Neg():
                return f'-{arg}'
            case Not():
                return f'not {arg}'
            case _:
                return f'{e.name}({arg})'

    def _visit_binaryop(self, e: BinaryOp, ctx: _Ctx):
        lhs = self._visit_expr(e.first, ctx)
        rhs = self._visit_expr(e.second, ctx)
        match e:
            case Add():
                return f'({lhs} + {rhs})'
            case Sub():
                return f'({lhs} - {rhs})'
            case Mul():
                return f'({lhs} * {rhs})'
            case Div():
                return f'({lhs} / {rhs})'
            case _:
                return f'{e.name}({lhs}, {rhs})'

    def _visit_ternaryop(self, e: TernaryOp, ctx: _Ctx):
        arg0 = self._visit_expr(e.first, ctx)
        arg1 = self._visit_expr(e.second, ctx)
        arg2 = self._visit_expr(e.third, ctx)
        match e:
            case Fma():
                return f'fma({arg0}, {arg1}, {arg2})'
            case _:
                return f'{e.name}({arg0}, {arg1}, {arg2})'

    def _visit_naryop(self, e: NaryOp, ctx: _Ctx):
        args = [self._visit_expr(arg, ctx) for arg in e.args]
        match e:
            case And() | Or():
                return f' {e.name} '.join(args)
            case Zip():
                return f'zip({", ".join(args)})'
            case _:
                raise RuntimeError('unreachable', e)

    def _visit_compare(self, e: Compare, ctx: _Ctx):
        first = self._visit_expr(e.args[0], ctx)
        rest = [self._visit_expr(arg, ctx) for arg in e.args[1:]]
        s = ' '.join(f'{op.symbol()} {arg}' for op, arg in zip(e.ops, rest))
        return f'{first} {s}'

    def _visit_call(self, e: Call, ctx: _Ctx):
        args = [self._visit_expr(arg, ctx) for arg in e.args]
        arg_str = ', '.join(args)
        return f'{e.name}({arg_str})'

    def _visit_tuple_expr(self, e: TupleExpr, ctx: _Ctx):
        num_elts = len(e.args)
        if num_elts == 0:
            return '()'
        elif num_elts == 1:
            elt = self._visit_expr(e.args[0], ctx)
            return f'({elt},)'
        else:
            elts = [self._visit_expr(elt, ctx) for elt in e.args]
            return f'({", ".join(elts)})'

    def _visit_comp_expr(self, e: CompExpr, ctx: _Ctx):
        targets: list[str] = []
        for target in e.targets:
            match target:
                case Id():
                    targets.append(str(target))
                case TupleBinding():
                    s = self._visit_tuple_binding(target)
                    targets.append(f'({s})')
                case _:
                    raise RuntimeError('unreachable', target)

        elt = self._visit_expr(e.elt, ctx)
        iterables = [self._visit_expr(iterable, ctx) for iterable in e.iterables]
        s = ' '.join(f'for {target} in {iterable}' for target, iterable in zip(targets, iterables))
        return f'[{elt} {s}]'

    def _visit_tuple_ref(self, e: TupleRef, ctx: _Ctx):
        value = self._visit_expr(e.value, ctx)
        slices = [self._visit_expr(slice, ctx) for slice in e.slices]
        ref_str = ''.join(f'[{slice}]' for slice in slices)
        return f'{value}{ref_str}'

    def _visit_tuple_set(self, e: TupleSet, ctx: _Ctx):
        array = self._visit_expr(e.array, ctx)
        slices = [self._visit_expr(slice, ctx) for slice in e.slices]
        value = self._visit_expr(e.value, ctx)
        return f'tuple_set({array}, [{", ".join(slices)}], {value})'

    def _visit_if_expr(self, e: IfExpr, ctx: _Ctx):
        cond = self._visit_expr(e.cond, ctx)
        ift = self._visit_expr(e.ift, ctx)
        iff = self._visit_expr(e.iff, ctx)
        return f'({ift} if {cond} else {iff})'

    def _visit_foreign_attr(self, e: ForeignAttribute, ctx: _Ctx):
        attr_strs = [str(attr) for attr in e.attrs]
        return f'{e.name}.' + '.'.join(attr_strs)

    def _visit_context_expr(self, e: ContextExpr, ctx: _Ctx):
        match e.ctor:
            case ForeignAttribute():
                ctor_str = self._visit_foreign_attr(e.ctor, ctx)
            case Var():
                ctor_str = self._visit_var(e.ctor, ctx)

        arg_strs: list[str] = []
        for arg in e.args:
            match arg:
                case ForeignAttribute():
                    attr = self._visit_foreign_attr(arg, ctx)
                    arg_strs.append(attr)
                case _:
                    arg_strs.append(self._visit_expr(arg, ctx))

        return f'{ctor_str}({", ".join(arg_strs)})'

    def _visit_tuple_binding(self, vars: TupleBinding) -> str:
        elt_strs: list[str] = []
        for var in vars:
            match var:
                case Id():
                    elt_strs.append(str(var))
                case TupleBinding():
                    s = self._visit_tuple_binding(var)
                    elt_strs.append(f'({s})')
                case _:
                    raise NotImplementedError('unreachable', var)

        num_elts = len(elt_strs)
        if num_elts == 0:
            return '()'
        elif num_elts == 1:
            return f'({elt_strs[0]},)'
        else:
            return ', '.join(elt_strs)

    def _visit_assign(self, stmt: Assign, ctx: _Ctx):
        val = self._visit_expr(stmt.expr, ctx)
        match stmt.binding:
            case Id():
                self._add_line(f'{str(stmt.binding)} = {val}', ctx)
            case TupleBinding():
                vars_str = self._visit_tuple_binding(stmt.binding)
                self._add_line(f'{vars_str} = {val}', ctx)
            case _:
                raise NotImplementedError('unreachable', stmt.var)

    def _visit_indexed_assign(self, stmt: IndexedAssign, ctx: _Ctx):
        slices = [self._visit_expr(slice, ctx) for slice in stmt.slices]
        val = self._visit_expr(stmt.expr, ctx)
        ref_str = ''.join(f'[{slice}]' for slice in slices)
        self._add_line(f'{str(stmt.var)}{ref_str} = {val}', ctx)

    def _visit_if1(self, stmt: If1Stmt, ctx: _Ctx):
        cond = self._visit_expr(stmt.cond, ctx)
        self._add_line(f'if {cond}:', ctx)
        self._visit_block(stmt.body, ctx + 1)

    def _visit_if(self, stmt: IfStmt, ctx: _Ctx):
        cond = self._visit_expr(stmt.cond, ctx)
        self._add_line(f'if {cond}:', ctx)
        self._visit_block(stmt.ift, ctx + 1)
        self._add_line('else:', ctx)
        self._visit_block(stmt.iff, ctx + 1)

    def _visit_while(self, stmt: WhileStmt, ctx: _Ctx):
        cond = self._visit_expr(stmt.cond, ctx)
        self._add_line(f'while {cond}:', ctx)
        self._visit_block(stmt.body, ctx + 1)

    def _visit_for(self, stmt: ForStmt, ctx: _Ctx):
        match stmt.target:
            case Id():
                target = str(stmt.target)
            case TupleBinding():
                target = self._visit_tuple_binding(stmt.target)
            case _:
                raise RuntimeError('unreachable', stmt.target)

        iterable = self._visit_expr(stmt.iterable, ctx)
        self._add_line(f'for {target} in {iterable}:', ctx)
        self._visit_block(stmt.body, ctx + 1)

    def _visit_context(self, stmt: ContextStmt, ctx: _Ctx):
        context = self._visit_expr(stmt.ctx, ctx)
        self._add_line(f'with {context} as {str(stmt.name)}:', ctx)
        self._visit_block(stmt.body, ctx + 1)

    def _visit_assert(self, stmt: AssertStmt, ctx: _Ctx):
        test = self._visit_expr(stmt.test, ctx)
        if stmt.msg is None:
            self._add_line(f'assert {test}', ctx)
        else:
            self._add_line(f'assert {test}, {stmt.msg}', ctx)

    def _visit_effect(self, stmt: EffectStmt, ctx: _Ctx):
        expr = self._visit_expr(stmt.expr, ctx)
        self._add_line(f'{expr}', ctx)

    def _visit_return(self, stmt: ReturnStmt, ctx: _Ctx):
        s = self._visit_expr(stmt.expr, ctx)
        self._add_line(f'return {s}', ctx)

    def _visit_block(self, block: StmtBlock, ctx: _Ctx):
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

    def _visit_function(self, func: FuncDef, ctx: _Ctx):
        # TODO: type annotation
        arg_strs = [str(arg.name) for arg in func.args]
        arg_str = ', '.join(arg_strs)

        # metadata
        props = func.metadata.copy()
        if func.ctx is not None:
            props['context'] = func.ctx
        self._format_decorator(props, arg_str, ctx)
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


