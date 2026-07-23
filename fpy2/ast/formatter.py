"""Pretty printing of FPy ASTs"""

from pprint import pformat
from types import ModuleType

from ..env import ForeignEnv
from ..fpc_context import FPCoreContext
from .fpyast import *
from .visitor import Visitor

_Ctx = int

class _FormatterInstance(Visitor):
    """Single-instance visitor for pretty printing FPy ASTs"""
    ast: Ast
    fmt: str
    _fpy_alias: str | None

    def __init__(self, ast: Ast):
        self.ast = ast
        self.fmt = ''
        # `fpy2`'s alias in the enclosing function's namespace; set per `FuncDef`
        self._fpy_alias = None

    def apply(self) -> str:
        match self.ast:
            case Expr():
                self.fmt = self._visit_expr(self.ast, 0)
            case Stmt():
                self._visit_statement(self.ast, 0)
            case StmtBlock():
                self._visit_block(self.ast, 0)
            case Argument():
                # TODO: type signature
                return str(self.ast.name)
            case TupleBinding():
                self.fmt = self._visit_tuple_binding(self.ast)
            case FuncDef():
                self._visit_function(self.ast, 0)
            case _:
                raise NotImplementedError('unsupported AST node', self.ast)
        return self.fmt.strip()

    def _add_line(self, line: str, indent: int):
        self.fmt += '    ' * indent + line + '\n'

    def _visit_function_name(self, func: Var | Attribute, ctx: _Ctx) -> str:
        match func:
            case Var():
                return self._visit_var(func, ctx)
            case Attribute():
                return self._visit_attribute(func, ctx)
            case _:
                raise RuntimeError('unreachable', func)

    @staticmethod
    def _find_fpy_alias(env: ForeignEnv | None) -> str | None:
        """Name that the `fpy2` package is bound to in `env`, if any.

        `env` iterates in nondeterministic (set-union) order, so pick the
        lexicographically smallest match to keep formatting stable when
        `fpy2` is bound under more than one name."""
        if env is None:
            return None
        aliases = [
            name for name in env
            if isinstance(env.get(name), ModuleType)
            and getattr(env.get(name), '__name__', None) == 'fpy2'
        ]
        return min(aliases, default=None)

    def _fpy_qualified(self, name: str) -> str:
        """Render an `fpy2` symbol as `<alias>.<name>` when the enclosing
        function imports `fpy2` under a module alias, else bare `name`."""
        import fpy2
        if self._fpy_alias is not None and hasattr(fpy2, name):
            return f'{self._fpy_alias}.{name}'
        return name

    def _op_name(self, e: Expr, ctx: _Ctx) -> str:
        """Surface name of a named-operator / literal node: the user's `func`
        symbol verbatim, or the canonical name when `func` is `None`
        (synthesized by a rewrite pass)."""
        func = getattr(e, 'func', None)
        if func is not None:
            return self._visit_function_name(func, ctx)
        return self._fpy_qualified(CANONICAL_OP_NAMES[type(e)])

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
        name = self._op_name(e, ctx)
        return f'{name}(\'{e.val}\')'

    def _visit_integer(self, e: Integer, ctx: _Ctx):
        return str(e.val)

    def _visit_rational(self, e: Rational, ctx: _Ctx):
        name = self._op_name(e, ctx)
        return f'{name}({e.p}, {e.q})'

    def _visit_digits(self, e: Digits, ctx: _Ctx):
        name = self._op_name(e, ctx)
        return f'{name}({e.m}, {e.e}, {e.b})'

    def _visit_nullaryop(self, e: NullaryOp, ctx: _Ctx):
        name = self._op_name(e, ctx)
        return f'{name}()'

    def _visit_unaryop(self, e: UnaryOp, ctx: _Ctx):
        arg = self._visit_expr(e.arg, ctx)
        match e:
            case NamedUnaryOp():
                name = self._op_name(e, ctx)
                return f'{name}({arg})'
            case Abs():
                return f'abs({arg})'
            case Neg():
                return f'-{arg}'
            case Not():
                return f'not {arg}'
            case _:
                raise RuntimeError('unreachable', e)

    def _visit_binaryop(self, e: BinaryOp, ctx: _Ctx):
        lhs = self._visit_expr(e.first, ctx)
        rhs = self._visit_expr(e.second, ctx)
        match e:
            case NamedBinaryOp():
                name = self._op_name(e, ctx)
                return f'{name}({lhs}, {rhs})'
            case Add():
                return f'({lhs} + {rhs})'
            case Sub():
                return f'({lhs} - {rhs})'
            case Mul():
                return f'({lhs} * {rhs})'
            case Div():
                return f'({lhs} / {rhs})'
            case Mod():
                return f'({lhs} % {rhs})'
            case _:
                raise RuntimeError('unreachable', e)

    def _visit_ternaryop(self, e: TernaryOp, ctx: _Ctx):
        arg0 = self._visit_expr(e.first, ctx)
        arg1 = self._visit_expr(e.second, ctx)
        arg2 = self._visit_expr(e.third, ctx)
        match e:
            case NamedTernaryOp():
                name = self._op_name(e, ctx)
                return f'{name}({arg0}, {arg1}, {arg2})'
            case _:
                raise RuntimeError('unreachable', e)

    def _visit_naryop(self, e: NaryOp, ctx: _Ctx):
        args = [self._visit_expr(arg, ctx) for arg in e.args]
        match e:
            case NamedNaryOp():
                name = self._op_name(e, ctx)
                return f'{name}({", ".join(args)})'
            case And():
                return ' and '.join(args)
            case Or():
                return ' or '.join(args)
            case _:
                raise RuntimeError('unreachable', e)

    def _visit_call(self, e: Call, ctx: _Ctx):
        name = self._visit_function_name(e.func, ctx)
        args = [self._visit_expr(arg, ctx) for arg in e.args]
        arg_str = ', '.join(args)
        return f'{name}({arg_str})'

    def _visit_compare(self, e: Compare, ctx: _Ctx):
        first = self._visit_expr(e.args[0], ctx)
        rest = [self._visit_expr(arg, ctx) for arg in e.args[1:]]
        s = ' '.join(f'{op.symbol()} {arg}' for op, arg in zip(e.ops, rest))
        return f'{first} {s}'

    def _visit_tuple_expr(self, e: TupleExpr, ctx: _Ctx):
        num_elts = len(e.elts)
        if num_elts == 0:
            return '()'
        elif num_elts == 1:
            elt = self._visit_expr(e.elts[0], ctx)
            return f'({elt},)'
        else:
            elts = [self._visit_expr(elt, ctx) for elt in e.elts]
            return f'({", ".join(elts)})'

    def _visit_list_expr(self, e: ListExpr, ctx: _Ctx):
        num_elts = len(e.elts)
        if num_elts == 0:
            return '[]'
        elif num_elts == 1:
            elt = self._visit_expr(e.elts[0], ctx)
            return f'[{elt}]'
        else:
            elts = [self._visit_expr(elt, ctx) for elt in e.elts]
            return f'[{", ".join(elts)}]'

    def _visit_list_comp(self, e: ListComp, ctx: _Ctx):
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

    def _visit_list_ref(self, e: ListRef, ctx: _Ctx):
        value = self._visit_expr(e.value, ctx)
        index = self._visit_expr(e.index, ctx)
        return f'{value}[{index}]'

    def _visit_list_slice(self, e: ListSlice, ctx: _Ctx):
        value = self._visit_expr(e.value, ctx)
        start = '' if e.start is None else self._visit_expr(e.start, ctx)
        stop = '' if e.stop is None else self._visit_expr(e.stop, ctx)
        return f'{value}[{start}:{stop}]'

    def _visit_if_expr(self, e: IfExpr, ctx: _Ctx):
        cond = self._visit_expr(e.cond, ctx)
        ift = self._visit_expr(e.ift, ctx)
        iff = self._visit_expr(e.iff, ctx)
        return f'({ift} if {cond} else {iff})'

    def _visit_attribute(self, e: Attribute, ctx: _Ctx):
        value = self._visit_expr(e.value, ctx)
        return f'{value}.{e.attr}'

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
        match stmt.target:
            case Id():
                self._add_line(f'{stmt.target!s} = {val}', ctx)
            case TupleBinding():
                vars_str = self._visit_tuple_binding(stmt.target)
                self._add_line(f'{vars_str} = {val}', ctx)
            case _:
                raise NotImplementedError('unreachable', stmt.var)

    def _visit_indexed_assign(self, stmt: IndexedAssign, ctx: _Ctx):
        slices = [self._visit_expr(slice, ctx) for slice in stmt.indices]
        val = self._visit_expr(stmt.expr, ctx)
        ref_str = ''.join(f'[{slice}]' for slice in slices)
        self._add_line(f'{stmt.var!s}{ref_str} = {val}', ctx)

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
        match stmt.target:
            case NamedId():
                self._add_line(f'with {context} as {stmt.target!s}:', ctx)
            case UnderscoreId():
                self._add_line(f'with {context}:', ctx)
            case _:
                raise RuntimeError('unreachable', stmt.target)
        self._visit_block(stmt.body, ctx + 1)

    def _visit_assert(self, stmt: AssertStmt, ctx: _Ctx):
        test = self._visit_expr(stmt.test, ctx)
        if stmt.msg is None:
            self._add_line(f'assert {test}', ctx)
        else:
            msg = self._visit_expr(stmt.msg, ctx)
            self._add_line(f'assert {test}, {msg}', ctx)

    def _visit_effect(self, stmt: EffectStmt, ctx: _Ctx):
        expr = self._visit_expr(stmt.expr, ctx)
        self._add_line(f'{expr}', ctx)

    def _visit_return(self, stmt: ReturnStmt, ctx: _Ctx):
        s = self._visit_expr(stmt.expr, ctx)
        self._add_line(f'return {s}', ctx)

    def _visit_pass(self, stmt: PassStmt, ctx: _Ctx):
        self._add_line('pass', ctx)

    def _visit_block(self, block: StmtBlock, ctx: _Ctx):
        for stmt in block.stmts:
            self._visit_statement(stmt, ctx)

    def _format_data(self, data, arg_str: str):
        if isinstance(data, Expr):
            e = self._visit_expr(data, 0)
            return f'lambda {arg_str}: {e}'
        else:
            return pformat(data)

    def _format_decorator(self, arg_str: str, fmeta: FuncMeta, ctx: _Ctx):
        rctx = fmeta.ctx
        spec = fmeta.spec
        meta = fmeta.props

        name = self._fpy_qualified('fpy')
        if not (rctx or spec or meta):
            # no keyword arguments: emit the bare decorator without `(...)`
            self._add_line(f'@{name}', ctx)
        else:
            self._add_line(f'@{name}(', ctx)
            if rctx:
                v = self._format_data(rctx, arg_str)
                self._add_line(f'ctx={v},', ctx + 1)
            if spec:
                v = self._format_data(spec, arg_str)
                self._add_line(f'spec={v},', ctx + 1)
            if meta:
                self._add_line('meta={', ctx + 1)
                for k, v in meta.items():
                    v = self._format_data(v, arg_str)
                    self._add_line(f'\'{k}\': {v},', ctx + 2)
                self._add_line('}', ctx + 1)
            self._add_line(')', ctx)

    def _visit_function(self, func: FuncDef, ctx: _Ctx):
        # record how `fpy2` is imported so synthesized operators can be named
        self._fpy_alias = self._find_fpy_alias(func.env)

        # TODO: type annotation
        arg_strs = [str(arg.name) for arg in func.args]
        arg_str = ', '.join(arg_strs)

        self._format_decorator(arg_str, func.meta, ctx)
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


