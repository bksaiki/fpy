"""
This module defines an FPy interpreter that uses the Rival interval library
to compute the true real number result.
"""

from fractions import Fraction

from typing import Any, Optional, Sequence, TypeAlias

from titanfp.arithmetic.evalctx import EvalCtx, determine_ctx
from titanfp.arithmetic.ieee754 import Float, IEEECtx, ieee_ctx
from titanfp.titanic.ndarray import NDArray
from titanfp.titanic.digital import Digital

from .interval import RealInterval
from .rival_manager import RivalManager, InsufficientPrecisionError, PrecisionLimitExceeded

from ..common import ExprTraceEntry
from ..function import Interpreter, Function, FunctionReturnException
from ...ir import *

ScalarVal: TypeAlias = str | bool | RealInterval
"""Type of scalar values in FPy programs."""
TensorVal: TypeAlias = NDArray
"""Type of tensor values in FPy programs."""

ScalarArg: TypeAlias = ScalarVal | str | int | float
"""Type of scalar arguments in FPy programs; includes native Python types"""
TensorArg: TypeAlias = NDArray | tuple | list
"""Type of tensor arguments in FPy programs; includes native Python types"""

MAX_ITERS = 50

"""Maps python operator to the corresponding operator in Rival"""
_method_table: dict[str, str] = {
    '+': '+',
    '-': '-',
    '*': '*',
    '/': '/',
    'fabs': 'fabs',
    'sqrt': 'sqrt',
    'fma': 'fma',
    'neg': 'neg',
    'copysign': 'copysign',
    'fdim': 'fdim',
    'fmax': 'fmax',
    'fmin': 'fmin',
    'fmod': 'fmod',
    'remainder': 'remainder',
    'hypot': 'hypot',
    'cbrt': 'cbrt',
    'ceil': 'ceil',
    'floor': 'floor',
    'nearbyint': 'nearbyint',
    'round': 'round',
    'trunc': 'trunc',
    'acos': 'acos',
    'asin': 'asin',
    'atan': 'atan',
    'atan2': 'atan2',
    'cos': 'cos',
    'sin': 'sin',
    'tan': 'tan',
    'acosh': 'acosh',
    'asinh': 'asinh',
    'atanh': 'atanh',
    'cosh': 'cosh',
    'sinh': 'sinh',
    'tanh': 'tanh',
    'exp': 'exp',
    'exp2': 'exp2',
    'expm1': 'expm1',
    'log': 'log',
    'log10': 'log10',
    'log1p': 'log1p',
    'log2': 'log2',
    'pow': 'pow',
    'erf': 'erf',
    'erfc': 'erfc',
    'lgamma': 'lgamma',
    'tgamma': 'tgamma',
    'isfinite': 'isfinite',
    'isinf': 'isinf',
    'isnan': 'isnan',
    'isnormal': 'isnormal',
    'signbit': 'signbit',
}

def _interval_to_real(val: RealInterval, ctx: EvalCtx):
    # rounding contexts
    assert isinstance(ctx, IEEECtx)
    # compute the midpoint
    # TODO: not entirely sound, should be inside the rounding envelope
    lo = Fraction(val.lo)
    hi = Fraction(val.hi)
    mid = (lo + hi) / 2
    return Float(x=mid, ctx=ctx)

def _digital_to_str(x: Digital) -> str:
    m = (-1 if x.negative else 1) * x.c
    pow2 = Fraction(2) ** x.exp
    return str(m * pow2)


class _Interpreter(ReduceVisitor):
    """Single-use real number interpreter"""

    rival: RivalManager
    """Rival object for evaluating expressions"""

    env: dict[NamedId, ScalarVal]
    """mappping from variable names to values"""
    curr_prec: dict[NamedId, int]
    """mapping from variables names to current precision"""
    req_prec: dict[NamedId, int]
    """mapping from variables names to required precision"""
    visited: set[NamedId]
    """"set of visited variables during this iteration"""
    dirty: bool
    """has a variable precision been updated?"""

    expr_trace: list[ExprTraceEntry]
    """expression trace"""
    trace: bool
    """expression tracing enabled?"""

    def __init__(self, rival: RivalManager, trace: bool):
        self.rival = rival
        self.env = {}
        self.curr_prec = {}
        self.req_prec = {}
        self.visited = set()
        self.dirty = False
        self.expr_trace = []
        self.trace = trace

    def _arg_to_real(self, arg: Any):
        if isinstance(arg, str):
            return arg
        elif isinstance(arg, int | float):
            return str(arg)
        elif isinstance(arg, Digital):
            return _digital_to_str(arg)
        elif isinstance(arg, tuple | list):
            raise NotImplementedError()
        else:
            raise NotImplementedError(f'unknown argument type {arg}')

    def eval(self,
        func: FunctionDef,
        args: Sequence[Any],
        ctx: Optional[EvalCtx] = None,
    ):
        if not isinstance(func, FunctionDef):
            raise TypeError(f'Expected Function, got {type(func)}')

        # check arity
        args = tuple(args)
        if len(args) != len(func.args):
            raise TypeError(f'Expected {len(func.args)} arguments, got {len(args)}')

        # default context if none is specified
        if ctx is None:
            ctx = ieee_ctx(11, 64)
        ctx = determine_ctx(ctx, func.ctx)
        self.rival.set_precision(ctx.p)

        for val, arg in zip(args, func.args):
            match arg.ty:
                case AnyType():
                    x = self._arg_to_real(val)
                    if isinstance(arg.name, NamedId):
                        self.env[arg.name] = x
                case RealType():
                    x = self._arg_to_real(val)
                    if isinstance(arg.name, NamedId):
                        self.env[arg.name] = x
                case _:
                    raise NotImplementedError(f'unsupported argument type {arg.ty}')

        iter_num = 0
        while iter_num < MAX_ITERS:
            try:
                self.visited = set()
                self.dirty = False
                self.expr_trace.clear()
                self._visit_block(func.body, ctx)
                raise RuntimeError('no return statement encountered')
            except FunctionReturnException as e:
                return e.value
            except InsufficientPrecisionError as e:
                if self.rival.logging:
                    print(f"Insufficient precision, retrying iter={iter_num}, e={e.expr}, prec={e.prec}")
                if iter_num > 0 and not self.dirty:
                    # we didn't increase precision anywhere
                    raise PrecisionLimitExceeded('precision limit exceeded') from e
                iter_num += 1

        # something has definitely went wrong
        raise NotImplementedError('unreachable')


    def _lookup(self, name: NamedId):
        if name not in self.env:
            raise RuntimeError(f'unbound variable {name}')
        return name.base # We return the name rather than the value in expression

    def _visit_var(self, e: Var, ctx: EvalCtx):
        return e.name

    def _visit_bool(self, e: Bool, ctx: Any):
        return e.val

    def _visit_decnum(self, e: Decnum, ctx: EvalCtx):
        return str(e.val)

    def _visit_hexnum(self, e: Hexnum, ctx: EvalCtx):
        return str(e.val)

    def _visit_integer(self, e: Integer, ctx: EvalCtx):
        return str(e.val)

    def _visit_rational(self, e: Rational, ctx: EvalCtx):
        return f'{e.p}/{e.q}'

    def _visit_digits(self, e: Digits, ctx: EvalCtx):
        x = Fraction(e.b) ** e.e
        return str(e.m * x)

    def _visit_nary_expr(self, e: NaryExpr, ctx: EvalCtx):
        if e.name in _method_table:
            return self._apply_method(e, ctx)
        elif isinstance(e, Cast):
            return self._visit_expr(e.children[0], ctx)
        elif isinstance(e, Not):
            return self._apply_not(e, ctx)
        elif isinstance(e, And):
            return self._apply_and(e, ctx)
        elif isinstance(e, Or):
            return self._apply_or(e, ctx)
        elif isinstance(e, Range):
            return self._apply_range(e, ctx)
        else:
            raise NotImplementedError('unknown n-ary expression', e)

    def _apply_method(self, e: NaryExpr, ctx: EvalCtx):
        fn = _method_table[e.name]
        args = [self._visit_expr(arg, ctx) for arg in e.children]
        arg_values_str = " ".join(map(str, args))
        return f"({fn} {arg_values_str})"

    def _apply_not(self, e: Not, ctx: EvalCtx):
        arg = self._visit_expr(e.children[0], ctx)
        return f'(not {arg})'

    def _nary_to_2ary(self, op: str, args: list) -> str:
        if len(args) == 1:
            return args[0]
        elif len(args) == 2:
            return f'({op} {" ".join(args)})'
        else:
            return f'({op} {self._nary_to_2ary(op, args[:-1])} {args[-1]})'

    def _apply_and(self, e: And, ctx: EvalCtx):
        args = [self._visit_expr(arg, ctx) for arg in e.children]
        return self._nary_to_2ary('and', args)

    def _apply_or(self, e: Or, ctx: EvalCtx):
        args = [self._visit_expr(arg, ctx) for arg in e.children]
        return self._nary_to_2ary('or', args)

    def _apply_range(self, e: Range, ctx: EvalCtx):
        stop = self._force_value(self._eval_rival(e.children[0], ctx), ctx)
        if not isinstance(stop, Float):
            raise TypeError(f'expected a real number argument, got {stop}')
        if not stop.is_integer():
            raise TypeError(f'expected an integer argument, got {stop}')
        return NDArray([str(i) for i in range(int(stop))])

    def _visit_comp_expr(self, e: CompExpr, ctx: EvalCtx):
        raise NotImplementedError

    def _visit_unknown(self, e: UnknownCall, ctx: EvalCtx):
        raise NotImplementedError

    def _visit_constant(self, e: Constant, ctx: EvalCtx):
        raise NotImplementedError

    def _visit_tuple_expr(self, e: TupleExpr, ctx: EvalCtx):
        elts = [self._eval_rival(elt, ctx) for elt in e.children]
        return NDArray(elts)

    def _visit_tuple_ref(self, e: TupleRef, ctx: EvalCtx):
        raise NotImplementedError

    def _visit_tuple_set(self, e: TupleSet, ctx: EvalCtx):
        raise NotImplementedError

    def _apply_cmp2(self, op: CompareOp, lhs, rhs):
        match op:
            case CompareOp.EQ:
                return f'(== {lhs} {rhs})'
            case CompareOp.NE:
                return f'(!= {lhs} {rhs})'
            case CompareOp.LT:
                return f'(< {lhs} {rhs})'
            case CompareOp.LE:
                return f'(<= {lhs} {rhs})'
            case CompareOp.GT:
                return f'(> {lhs} {rhs})'
            case CompareOp.GE:
                return f'(>= {lhs} {rhs})'
            case _:
                raise NotImplementedError('unknown comparison operator', op)

    def _visit_compare(self, e: Compare, ctx: EvalCtx):
        args: list[str] = []
        lhs = self._visit_expr(e.children[0], ctx)
        for op, arg in zip(e.ops, e.children[1:]):
            rhs = self._visit_expr(arg, ctx)
            args.append(self._apply_cmp2(op, lhs, rhs))
            lhs = rhs
        return self._nary_to_2ary('and', args)

    def _visit_if_expr(self, e: IfExpr, ctx: EvalCtx):
        cond = self._visit_expr(e.cond, ctx)
        ift = self._visit_expr(e.ift, ctx)
        iff = self._visit_expr(e.iff, ctx)
        return f'(if {cond} {ift} {iff})'

    def _arg_to_rival(self, arg: ScalarVal):
        match arg:
            case bool():
                return '#t' if arg else '#f'
            case str():
                if arg == 'nan':
                    return '+nan.0'
                elif arg == '+inf':
                    return '+inf.0'
                elif arg == '-inf':
                    return '-inf.0'
                else:
                    return arg
            case RealInterval():
                return f'(ival {arg.lo} {arg.hi} {arg.prec})'
            case _:
                raise NotImplementedError(f'unknown type {arg} {type(arg)}')

    def _eval_rival_inner(self, expr: Expr, ctx: EvalCtx):
        match self._visit_expr(expr, ctx):
            case bool() as b:
                return b
            case NamedId() as var:
                return self.env[var]
            case NDArray() as arr:
                return arr
            case str() as s:
                if s.startswith('('):
                    # hacky way to check if we need Rival to evaluate
                    fun_str = f"(f {' '.join(map(str, self.env.keys()))}) {s}"
                    self.rival.define_function(fun_str)
                    rival_env = list(map(self._arg_to_rival, self.env.values()))
                    return self.rival.eval_expr(f"f {' '.join(rival_env)}", fun_str)
                else:
                    # numerical constant
                    return s

    def _eval_rival(self, expr: Expr, ctx: EvalCtx):
        """
        Applies Rival to an expression.
        If the expression is exact, returns its value as a string.
        """
        val = self._eval_rival_inner(expr, ctx)
        if self.trace:
            env = {k: self._force_value(v, ctx) for k, v in self.env.items()}
            trace = ExprTraceEntry(expr, self._force_value(val, ctx), env, ctx)
            self.expr_trace.append(trace)
        return val

    def _force_value(self, val: ScalarVal, ctx: EvalCtx):
        """
        Not every expression is evaluated to a concrete value.
        This function ensures that the result is a concrete value.
        """
        match val:
            case bool():
                return val
            case str():
                match self.rival.eval_expr(self._arg_to_rival(val), val):
                    case bool() as b:
                        return b
                    case str() as s:
                        assert isinstance(ctx, IEEECtx), 'expected an IEEECtx'
                        if s == 'nan':
                            return Float(isnan=True, ctx=ctx)
                        elif s == '+inf':
                            return Float(isinf=True, negative=False, ctx=ctx)
                        elif s == '-inf':
                            return Float(isinf=True, negative=True, ctx=ctx)
                        else:
                            raise NotImplementedError(f'unknown string value {s}')
                    case RealInterval() as ival:
                        return _interval_to_real(ival, ctx)
                    case default:
                        raise NotImplementedError(f'unreachable {default}')
            case RealInterval():
                return _interval_to_real(val, ctx)
            case NDArray():
                return NDArray([self._force_value(elt, ctx) for elt in val])
            case _:
                raise NotImplementedError(f'unreachable {val}')

    def _visit_var_assign(self, stmt: VarAssign, ctx: EvalCtx):
        match stmt.var:
            case NamedId():
                # only `SourceId` comes from the parser
                # if isinstance(stmt.var, SourceId):
                #   <do something>
                if stmt.var not in self.req_prec:
                    # first time visiting this assignment
                    self.req_prec[stmt.var] = ctx.p
                    self.curr_prec[stmt.var] = ctx.p
                else:
                    # revisiting this assignment
                    if stmt.var not in self.visited:
                        # first time visiting during this iteration
                        self.curr_prec[stmt.var] = 2 * self.curr_prec[stmt.var]
                        self.visited.add(stmt.var)
                        self.dirty = True

                self.rival.set_precision(self.curr_prec[stmt.var])
                self.env[stmt.var] = self._eval_rival(stmt.expr, ctx)
            case UnderscoreId():
                pass
            case _:
                raise NotImplementedError('unknown variable', stmt.var)

    def _visit_cond(self, cond: Expr, ctx: EvalCtx):
        v = self._eval_rival(cond, ctx)
        if v == 'nan':
            # Rival instance returns +nan.0 even if the expression is a boolean
            return False
        else:
            val = self._force_value(v, ctx)
            if not isinstance(val, bool):
                raise TypeError(f'expected a boolean, got {val}')
            return val

    def _visit_if1_stmt(self, stmt: If1Stmt, ctx: EvalCtx):
        if self._visit_cond(stmt.cond, ctx):
            self._visit_block(stmt.body, ctx)
            for phi in stmt.phis:
                self.env[phi.name] = self.env[phi.rhs]
        else:
            for phi in stmt.phis:
                self.env[phi.name] = self.env[phi.lhs]

    def _visit_if_stmt(self, stmt: IfStmt, ctx: EvalCtx):
        if self._visit_cond(stmt.cond, ctx):
            self._visit_block(stmt.ift, ctx)
            for phi in stmt.phis:
                self.env[phi.name] = self.env[phi.lhs]
        else:
            self._visit_block(stmt.iff, ctx)
            for phi in stmt.phis:
                self.env[phi.name] = self.env[phi.rhs]

    def _visit_context(self, stmt: ContextStmt, ctx: EvalCtx):
        ctx = determine_ctx(ctx, stmt.props)
        return self._visit_block(stmt.body, ctx)

    def _visit_assert(self, stmt: AssertStmt, ctx: EvalCtx):
        test = self._visit_cond(stmt.test, ctx)
        if not test:
            raise AssertionError(stmt.msg)
        return ctx

    def _visit_return(self, stmt: Return, ctx: EvalCtx) -> bool | float:
        # since we are returning we actually want a value
        self.rival.set_precision(ctx.p)
        val = self._eval_rival(stmt.expr, ctx)
        return self._force_value(val, ctx)

    def _visit_block(self, block: Block, ctx: EvalCtx):
        for stmt in block.stmts:
            if isinstance(stmt, Return):
                v = self._visit_return(stmt, ctx)
                raise FunctionReturnException(v)
            else:
                self._visit_statement(stmt, ctx)

    def _visit_while_stmt(self, stmt: WhileStmt, ctx: EvalCtx) -> None:
        for phi in stmt.phis:
            self.env[phi.name] = self.env[phi.lhs]
            del self.env[phi.lhs]

        while self._visit_cond(stmt.cond, ctx):
            self._visit_block(stmt.body, ctx)
            for phi in stmt.phis:
                self.env[phi.name] = self.env[phi.rhs]
                del self.env[phi.rhs]

    def _visit_tuple_assign(self, stmt: TupleAssign, ctx: EvalCtx):
        raise NotImplementedError

    def _visit_ref_assign(self, stmt: RefAssign, ctx: EvalCtx):
        raise NotImplementedError

    def _visit_for_stmt(self, stmt: ForStmt, ctx: EvalCtx):
        for phi in stmt.phis:
            self.env[phi.name] = self.env[phi.lhs]
            del self.env[phi.lhs]

        iterable = self._visit_expr(stmt.iterable, ctx)
        if not isinstance(iterable, NDArray):
            raise TypeError(f'expected a tensor, got {iterable}')

        for val in iterable:
            if isinstance(stmt.var, NamedId):
                self.env[stmt.var] = val
            self._visit_block(stmt.body, ctx)
            for phi in stmt.phis:
                self.env[phi.name] = self.env[phi.rhs]
                del self.env[phi.rhs]

    def _visit_phis(self, phis: list[PhiNode], lctx: EvalCtx, rctx: EvalCtx):
        raise NotImplementedError('do not call directly')

    def _visit_loop_phis(self, phis: list[PhiNode], lctx: EvalCtx, rctx: EvalCtx):
        raise NotImplementedError('do not call directly')

    def _visit_function(self, func: FunctionDef, ctx: EvalCtx):
        raise NotImplementedError('do not call directly')
    
    # override for typing
    def _visit_expr(self, e: Expr, ctx: EvalCtx) -> NDArray | NamedId | str | bool:
        return super()._visit_expr(e, ctx)


class RealInterpreter(Interpreter):
    """
    Real-number interpreter for FPy functions.

    Computes the true real number result of a function,
    rounded to the nearest floating-point value at some precision.
    This interpreter leverages the Rival interval library developed by the Herbie project.
    More information on the Rival library and the Herbie project can
    be found here: https://herbie.uwplse.org/.
    """

    rival: RivalManager
    """Rival object for evaluating expressions"""

    def __init__(self, logging: bool = False):
        self.rival = RivalManager(logging=logging)
        self.rival.set_print_ival(True)

    def eval(
        self,
        func: Function,
        args: Sequence[Any],
        ctx: Optional[EvalCtx] = None
    ):
        if not isinstance(func, Function):
            raise TypeError(f'Expected Function, got {func}')
        rt = _Interpreter(self.rival, False)
        return rt.eval(func.ir, args, ctx)

    def eval_expr(self, expr, env, ctx):
        raise NotImplementedError

    def eval_with_trace(
        self,
        func: Function,
        args: Sequence[Any],
        ctx: Optional[EvalCtx] = None
    ):
        if not isinstance(func, Function):
            raise TypeError(f'Expected Function, got {func}')
        rt = _Interpreter(self.rival, True)
        result = rt.eval(func.ir, args, ctx)
        return (result, rt.expr_trace)

