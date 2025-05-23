"""
FPy runtime backed by the Python runtime.
"""

import math
import titanfp.titanic.gmpmath as gmpmath

from typing import Any, Callable, Optional, Sequence, TypeAlias

from ..number import Context, Float, IEEEContext, RM
from ..number.gmp import mpfr_constant
from ..function import Function
from ..runtime.env import ForeignEnv
from ..ir import *
from ..utils import digits_to_fraction

from .interpreter import Interpreter, FunctionReturnException

ScalarVal: TypeAlias = bool | float
"""Type of scalar values."""
TensorVal: TypeAlias = tuple
"""Type of tensor values."""

def _safe_div(x: float, y: float):
    if y == 0:
        if x == 0:
            return math.nan
        else:
            return math.copysign(math.inf, x)
    else:
        return x / y


_method_table: dict[str, Callable[..., Any]] = {
    '+': lambda x, y: x + y,
    '-': lambda x, y: x - y,
    '*': lambda x, y: x * y,
    '/': _safe_div,
    'fabs': math.fabs,
    'sqrt': math.sqrt,
    # TODO: only available in Python 3.13
    # 'fma': math.fma,
    'neg': lambda x: -x,
    'copysign': math.copysign,
    'fdim': lambda x, y: max(x - y, 0),
    'fmax': max,
    'fmin': min,
    'fmod': math.fmod,
    'remainder': math.remainder,
    'hypot': math.hypot,
    'cbrt': math.cbrt,
    'ceil': math.ceil,
    'floor': math.floor,
    'nearbyint': lambda x: round(x),
    'round': round,
    'trunc': math.trunc,
    'acos': math.acos,
    'asin': math.asin,
    'atan': math.atan,
    'atan2': math.atan2,
    'cos': math.cos,
    'sin': math.sin,
    'tan': math.tan,
    'acosh': math.acosh,
    'asinh': math.asinh,
    'atanh': math.atanh,
    'cosh': math.cosh,
    'sinh': math.sinh,
    'tanh': math.tanh,
    'exp': math.exp,
    'exp2': lambda x: 2 ** x,
    'expm1': math.expm1,
    'log': math.log,
    'log10': math.log10,
    'log1p': math.log1p,
    'log2': math.log2,
    'pow': math.pow,
    'erf': math.erf,
    'erfc': math.erfc,
    'lgamma': math.lgamma,
    'tgamma': math.gamma,
    'isfinite': math.isfinite,
    'isinf': math.isinf,
    'isnan': math.isnan,
    'isnormal': lambda x: math.isfinite(x) and x != 0,
    'signbit': lambda x: math.copysign(1, x) < 0,
}

_Env: TypeAlias = dict[NamedId, ScalarVal | TensorVal]

_PY_CTX = IEEEContext(11, 64, RM.RNE)
"""the native Python floating-point context"""


class _Interpreter(ReduceVisitor):
    """Single-use interpreter for a function."""

    foreign: ForeignEnv
    """foreign environment"""
    env: _Env
    """environment mapping variable names to values"""

    def __init__(self, foreign: ForeignEnv, *, env: Optional[_Env] = None):
        if env is None:
            env = {}

        self.foreign = foreign
        self.env = env

    def _is_python_ctx(self, ctx: Context):
        return ctx == _PY_CTX

    def _arg_to_float(self, arg: Any):
        match arg:
            case int() | float():
                return arg
            case str() | Float():
                return float(arg)
            case tuple() | list():
                raise NotImplementedError('cannot convert tuple or list to float')
            case _:
                return arg

    def _lookup(self, name: NamedId):
        if name not in self.env:
            raise RuntimeError(f'unbound variable {name}')
        return self.env[name]

    def eval(
        self,
        func: FuncDef,
        args: Sequence[Any],
        ctx: Optional[Context] = None
    ):
        args = tuple(args)
        if len(args) != len(func.args):
            raise TypeError(f'Expected {len(func.args)} arguments, got {len(args)}')

        # TODO: what context is set when entering the FPy runtime?
        # determine context when specified
        if ctx is None:
            ctx = _PY_CTX

        # Python only has doubles
        if not self._is_python_ctx(ctx):
            raise RuntimeError(f'Unsupported context {ctx}, expected {_PY_CTX}')

        # bind arguments
        for val, arg in zip(args, func.args):
            match arg.ty:
                case AnyType():
                    x = self._arg_to_float(val)
                    if isinstance(arg.name, NamedId):
                        self.env[arg.name] = x
                case RealType():
                    x = self._arg_to_float(val)
                    if not isinstance(x, float):
                        raise NotImplementedError(f'argument is a scalar, got data {val}')
                    if isinstance(arg.name, NamedId):
                        self.env[arg.name] = x
                case _:
                    raise NotImplementedError(f'unknown argument type {arg.ty}')

        # process free variables
        for var in func.free_vars:
            x = self._arg_to_float(self.foreign[var.base])
            self.env[var] = x

        # evaluate the body
        try:
            self._visit_block(func.body, ctx)
            raise RuntimeError('no return statement encountered')
        except FunctionReturnException as e:
            return e.value

    def _visit_var(self, e: Var, ctx: Context):
        return self._lookup(e.name)

    def _visit_bool(self, e: BoolVal, ctx: Context):
        return e.val

    def _visit_foreign(self, e: ForeignVal, ctx: None):
        return e.val

    def _visit_decnum(self, e: Decnum, ctx: Context):
        return float(e.val)

    def _visit_hexnum(self, e: Hexnum, ctx: Context):
        return float.fromhex(e.val)

    def _visit_integer(self, e: Integer, ctx: Context):
        return float(e.val)

    def _visit_rational(self, e: Rational, ctx: Context):
        return e.p / e.q

    def _visit_constant(self, e: Constant, ctx: Context):
        prec, _ = ctx.round_params()
        assert isinstance(prec, int)
        x = mpfr_constant(e.val, prec=prec)
        return float(_PY_CTX.round(x))

    def _visit_digits(self, e: Digits, ctx: Context):
        # rely on Titanic for this
        return float(digits_to_fraction(e.m, e.e, e.b))

    def _visit_unknown(self, e: UnknownCall, ctx: Context):
        raise NotImplementedError('unknown call', e)

    def _apply_method(self, e: NaryExpr, ctx: Context):
        fn = _method_table[e.name]
        args: list[float] = []
        for arg in e.children:
            val = self._visit_expr(arg, ctx)
            if not isinstance(val, float):
                raise TypeError(f'expected a real number argument for {e.name}, got {val}')
            args.append(val)
        try:
            result = fn(*args)
        except OverflowError:
            # We could return an infinity, but we don't know which one
            result = math.nan
        except ValueError:
            # domain error means NaN
            result = math.nan

        return result

    def _apply_cast(self, e: Cast, ctx: Context):
        x = self._visit_expr(e.children[0], ctx)
        if not isinstance(x, float):
            raise TypeError(f'expected a float, got {x}')
        return x

    def _apply_not(self, e: Not, ctx: Context):
        arg = self._visit_expr(e.children[0], ctx)
        if not isinstance(arg, bool):
            raise TypeError(f'expected a boolean argument, got {arg}')
        return not arg

    def _apply_and(self, e: And, ctx: Context):
        args: list[bool] = []
        for arg in e.children:
            val = self._visit_expr(arg, ctx)
            if not isinstance(val, bool):
                raise TypeError(f'expected a boolean argument, got {val}')
            args.append(val)
        return all(args)

    def _apply_or(self, e: Or, ctx: Context):
        args: list[bool] = []
        for arg in e.children:
            val = self._visit_expr(arg, ctx)
            if not isinstance(val, bool):
                raise TypeError(f'expected a boolean argument, got {val}')
            args.append(val)
        return any(args)

    def _apply_range(self, e: Range, ctx: Context):
        stop = self._visit_expr(e.children[0], ctx)
        if not isinstance(stop, float):
            raise TypeError(f'expected a real number argument, got {stop}')
        if not stop.is_integer():
            raise TypeError(f'expected an integer argument, got {stop}')
        return tuple([float(i) for i in range(int(stop))])

    def _visit_nary_expr(self, e: NaryExpr, ctx: Context):
        if e.name in _method_table:
            return self._apply_method(e, ctx)
        elif e.name == 'fma':
            raise NotImplementedError('fma not supported in Python 3.11')
        elif isinstance(e, Cast):
            return self._apply_cast(e, ctx)
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

    def _apply_cmp2(self, op: CompareOp, lhs, rhs):
        match op:
            case CompareOp.EQ:
                return lhs == rhs
            case CompareOp.NE:
                return lhs != rhs
            case CompareOp.LT:
                return lhs < rhs
            case CompareOp.LE:
                return lhs <= rhs
            case CompareOp.GT:
                return lhs > rhs
            case CompareOp.GE:
                return lhs >= rhs
            case _:
                raise NotImplementedError('unknown comparison operator', op)

    def _visit_compare(self, e: Compare, ctx: Context):
        lhs = self._visit_expr(e.children[0], ctx)
        for op, arg in zip(e.ops, e.children[1:]):
            rhs = self._visit_expr(arg, ctx)
            if not self._apply_cmp2(op, lhs, rhs):
                return False
            lhs = rhs
        return True

    def _visit_tuple_expr(self, e: TupleExpr, ctx: Context):
        return tuple([self._visit_expr(x, ctx) for x in e.children])

    def _visit_tuple_ref(self, e: TupleRef, ctx: Context):
        value = self._visit_expr(e.value, ctx)
        if not isinstance(value, tuple):
            raise TypeError(f'expected a tensor, got {value}')

        elt = value
        for s in e.slices:
            val = self._visit_expr(s, ctx)
            if not isinstance(val, float):
                raise TypeError(f'expected a real number slice, got {val}')
            if not val.is_integer():
                raise TypeError(f'expected an integer slice, got {val}')
            elt = elt[int(val)]

        return elt

    def _visit_tuple_set(self, e: TupleSet, ctx: Context):
        raise NotImplementedError

    def _apply_comp(
        self,
        bindings: list[tuple[Id, Expr]],
        elt: Expr,
        ctx: Context,
        elts: list[Any]
    ):
        if bindings == []:
            elts.append(self._visit_expr(elt, ctx))
        else:
            var, iterable = bindings[0]
            array = self._visit_expr(iterable, ctx)
            if not isinstance(array, tuple):
                raise TypeError(f'expected a tensor, got {array}')
            for val in array:
                if isinstance(var, NamedId):
                    self.env[var] = val
                self._apply_comp(bindings[1:], elt, ctx, elts)

    def _visit_comp_expr(self, e: CompExpr, ctx: Context):
        # evaluate comprehension
        elts: list[Any] = []
        bindings = [(var, iterable) for var, iterable in zip(e.vars, e.iterables)]
        self._apply_comp(bindings, e.elt, ctx, elts)

        # remove temporarily bound variables
        for var in e.vars:
            if isinstance(var, NamedId):
                del self.env[var]
 
        # the result
        return tuple(elts)

    def _visit_if_expr(self, e: IfExpr, ctx: Context):
        cond = self._visit_expr(e.cond, ctx)
        if not isinstance(cond, bool):
            raise TypeError(f'expected a boolean, got {cond}')
        return self._visit_expr(e.ift if cond else e.iff, ctx)

    def _visit_simple_assign(self, stmt: SimpleAssign, ctx: Context):
        val = self._visit_expr(stmt.expr, ctx)
        match stmt.var:
            case NamedId():
                self.env[stmt.var] = val
            case UnderscoreId():
                pass
            case _:
                raise NotImplementedError('unknown variable', stmt.var)

    def _unpack_tuple(self, binding: TupleBinding, val: tuple, ctx: Context) -> None:
        if len(binding.elts) != len(val):
            raise NotImplementedError(f'unpacking {len(val)} values into {len(binding.elts)}')
        for elt, v in zip(binding.elts, val):
            match elt:
                case NamedId():
                    self.env[elt] = v
                case UnderscoreId():
                    pass
                case TupleBinding():
                    self._unpack_tuple(elt, v, ctx)
                case _:
                    raise NotImplementedError('unknown tuple element', elt)

    def _visit_tuple_unpack(self, stmt: TupleUnpack, ctx: Context):
        val = self._visit_expr(stmt.expr, ctx)
        if not isinstance(val, tuple):
            raise TypeError(f'expected a tuple, got {val}')
        self._unpack_tuple(stmt.binding, val, ctx)

    def _visit_index_assign(self, stmt: IndexAssign, ctx: Context):
        raise NotImplementedError

    def _visit_if1(self, stmt: If1Stmt, ctx: Context):
        cond = self._visit_expr(stmt.cond, ctx)
        if not isinstance(cond, bool):
            raise TypeError(f'expected a boolean, got {cond}')
        elif cond:
            self._visit_block(stmt.body, ctx)
            for phi in stmt.phis:
                self.env[phi.name] = self.env[phi.rhs]
                del self.env[phi.rhs]
        else:
            for phi in stmt.phis:
                self.env[phi.name] = self.env[phi.lhs]
                del self.env[phi.lhs]

    def _visit_if(self, stmt: IfStmt, ctx: Context):
        cond = self._visit_expr(stmt.cond, ctx)
        if not isinstance(cond, bool):
            raise TypeError(f'expected a boolean, got {cond}')
        elif cond:
            self._visit_block(stmt.ift, ctx)
            for phi in stmt.phis:
                self.env[phi.name] = self.env[phi.lhs]
                del self.env[phi.lhs]
        else:
            self._visit_block(stmt.iff, ctx)
            for phi in stmt.phis:
                self.env[phi.name] = self.env[phi.rhs]
                del self.env[phi.rhs]

    def _visit_while(self, stmt: WhileStmt, ctx: Context):
        for phi in stmt.phis:
            self.env[phi.name] = self.env[phi.lhs]
            del self.env[phi.lhs]

        cond = self._visit_expr(stmt.cond, ctx)
        if not isinstance(cond, bool):
            raise TypeError(f'expected a boolean, got {cond}')

        while cond:
            self._visit_block(stmt.body, ctx)
            for phi in stmt.phis:
                self.env[phi.name] = self.env[phi.rhs]
                del self.env[phi.rhs]

            cond = self._visit_expr(stmt.cond, ctx)
            if not isinstance(cond, bool):
                raise TypeError(f'expected a boolean, got {cond}')

    def _visit_for(self, stmt: ForStmt, ctx: Context):
        for phi in stmt.phis:
            self.env[phi.name] = self.env[phi.lhs]
            del self.env[phi.lhs]

        iterable = self._visit_expr(stmt.iterable, ctx)
        if not isinstance(iterable, tuple):
            raise TypeError(f'expected a tensor, got {iterable}')

        for val in iterable:
            if isinstance(stmt.var, NamedId):
                self.env[stmt.var] = val
            self._visit_block(stmt.body, ctx)
            for phi in stmt.phis:
                self.env[phi.name] = self.env[phi.rhs]
                del self.env[phi.rhs]

    def _visit_foreign_attr(self, e: ForeignAttribute):
        # lookup the root value (should be captured)
        val = self._lookup(e.name)
        # walk the attribute chain
        for attr_id in e.attrs:
            # need to manually lookup the attribute
            attr = str(attr_id)
            if isinstance(val, dict):
                if attr not in val:
                    raise RuntimeError(f'unknown attribute {attr} for {val}')
                val = val[attr]
            elif hasattr(val, attr):
                val = getattr(val, attr)
            else:
                raise RuntimeError(f'unknown attribute {attr} for {val}')
        return val

    def _visit_context_expr(self, e: ContextExpr, ctx: Context):
        match e.ctor:
            case ForeignAttribute():
                ctor = self._visit_foreign_attr(e.ctor)
            case Var():
                ctor = self._visit_var(e.ctor, ctx)

        args: list[Any] = []
        for arg in e.args:
            match arg:
                case ForeignAttribute():
                    args.append(self._visit_foreign_attr(arg))
                case _:
                    v = self._visit_expr(arg, ctx)
                    if isinstance(v, float) and v.is_integer():
                        # HACK: keeps things as specific as possible
                        args.append(int(v))
                    else:
                        args.append(v)
        return ctor(*args)

    def _visit_context(self, stmt: ContextStmt, ctx: Context):
        ctx = self._visit_expr(stmt.ctx, ctx)
        if not self._is_python_ctx(ctx):
            raise RuntimeError(f'Unsupported context {ctx}, expected {_PY_CTX}')
        return self._visit_block(stmt.body, ctx)

    def _visit_assert(self, stmt: AssertStmt, ctx: Context):
        test = self._visit_expr(stmt.test, ctx)
        if not isinstance(test, bool):
            raise TypeError(f'expected a boolean, got {test}')
        if not test:
            raise AssertionError(stmt.msg)
        return ctx

    def _visit_effect(self, stmt, ctx):
        self._visit_expr(stmt.expr, ctx)
        return ctx

    def _visit_return(self, stmt: ReturnStmt, ctx: Context):
        return self._visit_expr(stmt.expr, ctx)

    def _visit_block(self, block: StmtBlock, ctx: Context):
        for stmt in block.stmts:
            if isinstance(stmt, ReturnStmt):
                x = self._visit_return(stmt, ctx)
                raise FunctionReturnException(x)
            self._visit_statement(stmt, ctx)

    def _visit_function(self, func: FuncDef, ctx: Context):
        raise NotImplementedError('do not call directly')


class PythonInterpreter(Interpreter):
    """
    Python-backed interpreter for FPy programs.

    Programs are evaluated using Python's `math` library.
    Booleans are Python `bool` values, real numbers are `float` values,
    and tensors are Python `tuple` values.
    """

    def eval(
        self,
        func: Function,
        args: Sequence[Any],
        ctx: Optional[Context] = None
    ):
        if not isinstance(func, Function):
            raise TypeError(f'Expected Function, got {func}')
        rt = _Interpreter(func.env)
        return rt.eval(func.to_ir(), args, ctx)

    def eval_with_trace(self, func: Function, args: Sequence[Any], ctx: Optional[Context] = None):
        raise NotImplementedError('not implemented')

    def eval_expr(self, expr: Expr, env: _Env, ctx: Context):
        raise NotImplementedError('not implemented')
