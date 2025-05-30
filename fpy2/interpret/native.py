"""
FPy runtime backed by the Python runtime.
"""

import math

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, TypeAlias

from ..ast import *
from ..number import Context, Float, IEEEContext, RM
from ..number.gmp import mpfr_constant
from ..function import Function
from ..env import ForeignEnv
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

_unary_table: dict[type[UnaryOp], Callable[[float], float | bool]] = {
    Neg: lambda x: -x,
    Fabs: math.fabs,
    Sqrt: math.sqrt,
    Cbrt: math.cbrt,
    Ceil: math.ceil,
    Floor: math.floor,
    NearbyInt: lambda x: round(x),
    Round: round,
    Trunc: math.trunc,
    Acos: math.acos,
    Asin: math.asin,
    Atan: math.atan,
    Cos: math.cos,
    Sin: math.sin,
    Tan: math.tan,
    Acosh: math.acosh,
    Asinh: math.asinh,
    Atanh: math.atanh,
    Cosh: math.cosh,
    Sinh: math.sinh,
    Tanh: math.tanh,
    Exp: math.exp,
    Exp2: lambda x: 2 ** x,
    Expm1: math.expm1,
    Log: math.log,
    Log10: math.log10,
    Log1p: math.log1p,
    Log2: math.log2,
    Erf: math.erf,
    Erfc: math.erfc,
    Lgamma: math.lgamma,
    Tgamma: math.gamma,
    IsFinite: math.isfinite,
    IsInf: math.isinf,
    IsNan: math.isnan,
    IsNormal: lambda x: math.isfinite(x) and x != 0,
    Signbit: lambda x: math.copysign(1, x) < 0,
}

_binary_table: dict[type[BinaryOp], Callable[[float, float], float]] = {
    Add: lambda x, y: x + y,
    Sub: lambda x, y: x - y,
    Mul: lambda x, y: x * y,
    Div: _safe_div,
    Copysign: math.copysign,
    Fdim: lambda x, y: max(x - y, 0),
    Fmax: max,
    Fmin: min,
    Fmod: math.fmod,
    Remainder: math.remainder,
    Hypot: math.hypot,
    Atan2: math.atan2,
    Pow: math.pow,
}


_PY_CTX = IEEEContext(11, 64, RM.RNE)
"""the native Python floating-point context"""

_Env: TypeAlias = dict[NamedId, ScalarVal | TensorVal]
"""the native Python floating-point context"""

@dataclass
class _EvalCtx:
    env: _Env
    """mapping from variable names to values"""
    round_ctx: Context
    """rounding context for evaluation"""


class _Interpreter(Visitor):
    """Single-use interpreter for a function."""

    foreign: ForeignEnv
    """foreign environment"""

    def __init__(self, foreign: ForeignEnv):
        self.foreign = foreign

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

    def _lookup(self, name: NamedId, ctx: _EvalCtx):
        if name not in ctx.env:
            raise RuntimeError(f'unbound variable {name}')
        return ctx.env[name]

    def eval(
        self,
        func: FuncDef,
        args: Sequence[Any],
        ctx: Context
    ):
        args = tuple(args)
        if len(args) != len(func.args):
            raise TypeError(f'Expected {len(func.args)} arguments, got {len(args)}')

        # Python only has doubles
        if not self._is_python_ctx(ctx):
            raise RuntimeError(f'Unsupported context {ctx}, expected {_PY_CTX}')

        # create the environment
        eval_ctx = _EvalCtx({}, ctx)

        # bind arguments
        for val, arg in zip(args, func.args):
            match arg.type:
                case AnyTypeAnn():
                    x = self._arg_to_float(val)
                    if isinstance(arg.name, NamedId):
                        eval_ctx.env[arg.name] = x
                case RealTypeAnn():
                    x = self._arg_to_float(val)
                    if not isinstance(x, float):
                        raise NotImplementedError(f'argument is a scalar, got data {val}')
                    if isinstance(arg.name, NamedId):
                        eval_ctx.env[arg.name] = x
                case _:
                    raise NotImplementedError(f'unknown argument type {arg.type}')

        # process free variables
        for var in func.free_vars:
            x = self._arg_to_float(self.foreign[var.base])
            eval_ctx.env[var] = x

        # evaluate the body
        try:
            self._visit_block(func.body, eval_ctx)
            raise RuntimeError('no return statement encountered')
        except FunctionReturnException as e:
            return e.value

    def _visit_var(self, e: Var, ctx: _EvalCtx):
        return self._lookup(e.name, ctx)

    def _visit_bool(self, e: BoolVal, ctx: _EvalCtx):
        return e.val

    def _visit_foreign(self, e: ForeignVal, ctx: None):
        return e.val

    def _visit_decnum(self, e: Decnum, ctx: _EvalCtx):
        return float(e.val)

    def _visit_hexnum(self, e: Hexnum, ctx: _EvalCtx):
        return float.fromhex(e.val)

    def _visit_integer(self, e: Integer, ctx: _EvalCtx):
        return float(e.val)

    def _visit_rational(self, e: Rational, ctx: _EvalCtx):
        return e.p / e.q

    def _visit_constant(self, e: Constant, ctx: _EvalCtx):
        prec, _ = ctx.round_ctx.round_params()
        assert isinstance(prec, int)
        x = mpfr_constant(e.val, prec=prec)
        return float(_PY_CTX.round(x))

    def _visit_digits(self, e: Digits, ctx: _EvalCtx):
        return float(digits_to_fraction(e.m, e.e, e.b))

    def _visit_call(self, e: Call, ctx: _EvalCtx):
        raise NotImplementedError('unknown call', e)

    def _apply_method(self, fn: Callable[..., Any], args: Sequence[Expr], ctx: _EvalCtx):
        vals: list[float] = []
        for arg in args:
            val = self._visit_expr(arg, ctx)
            if not isinstance(val, float):
                raise TypeError(f'expected a real number argument, got {val}')
            vals.append(val)
        try:
            result = fn(*vals)
        except OverflowError:
            # We could return an infinity, but we don't know which one
            result = math.nan
        except ValueError:
            # domain error means NaN
            result = math.nan

        return result

    def _apply_cast(self, arg: Expr, ctx: _EvalCtx):
        x = self._visit_expr(arg, ctx)
        if not isinstance(x, float):
            raise TypeError(f'expected a float, got {x}')
        return x

    def _apply_not(self, arg: Expr, ctx: _EvalCtx):
        arg = self._visit_expr(arg, ctx)
        if not isinstance(arg, bool):
            raise TypeError(f'expected a boolean argument, got {arg}')
        return not arg

    def _apply_and(self, args: Sequence[Expr], ctx: _EvalCtx):
        vals: list[bool] = []
        for arg in args:
            val = self._visit_expr(arg, ctx)
            if not isinstance(val, bool):
                raise TypeError(f'expected a boolean argument, got {val}')
            vals.append(val)
        return all(vals)

    def _apply_or(self, args: Sequence[Expr], ctx: _EvalCtx):
        vals: list[bool] = []
        for arg in args:
            val = self._visit_expr(arg, ctx)
            if not isinstance(val, bool):
                raise TypeError(f'expected a boolean argument, got {val}')
            vals.append(val)
        return any(vals)

    def _apply_range(self, arg: Expr, ctx: _EvalCtx):
        stop = self._visit_expr(arg, ctx)
        if not isinstance(stop, float):
            raise TypeError(f'expected a real number argument, got {stop}')
        if not stop.is_integer():
            raise TypeError(f'expected an integer argument, got {stop}')
        return [float(i) for i in range(int(stop))]

    def _tensor_shape(self, v: list) -> tuple[int, ...]:
        """
        Computes the shape of a tensor represented as a nested tuple.
        Returns a tuple of floats representing the shape.
        """
        shape: list[int] = []
        while isinstance(v, list):
            size = len(v)
            shape.append(size)
            if size == 0:
                # empty tensor
                break
            else:
                v = v[0] # assuming a flat structure for simplicity
        return tuple(shape)

    def _apply_shape(self, arg: Expr, ctx: _EvalCtx):
        v = self._visit_expr(arg, ctx)
        if not isinstance(v, list):
            raise TypeError(f'expected a tensor, got {v}')
        # compute shape of tensor
        shape = self._tensor_shape(v)
        return list([float(s) for s in shape])

    def _apply_dim(self, arg: Expr, ctx: _EvalCtx):
        v = self._visit_expr(arg, ctx)
        if not isinstance(v, list):
            raise TypeError(f'expected a tensor, got {v}')
        # compute shape of tensor
        shape = self._tensor_shape(v)
        return float(len(shape))

    def _apply_size(self, arr: Expr, idx: Expr, ctx: _EvalCtx):
        v = self._visit_expr(arr, ctx)
        if not isinstance(v, list):
            raise TypeError(f'expected a tensor, got {v}')
        dim = self._visit_expr(idx, ctx)
        if not isinstance(dim, float):
            raise TypeError(f'expected a real number argument, got {dim}')
        if not dim.is_integer():
            raise TypeError(f'expected an integer argument, got {dim}')
        # compute shape of tensor
        shape = self._tensor_shape(v)
        return float(shape[int(dim)])

    def _apply_zip(self, args: Sequence[Expr], ctx: _EvalCtx):
        """Apply the `zip` method to the given n-ary expression."""
        if len(args) == 0:
            return ()

        # evaluate all children
        arrays: list[list] = []
        for arg in args:
            val = self._visit_expr(arg, ctx)
            if not isinstance(val, list):
                raise TypeError(f'expected a tensor argument, got {val}')
            arrays.append(val)

        # zip the arrays
        return list(zip(*arrays))

    def _visit_unaryop(self, e: UnaryOp, ctx: _EvalCtx):
        fn = _unary_table.get(type(e))
        if fn is not None:
            return self._apply_method(fn, (e.arg,), ctx)
        else:
            match e:
                case Cast():
                    return self._apply_cast(e.arg, ctx)
                case Not():
                    return self._apply_not(e.arg, ctx)
                case Range():
                    return self._apply_range(e.arg, ctx)
                case Shape():
                    return self._apply_shape(e.arg, ctx)
                case Dim():
                    return self._apply_dim(e.arg, ctx)
                case _:
                    raise RuntimeError('unknown operator', e)

    def _visit_binaryop(self, e: BinaryOp, ctx: _EvalCtx):
        fn = _binary_table.get(type(e))
        if fn is not None:
            return self._apply_method(fn, (e.first, e.second), ctx)
        else:
            match e:
                case Size():
                    return self._apply_size(e.first, e.second, ctx)
                case _:
                    raise RuntimeError('unknown operator', e)

    def _visit_ternaryop(self, e: TernaryOp, ctx: _EvalCtx):
        match e:
            case Fma():
                raise NotImplementedError('fma not supported in Python 3.11')
            case _:
                raise RuntimeError('unknown operator', e)

        return super()._visit_ternaryop(e, ctx)

    def _visit_naryop(self, e: NaryOp, ctx: _EvalCtx):
        match e:
            case And():
                return self._apply_and(e.args, ctx)
            case Or():
                return self._apply_or(e.args, ctx)
            case Zip():
                return self._apply_zip(e.args, ctx)
            case _:
                raise RuntimeError('unknown operator', e)

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

    def _visit_compare(self, e: Compare, ctx: _EvalCtx):
        lhs = self._visit_expr(e.args[0], ctx)
        for op, arg in zip(e.ops, e.args[1:]):
            rhs = self._visit_expr(arg, ctx)
            if not self._apply_cmp2(op, lhs, rhs):
                return False
            lhs = rhs
        return True

    def _visit_tuple_expr(self, e: TupleExpr, ctx: _EvalCtx):
        # TODO: check for ragged arrays
        return list([self._visit_expr(x, ctx) for x in e.args])

    def _visit_tuple_ref(self, e: TupleRef, ctx: _EvalCtx):
        value = self._visit_expr(e.value, ctx)
        if not isinstance(value, list):
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

    def _visit_tuple_set(self, e: TupleSet, ctx: _EvalCtx):
        raise NotImplementedError

    def _apply_comp(
        self,
        bindings: list[tuple[Id | TupleBinding, Expr]],
        elt: Expr,
        ctx: _EvalCtx,
        elts: list[Any]
    ):
        if bindings == []:
            elts.append(self._visit_expr(elt, ctx))
        else:
            target, iterable = bindings[0]
            array = self._visit_expr(iterable, ctx)
            if not isinstance(array, list):
                raise TypeError(f'expected a tensor, got {array}')
            for val in array:
                match target:
                    case NamedId():
                        ctx.env[target] = val
                    case TupleBinding():
                        self._unpack_tuple(target, val, ctx)
                    case _:
                        raise RuntimeError('unreachable', target)
                self._apply_comp(bindings[1:], elt, ctx, elts)

    def _visit_comp_expr(self, e: CompExpr, ctx: _EvalCtx):
        # evaluate comprehension
        elts: list[Any] = []
        bindings = list(zip(e.targets, e.iterables))
        self._apply_comp(bindings, e.elt, ctx, elts)

        # remove temporarily bound variables
        for target in e.targets:
            for name in target.names():
                del ctx.env[name]
        # the result
        return tuple(elts)

    def _visit_if_expr(self, e: IfExpr, ctx: _EvalCtx):
        cond = self._visit_expr(e.cond, ctx)
        if not isinstance(cond, bool):
            raise TypeError(f'expected a boolean, got {cond}')
        return self._visit_expr(e.ift if cond else e.iff, ctx)

    def _unpack_tuple(self, binding: TupleBinding, val: list, ctx: _EvalCtx) -> None:
        if len(binding.elts) != len(val):
            raise NotImplementedError(f'unpacking {len(val)} values into {len(binding.elts)}')
        for elt, v in zip(binding.elts, val):
            match elt:
                case NamedId():
                    ctx.env[elt] = v
                case UnderscoreId():
                    pass
                case TupleBinding():
                    self._unpack_tuple(elt, v, ctx)
                case _:
                    raise NotImplementedError('unknown tuple element', elt)

    def _visit_assign(self, stmt: Assign, ctx: _EvalCtx):
        val = self._visit_expr(stmt.expr, ctx)
        match stmt.binding:
            case NamedId():
                ctx.env[stmt.binding] = val
            case TupleBinding():
                self._unpack_tuple(stmt.binding, val, ctx)

    def _visit_indexed_assign(self, stmt: IndexedAssign, ctx: _EvalCtx):
        # lookup the array
        array0 = self._lookup(stmt.var, ctx)

        # evaluate indices
        array = array0
        dim = len(stmt.slices)
        for i, s in enumerate(stmt.slices):
            val = self._visit_expr(s, ctx)
            if not isinstance(array, list):
                raise TypeError(f'expected a tensor, got {array0}')
            if not isinstance(val, float):
                raise TypeError(f'expected a real number slice, got {val}')
            if not val.is_integer():
                raise TypeError(f'expected an integer slice, got {val}')

            if i == dim - 1:
                # last slice: evaluate and update array
                val = self._visit_expr(stmt.expr, ctx)
                array[int(val)] = val
            else:
                # intermediate slice: update array to the next dimension
                array = array[int(val)]

    def _visit_if1(self, stmt: If1Stmt, ctx: _EvalCtx):
        cond = self._visit_expr(stmt.cond, ctx)
        if not isinstance(cond, bool):
            raise TypeError(f'expected a boolean, got {cond}')
        elif cond:
            # evaluate the if-true branch
            ift_ctx = self._visit_block(stmt.body, ctx)
            # update the environment with the values from the ift context
            # a variable cannot be introduced within the branch
            for var in ctx.env:
                ctx.env[var] = ift_ctx.env[var]

    def _visit_if(self, stmt: IfStmt, ctx: _EvalCtx):
        cond = self._visit_expr(stmt.cond, ctx)
        if not isinstance(cond, bool):
            raise TypeError(f'expected a boolean, got {cond}')

        if cond:
            # evaluate the if-true branch
            ift_ctx = self._visit_block(stmt.ift, ctx)
            # TODO: we should take the intserction of defined variables on either branch
            ctx.env = ift_ctx.env
        else:
            # evaluate the if-false branch
            iff_ctx = self._visit_block(stmt.iff, ctx)
            # TODO: we should take the intserction of defined variables on either branch
            ctx.env = iff_ctx.env

    def _visit_while(self, stmt: WhileStmt, ctx: _EvalCtx):
        # evaluate the condition
        cond = self._visit_expr(stmt.cond, ctx)
        if not isinstance(cond, bool):
            raise TypeError(f'expected a boolean, got {cond}')

        while cond:
            # evaluate the while body
            while_ctx = self._visit_block(stmt.body, ctx)
            # update the environment with the values from the while context
            # a variable cannot be introduced within the body
            for var in ctx.env:
                ctx.env[var] = while_ctx.env[var]
            # evaluate the condition
            cond = self._visit_expr(stmt.cond, ctx)
            if not isinstance(cond, bool):
                raise TypeError(f'expected a boolean, got {cond}')

    def _visit_for(self, stmt: ForStmt, ctx: _EvalCtx):
        # evaluate the iterable data
        iterable = self._visit_expr(stmt.iterable, ctx)
        if not isinstance(iterable, list):
            raise TypeError(f'expected a tensor, got {iterable}')
        # iterate over each element
        for val in iterable:
            # bind the value to the target variable
            body_ctx = _EvalCtx(ctx.env.copy(), ctx.round_ctx)
            match stmt.target:
                case NamedId():
                    body_ctx.env[stmt.target] = val
                case TupleBinding():
                    self._unpack_tuple(stmt.target, val, body_ctx)
            # evaluate the body of the loop
            body_ctx = self._visit_block(stmt.body, body_ctx)
            # update the environment with the values from the body context
            # a variable cannot be introduced within the body
            for var in ctx.env:
                ctx.env[var] = body_ctx.env[var]

    def _visit_foreign_attr(self, e: ForeignAttribute, ctx: _EvalCtx):
        # lookup the root value (should be captured)
        val = self._lookup(e.name, ctx)
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

    def _visit_context_expr(self, e: ContextExpr, ctx: _EvalCtx):
        match e.ctor:
            case ForeignAttribute():
                ctor = self._visit_foreign_attr(e.ctor, ctx)
            case Var():
                ctor = self._visit_var(e.ctor, ctx)

        args: list[Any] = []
        for arg in e.args:
            match arg:
                case ForeignAttribute():
                    args.append(self._visit_foreign_attr(arg, ctx))
                case _:
                    v = self._visit_expr(arg, ctx)
                    if isinstance(v, float) and v.is_integer():
                        # HACK: keeps things as specific as possible
                        args.append(int(v))
                    else:
                        args.append(v)
        return ctor(*args)

    def _visit_context(self, stmt: ContextStmt, ctx: _EvalCtx):
        round_ctx = self._visit_expr(stmt.ctx, ctx)
        if not self._is_python_ctx(round_ctx):
            raise RuntimeError(f'Unsupported context {round_ctx}, expected {_PY_CTX}')
        return self._visit_block(stmt.body, ctx)

    def _visit_assert(self, stmt: AssertStmt, ctx: _EvalCtx):
        test = self._visit_expr(stmt.test, ctx)
        if not isinstance(test, bool):
            raise TypeError(f'expected a boolean, got {test}')
        if not test:
            raise AssertionError(stmt.msg)
        return ctx

    def _visit_effect(self, stmt, ctx):
        self._visit_expr(stmt.expr, ctx)
        return ctx

    def _visit_return(self, stmt: ReturnStmt, ctx: _EvalCtx):
        return self._visit_expr(stmt.expr, ctx)

    def _visit_block(self, block: StmtBlock, ctx: _EvalCtx):
        ctx = _EvalCtx(ctx.env.copy(), ctx.round_ctx)
        for stmt in block.stmts:
            if isinstance(stmt, ReturnStmt):
                x = self._visit_return(stmt, ctx)
                raise FunctionReturnException(x)
            self._visit_statement(stmt, ctx)
        return ctx

    def _visit_function(self, func: FuncDef, ctx: _EvalCtx):
        raise NotImplementedError('do not call directly')


class PythonInterpreter(Interpreter):
    """
    Python-backed interpreter for FPy programs.

    Programs are evaluated using Python's `math` library.
    Booleans are Python `bool` values, real numbers are `float` values,
    and tensors are Python `list` values.
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
        ctx = self._func_ctx(func, ctx)
        return rt.eval(func.ast, args, ctx)
