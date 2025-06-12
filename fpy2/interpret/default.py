"""
FPy runtime backed by the Titanic library.
"""

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Callable, Optional, Sequence, TypeAlias

from titanfp.titanic.ndarray import NDArray

from .. import math

from ..ast import *
from ..fpc_context import FPCoreContext
from ..number import Context, Float
from ..number.gmp import mpfr_constant
from ..env import ForeignEnv
from ..function import Function
from ..primitive import Primitive
from ..utils import decnum_to_fraction, hexnum_to_fraction, digits_to_fraction

from .interpreter import Interpreter, FunctionReturnException

ScalarVal: TypeAlias = bool | Float
"""Type of scalar values in FPy programs."""
TensorVal: TypeAlias = NDArray
"""Type of tensor values in FPy programs."""

ScalarArg: TypeAlias = ScalarVal | str | int | float
"""Type of scalar arguments in FPy programs; includes native Python types"""
TensorArg: TypeAlias = NDArray | tuple | list
"""Type of tensor arguments in FPy programs; includes native Python types"""

def _isfinite(x: Float, ctx: Context) -> bool:
    return x.is_finite()

def _isinf(x: Float, ctx: Context) -> bool:
    return x.isinf

def _isnan(x: Float, ctx: Context) -> bool:
    return x.isnan

def _isnormal(x: Float, ctx: Context) -> bool:
    return x.is_normal()

def _signbit(x: Float, ctx: Context) -> bool:
    # TODO: should all Floats have this property?
    return x.s

_unary_table: dict[type[UnaryOp], Callable[[Float, Context], Any]] = {
    Fabs: math.fabs,
    Sqrt: math.sqrt,
    Neg: math.neg,
    Cbrt: math.cbrt,
    Ceil: math.ceil,
    Floor: math.floor,
    NearbyInt: math.nearbyint,
    Round: math.round,
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
    Exp2: math.exp2,
    Expm1: math.expm1,
    Log: math.log,
    Log10: math.log10,
    Log1p: math.log1p,
    Log2: math.log2,
    Erf: math.erf,
    Erfc: math.erfc,
    Lgamma: math.lgamma,
    Tgamma: math.tgamma,
    IsFinite: _isfinite,
    IsInf: _isinf,
    IsNan: _isnan,
    IsNormal: _isnormal,
    Signbit: _signbit,
}

_binary_table: dict[type[BinaryOp], Callable[[Float, Float, Context], Any]] = {
    Add: math.add,
    Sub: math.sub,
    Mul: math.mul,
    Div: math.div,
    Copysign: math.copysign,
    Fdim: math.fdim,
    Fmax: math.fmax,
    Fmin: math.fmin,
    Fmod: math.fmod,
    Remainder: math.remainder,
    Hypot: math.hypot,
    Atan2: math.atan2,
    Pow: math.pow,
}

_ternary_table: dict[type[TernaryOp], Callable[[Float, Float, Float, Context], Any]] = {
    Fma: math.fma,
}

_Env: TypeAlias = dict[NamedId, ScalarVal | TensorVal]
"""Type of the environment used by the interpreter."""

@dataclass
class _EvalCtx:
    env: _Env
    """mapping from variable names to values"""
    round_ctx: Context
    """rounding context for evaluation"""


class _Interpreter(DefaultVisitor):
    """Single-use interpreter for a function"""

    foreign: ForeignEnv
    """foreign environment"""
    override_ctx: Optional[Context]
    """optional overriding context"""

    def __init__(
        self, 
        foreign: ForeignEnv,
        *,
        override_ctx: Optional[Context] = None,
    ):
        self.foreign = foreign
        self.override_ctx = override_ctx

    def _eval_ctx(self, env: _Env, ctx: Context | FPCoreContext):
        if self.override_ctx is not None:
            return _EvalCtx(env, self.override_ctx)
        match ctx:
            case Context():
                return _EvalCtx(env, ctx)
            case FPCoreContext():
                return _EvalCtx(env, ctx.to_context())
            case _:
                raise TypeError(f'Expected `Context` or `FPCoreContext`, got {ctx}')

    # TODO: what are the semantics of arguments
    def _arg_to_mpmf(self, arg: Any, ctx: _EvalCtx):
        match arg:
            case int():
                return Float.from_int(arg, ctx=ctx.round_ctx)
            case float():
                return Float.from_float(arg, ctx=ctx.round_ctx)
            case Float():
                return arg.round(ctx.round_ctx)
            case tuple() | list():
                return NDArray([self._arg_to_mpmf(x, ctx) for x in arg])
            case _:
                return arg

    def eval(
        self,
        func: FuncDef,
        args: Sequence[Any],
        ctx: Context
    ):
        # check arity
        args = tuple(args)
        if len(args) != len(func.args):
            raise TypeError(f'Expected {len(func.args)} arguments, got {len(args)}')

        # possibly override the context
        eval_ctx = self._eval_ctx({}, ctx)
        assert isinstance(ctx, Context)

        # process arguments and add to environment
        for val, arg in zip(args, func.args):
            match arg.type:
                case AnyTypeAnn() | None:
                    x = self._arg_to_mpmf(val, eval_ctx)
                    if isinstance(arg.name, NamedId):
                        eval_ctx.env[arg.name] = x
                case RealTypeAnn():
                    x = self._arg_to_mpmf(val, eval_ctx)
                    if not isinstance(x, Float):
                        raise NotImplementedError(f'argument is a scalar, got data {val}')
                    if isinstance(arg.name, NamedId):
                        eval_ctx.env[arg.name] = x
                case TensorTypeAnn():
                    # TODO: check shape
                    x = self._arg_to_mpmf(val, eval_ctx)
                    if not isinstance(x, NDArray):
                        raise NotImplementedError(f'argument is a tensor, got data {val}')
                    if isinstance(arg.name, NamedId):
                        eval_ctx.env[arg.name] = x
                case _:
                    raise NotImplementedError(f'unknown argument type {arg.type}')

        # process free variables
        for var in func.free_vars:
            x = self._arg_to_mpmf(self.foreign[var.base], eval_ctx)
            eval_ctx.env[var] = x

        # evaluation
        try:
            self._visit_block(func.body, eval_ctx)
            raise RuntimeError('no return statement encountered')
        except FunctionReturnException as e:
            return e.value

    def _lookup(self, name: NamedId, ctx: _EvalCtx):
        if name not in ctx.env:
            raise RuntimeError(f'unbound variable {name}')
        return ctx.env[name]

    def _visit_var(self, e: Var, ctx: _EvalCtx):
        return self._lookup(e.name, ctx)

    def _visit_bool(self, e: BoolVal, ctx: Any):
        return e.val

    def _visit_foreign(self, e: ForeignVal, ctx: None):
        return e.val

    def _visit_decnum(self, e: Decnum, ctx: _EvalCtx):
        x = decnum_to_fraction(e.val)
        return ctx.round_ctx.round(x)

    def _visit_integer(self, e: Integer, ctx: _EvalCtx):
        return ctx.round_ctx.round(e.val)

    def _visit_hexnum(self, e: Hexnum, ctx: _EvalCtx):
        x = hexnum_to_fraction(e.val)
        return ctx.round_ctx.round(x)

    def _visit_rational(self, e: Rational, ctx: _EvalCtx):
        x = Fraction(e.p, e.q)
        return ctx.round_ctx.round(x)

    def _visit_constant(self, e: Constant, ctx: _EvalCtx):
        prec, _ = ctx.round_ctx.round_params()
        assert isinstance(prec, int) # TODO: not every context produces has a known precision
        x = mpfr_constant(e.val, prec=prec)
        return ctx.round_ctx.round(x)

    def _visit_digits(self, e: Digits, ctx: _EvalCtx):
        x = digits_to_fraction(e.m, e.e, e.b)
        return ctx.round_ctx.round(x)

    def _apply_method(self, fn: Callable[..., Any], args: Sequence[Expr], ctx: _EvalCtx):
        # fn: Callable[[Float, ..., Context], Float]
        vals: list[Float] = []
        for arg in args:
            val = self._visit_expr(arg, ctx)
            if not isinstance(val, Float):
                raise TypeError(f'expected a real number argument, got {val}')
            vals.append(val)
        # compute the result
        return fn(*vals, ctx=ctx.round_ctx)

    def _apply_cast(self, arg: Expr, ctx: _EvalCtx):
        x = self._visit_expr(arg, ctx)
        if not isinstance(x, Float):
            raise TypeError(f'expected a real number argument, got {x}')
        return ctx.round_ctx.round(x)

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

    def _apply_shape(self, arg: Expr, ctx: _EvalCtx):
        v = self._visit_expr(arg, ctx)
        if not isinstance(v, NDArray):
            raise TypeError(f'expected a tensor, got {v}')
        return NDArray([ctx.round_ctx.round(x) for x in v.shape])

    def _apply_range(self, arg: Expr, ctx: _EvalCtx):
        stop = self._visit_expr(arg, ctx)
        if not isinstance(stop, Float):
            raise TypeError(f'expected a real number argument, got {stop}')
        if not stop.is_integer():
            raise TypeError(f'expected an integer argument, got {stop}')
        n = int(stop)

        elts: list[Float] = []
        for i in range(n):
            elts.append(Float.from_int(i, ctx=ctx.round_ctx))
        return NDArray(elts, shape=(n,))

    def _apply_dim(self, arg: Expr, ctx: _EvalCtx):
        v = self._visit_expr(arg, ctx)
        if not isinstance(v, NDArray):
            raise TypeError(f'expected a tensor, got {v}')
        return Float.from_int(len(v.shape), ctx=ctx.round_ctx)

    def _apply_size(self, arr: Expr, idx: Expr, ctx: _EvalCtx):
        v = self._visit_expr(arr, ctx)
        if not isinstance(v, NDArray):
            raise TypeError(f'expected a tensor, got {v}')
        dim = self._visit_expr(idx, ctx)
        if not isinstance(dim, Float):
            raise TypeError(f'expected a real number argument, got {dim}')
        if not dim.is_integer():
            raise TypeError(f'expected an integer argument, got {dim}')
        return Float.from_int(v.shape[int(dim)], ctx=ctx.round_ctx)

    def _apply_zip(self, args: Sequence[Expr], ctx: _EvalCtx):
        """Apply the `zip` method to the given n-ary expression."""
        if len(args) == 0:
            return NDArray([])

        # evaluate all children
        arrays: list[NDArray] = []
        for arg in args:
            val = self._visit_expr(arg, ctx)
            if not isinstance(val, NDArray):
                raise TypeError(f'expected a tensor argument, got {val}')
            arrays.append(val)

        # zip the arrays
        return NDArray(zip(*arrays))

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
        fn = _ternary_table.get(type(e))
        if fn is not None:
            return self._apply_method(fn, (e.first, e.second, e.third), ctx)
        else:
            raise RuntimeError('unknown operator', e)

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

    def _visit_call(self, e: Call, ctx: _EvalCtx):
        args = [self._visit_expr(arg, ctx) for arg in e.args]
        fn = self.foreign[e.name]
        match fn:
            case Function():
                # calling FPy function
                rt = _Interpreter(fn.env, override_ctx=self.override_ctx)
                return rt.eval(fn.ast, args, ctx.round_ctx)
            case Primitive():
                # calling FPy primitive
                return fn(*args, ctx=ctx.round_ctx)
            case _:
                # calling foreign function
                if not callable(fn):
                    raise RuntimeError(f'unknown function {e.name}')
                return fn(*args)

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
        return NDArray([self._visit_expr(x, ctx) for x in e.args])

    def _visit_tuple_ref(self, e: TupleRef, ctx: _EvalCtx):
        value = self._visit_expr(e.value, ctx)
        if not isinstance(value, NDArray):
            raise TypeError(f'expected a tensor, got {value}')

        slices: list[int] = []
        for s in e.slices:
            val = self._visit_expr(s, ctx)
            if not isinstance(val, Float):
                raise TypeError(f'expected a real number slice, got {val}')
            if not val.is_integer():
                raise TypeError(f'expected an integer slice, got {val}')
            slices.append(int(val))

        return value[slices]

    def _visit_tuple_set(self, e: TupleSet, ctx: _EvalCtx):
        value = self._visit_expr(e.array, ctx)
        if not isinstance(value, NDArray):
            raise TypeError(f'expected a tensor, got {value}')
        value = NDArray(value) # make a copy

        slices: list[int] = []
        for s in e.slices:
            val = self._visit_expr(s, ctx)
            if not isinstance(val, Float):
                raise TypeError(f'expected a real number slice, got {val}')
            if not val.is_integer():
                raise TypeError(f'expected an integer slice, got {val}')
            slices.append(int(val))

        val = self._visit_expr(e.value, ctx)
        value[slices] = val
        return value

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
            if not isinstance(array, NDArray):
                raise TypeError(f'expected a tensor, got {array}')
            for val in array:
                match target:
                    case NamedId():
                        ctx.env[target] = val
                    case TupleBinding():
                        self._unpack_tuple(target, val, ctx)
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
        return NDArray(elts)

    def _visit_if_expr(self, e: IfExpr, ctx: _EvalCtx):
        cond = self._visit_expr(e.cond, ctx)
        if not isinstance(cond, bool):
            raise TypeError(f'expected a boolean, got {cond}')
        return self._visit_expr(e.ift if cond else e.iff, ctx)

    def _unpack_tuple(self, binding: TupleBinding, val: NDArray, ctx: _EvalCtx) -> None:
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

    def _visit_assign(self, stmt: Assign, ctx: _EvalCtx) -> None:
        val = self._visit_expr(stmt.expr, ctx)
        match stmt.binding:
            case NamedId():
                ctx.env[stmt.binding] = val
            case TupleBinding():
                self._unpack_tuple(stmt.binding, val, ctx)

    def _visit_indexed_assign(self, stmt: IndexedAssign, ctx: _EvalCtx) -> None:
        # lookup the array
        array = self._lookup(stmt.var, ctx)

        # evaluate indices
        slices: list[int] = []
        for s in stmt.slices:
            val = self._visit_expr(s, ctx)
            if not isinstance(val, Float):
                raise TypeError(f'expected a real number slice, got {val}')
            if not val.is_integer():
                raise TypeError(f'expected an integer slice, got {val}')
            slices.append(int(val))

        # evaluate and update array
        val = self._visit_expr(stmt.expr, ctx)
        array[slices] = val

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

    def _visit_if(self, stmt: IfStmt, ctx: _EvalCtx) -> None:
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

    def _visit_while(self, stmt: WhileStmt, ctx: _EvalCtx) -> None:
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

    def _visit_for(self, stmt: ForStmt, ctx: _EvalCtx) -> None:
        # evaluate the iterable data
        iterable = self._visit_expr(stmt.iterable, ctx)
        if not isinstance(iterable, NDArray):
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
                    if isinstance(v, Float) and v.is_integer():
                        # HACK: keeps things as specific as possible
                        args.append(int(v))
                    else:
                        args.append(v)

        kwargs: dict[str, Any] = {}
        for k, v in e.kwargs:
            match v:
                case ForeignAttribute():
                    kwargs[k] = self._visit_foreign_attr(v, ctx)
                case _:
                    v = self._visit_expr(v, ctx)
                    if isinstance(v, Float) and v.is_integer():
                        kwargs[k] = int(v)
                    else:
                        kwargs[k] = v

        return ctor(*args, **kwargs)

    def _visit_context(self, stmt: ContextStmt, ctx: _EvalCtx):
        round_ctx = self._visit_expr(stmt.ctx, ctx)
        block_ctx = self._visit_block(stmt.body, self._eval_ctx(ctx.env, round_ctx))
        ctx.env = block_ctx.env

    def _visit_assert(self, stmt: AssertStmt, ctx: _EvalCtx):
        test = self._visit_expr(stmt.test, ctx)
        if not isinstance(test, bool):
            raise TypeError(f'expected a boolean, got {test}')
        if not test:
            raise AssertionError(stmt.msg)

    def _visit_effect(self, stmt: EffectStmt, ctx: _EvalCtx):
        self._visit_expr(stmt.expr, ctx)

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

    # override typing hint
    def _visit_statement(self, stmt, ctx: _EvalCtx) -> None:
        return super()._visit_statement(stmt, ctx)


class DefaultInterpreter(Interpreter):
    """
    Standard interpreter for FPy programs.

    Values:
     - booleans are Python `bool` values,
     - real numbers are FPy `float` values,
     - tensors are Titanic `NDArray` values.

    All operations are correctly-rounded.
    """

    ctx: Optional[Context] = None
    """optionaly overriding context"""

    def __init__(self, ctx: Optional[Context] = None):
        self.ctx = ctx

    def eval(
        self,
        func: Function,
        args: Sequence[Any],
        ctx: Optional[Context] = None
    ):
        if not isinstance(func, Function):
            raise TypeError(f'Expected Function, got {func}')

        rt = _Interpreter(func.env, override_ctx=self.ctx)
        ctx = self._func_ctx(func, ctx)
        return rt.eval(func.ast, args, ctx)
