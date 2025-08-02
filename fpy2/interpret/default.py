"""
FPy runtime backed by the Titanic library.
"""

import copy
import functools

from typing import Any, Callable, Optional, Sequence, TypeAlias

from .. import ops

from ..ast import *
from ..fpc_context import FPCoreContext
from ..number import Context, Float, FP64, INTEGER
from ..env import ForeignEnv
from ..function import Function
from ..primitive import Primitive

from .interpreter import Interpreter, FunctionReturnException

ScalarVal: TypeAlias = bool | Float
"""Type of scalar values in FPy programs."""
TensorVal: TypeAlias = list
"""Type of list values in FPy programs."""

ScalarArg: TypeAlias = ScalarVal | str | int | float
"""Type of scalar arguments in FPy programs; includes native Python types"""
TensorArg: TypeAlias = tuple | list
"""Type of list arguments in FPy programs; includes native Python types"""

_Env: TypeAlias = dict[NamedId, ScalarVal | TensorVal]
"""Type of the environment used by the interpreter."""

# Pre-built lookup tables for better performance
_NULLARY_TABLE: dict[type[NullaryOp], Callable[..., Any]] = {
    ConstNan: ops.nan,
    ConstInf: ops.inf,
    ConstPi: ops.const_pi,
    ConstE: ops.const_e,
    ConstLog2E: ops.const_log2e,
    ConstLog10E: ops.const_log10e,
    ConstLn2: ops.const_ln2,
    ConstPi_2: ops.const_pi_2,
    ConstPi_4: ops.const_pi_4,
    Const1_Pi: ops.const_1_pi,
    Const2_Pi: ops.const_2_pi,
    Const2_SqrtPi: ops.const_2_sqrt_pi,
    ConstSqrt2: ops.const_sqrt2,
    ConstSqrt1_2: ops.const_sqrt1_2,
}

_UNARY_TABLE: dict[type[UnaryOp], Callable[..., Any]] = {
    Fabs: ops.fabs,
    Sqrt: ops.sqrt,
    Neg: ops.neg,
    Cbrt: ops.cbrt,
    Ceil: ops.ceil,
    Floor: ops.floor,
    NearbyInt: ops.nearbyint,
    RoundInt: ops.roundint,
    Trunc: ops.trunc,
    Acos: ops.acos,
    Asin: ops.asin,
    Atan: ops.atan,
    Cos: ops.cos,
    Sin: ops.sin,
    Tan: ops.tan,
    Acosh: ops.acosh,
    Asinh: ops.asinh,
    Atanh: ops.atanh,
    Cosh: ops.cosh,
    Sinh: ops.sinh,
    Tanh: ops.tanh,
    Exp: ops.exp,
    Exp2: ops.exp2,
    Expm1: ops.expm1,
    Log: ops.log,
    Log10: ops.log10,
    Log1p: ops.log1p,
    Log2: ops.log2,
    Erf: ops.erf,
    Erfc: ops.erfc,
    Lgamma: ops.lgamma,
    Tgamma: ops.tgamma,
    IsFinite: ops.isfinite,
    IsInf: ops.isinf,
    IsNan: ops.isnan,
    IsNormal: ops.isnormal,
    Signbit: ops.signbit,
    Round: ops.round,
    RoundExact: ops.round_exact
}

_BINARY_TABLE: dict[type[BinaryOp], Callable[..., Any]] = {
    Add: ops.add,
    Sub: ops.sub,
    Mul: ops.mul,
    Div: ops.div,
    Copysign: ops.copysign,
    Fdim: ops.fdim,
    Fmod: ops.fmod,
    Remainder: ops.remainder,
    Hypot: ops.hypot,
    Atan2: ops.atan2,
    Pow: ops.pow,
    RoundAt: ops.round_at
}

_TERNARY_TABLE: dict[type[TernaryOp], Callable[..., Any]] = {
    Fma: ops.fma,
}


class _Interpreter(Visitor):
    """Single-use interpreter for a function"""

    __slots__ = ('env', 'foreign', 'override_ctx')

    env: _Env
    """mapping from variable names to values"""
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
        self.env = {}
        self.foreign = foreign
        self.override_ctx = override_ctx

    def _eval_ctx(self, ctx: Context | FPCoreContext):
        if self.override_ctx is not None:
            return self.override_ctx
        match ctx:
            case Context():
                return ctx
            case FPCoreContext():
                return ctx.to_context()
            case _:
                raise TypeError(f'Expected `Context` or `FPCoreContext`, got {ctx}')

    def _arg_to_value(self, arg: Any):
        match arg:
            case int():
                return Float.from_int(arg, ctx=INTEGER, checked=False)
            case float():
                return Float.from_float(arg, ctx=FP64, checked=False)
            case Float():
                return arg
            case tuple():
                return tuple(self._arg_to_value(x) for x in arg)
            case list():
                return [self._arg_to_value(x) for x in arg]
            case _:
                return arg

    def eval(self, func: FuncDef, args: Sequence[Any], ctx: Context):
        # check arity
        if len(args) != len(func.args):
            raise TypeError(f'Expected {len(func.args)} arguments, got {len(args)}')

        # possibly override the context
        eval_ctx = self._eval_ctx(ctx)

        # process arguments and add to environment
        for val, arg in zip(args, func.args):
            match arg.type:
                case AnyTypeAnn() | None:
                    x = self._arg_to_value(val)
                    if isinstance(arg.name, NamedId):
                        self.env[arg.name] = x
                case RealTypeAnn():
                    x = self._arg_to_value(val)
                    if not isinstance(x, Float):
                        raise NotImplementedError(f'argument is a scalar, got data {val}')
                    if isinstance(arg.name, NamedId):
                        self.env[arg.name] = x
                case TensorTypeAnn():
                    # TODO: check shape
                    x = self._arg_to_value(val)
                    if not isinstance(x, list):
                        raise NotImplementedError(f'argument is a list, got data {val}')
                    if isinstance(arg.name, NamedId):
                        self.env[arg.name] = x
                case _:
                    raise NotImplementedError(f'unknown argument type {arg.type}')

        # process free variables
        for var in func.free_vars:
            x = self._arg_to_value(self.foreign[var.base])
            self.env[var] = x

        # evaluation
        try:
            self._visit_block(func.body, eval_ctx)
            raise RuntimeError('no return statement encountered')
        except FunctionReturnException as e:
            return e.value

    def _lookup(self, name: NamedId):
        try:
            return self.env[name]
        except KeyError as exc:
            raise RuntimeError(f'unbound variable {name}') from exc

    def _visit_var(self, e: Var, ctx: Context):
        return self._lookup(e.name)

    def _visit_bool(self, e: BoolVal, ctx: Any):
        return e.val

    def _visit_foreign(self, e: ForeignVal, ctx: None):
        return e.val

    def _visit_decnum(self, e: Decnum, ctx: Context):
        return ctx.round(e.as_rational())

    def _visit_integer(self, e: Integer, ctx: Context):
        return ctx.round(e.val)

    def _visit_hexnum(self, e: Hexnum, ctx: Context):
        return ctx.round(e.as_rational())

    def _visit_rational(self, e: Rational, ctx: Context):
        return ctx.round(e.as_rational())

    def _visit_digits(self, e: Digits, ctx: Context):
        return ctx.round(e.as_rational())

    def _apply_method(self, fn: Callable[..., Any], args: Sequence[Expr], ctx: Context):
        # fn: Callable[[Float, ..., Context], Float]
        vals = tuple(self._visit_expr(arg, ctx) for arg in args)
        for val in vals:
            if not isinstance(val, Float):
                raise TypeError(f'expected a real number argument, got {val}')
        # compute the result
        return fn(*vals, ctx=ctx)

    def _apply_not(self, arg: Expr, ctx: Context):
        arg = self._visit_expr(arg, ctx)
        if not isinstance(arg, bool):
            raise TypeError(f'expected a boolean argument, got {arg}')
        return not arg

    def _apply_and(self, args: Sequence[Expr], ctx: Context):
        for arg in args:
            val = self._visit_expr(arg, ctx)
            if not isinstance(val, bool):
                raise TypeError(f'expected a boolean argument, got {val}')
            if not val:  # Short-circuit evaluation
                return False
        return True

    def _apply_or(self, args: Sequence[Expr], ctx: Context):
        for arg in args:
            val = self._visit_expr(arg, ctx)
            if not isinstance(val, bool):
                raise TypeError(f'expected a boolean argument, got {val}')
            if val:  # Short-circuit evaluation
                return True
        return False

    def _apply_range(self, arg: Expr, ctx: Context):
        stop = self._visit_expr(arg, ctx)
        if not isinstance(stop, Float):
            raise TypeError(f'expected a real number argument, got {stop}')
        if not stop.is_integer():
            raise TypeError(f'expected an integer argument, got {stop}')
        n = int(stop)
        return [Float.from_int(i, ctx=ctx) for i in range(n)]

    def _apply_dim(self, arg: Expr, ctx: Context):
        v = self._visit_expr(arg, ctx)
        if not isinstance(v, list):
            raise TypeError(f'expected a list, got {v}')
        return ops.dim(v, ctx)

    def _apply_enumerate(self, arg: Expr, ctx: Context):
        v = self._visit_expr(arg, ctx)
        if not isinstance(v, list):
            raise TypeError(f'expected a list, got {v}')
        return [
            [Float.from_int(i, ctx=ctx), val]
            for i, val in enumerate(v)
        ]

    def _apply_size(self, arr: Expr, idx: Expr, ctx: Context):
        v = self._visit_expr(arr, ctx)
        if not isinstance(v, list):
            raise TypeError(f'expected a list, got {v}')
        dim = self._visit_expr(idx, ctx)
        if not isinstance(dim, Float):
            raise TypeError(f'expected a real number argument, got {dim}')
        if not dim.is_integer():
            raise TypeError(f'expected an integer argument, got {dim}')
        return ops.size(v, dim, ctx)

    def _apply_zip(self, args: Sequence[Expr], ctx: Context):
        """Apply the `zip` method to the given n-ary expression."""
        if len(args) == 0:
            return []

        # evaluate all children
        arrays = tuple(self._visit_expr(arg, ctx) for arg in args)
        for val in arrays:
            if not isinstance(val, list):
                raise TypeError(f'expected a list argument, got {val}')
        return list(zip(*arrays))

    def _apply_min(self, args: Sequence[Expr], ctx: Context):
        """Apply the `min` method to the given n-ary expression."""
        vals: list[Float] = []
        for arg in args:
            val = self._visit_expr(arg, ctx)
            if not isinstance(val, Float):
                raise TypeError(f'expected a real number argument, got {val}')
            if not val.isnan:
                vals.append(val)

        len_vals = len(vals)
        if len_vals == 0:
            return Float(isnan=True, ctx=ctx)
        elif len_vals == 1:
            return vals[0]
        else:
            return min(*vals)

    def _apply_max(self, args: Sequence[Expr], ctx: Context):
        """Apply the `max` method to the given n-ary expression."""
        # evaluate all children
        vals: list[Float] = []
        for arg in args:
            val = self._visit_expr(arg, ctx)
            if not isinstance(val, Float):
                raise TypeError(f'expected a real number argument, got {val}')
            if not val.isnan:
                vals.append(val)

        len_vals = len(vals)
        if len_vals == 0:
            return Float(isnan=True, ctx=ctx)
        elif len_vals == 1:
            return vals[0]
        else:
            return max(*vals)

    def _apply_sum(self, arg: Expr, ctx: Context):
        """Apply the `sum` method to the given n-ary expression."""
        val = self._visit_expr(arg, ctx)
        if not isinstance(val, list):
            raise TypeError(f'expected a list, got {val}')
        if not len(val) > 0:
            raise ValueError('cannot sum an empty list')

        for x in val:
            if not isinstance(x, Float):
                raise TypeError(f'expected a real number argument, got {x}')

        accum = val[0]
        for x in val[1:]:
            accum = ops.add(accum, x, ctx=ctx)
        return accum

    def _visit_nullaryop(self, e: NullaryOp, ctx: Context):
        fn = _NULLARY_TABLE.get(type(e))
        if fn is not None:
            return fn(ctx=ctx)
        else:
            raise RuntimeError('unknown operator', e)

    def _visit_unaryop(self, e: UnaryOp, ctx: Context):
        fn = _UNARY_TABLE.get(type(e))
        if fn is not None:
            arg = self._visit_expr(e.arg, ctx)
            if not isinstance(arg, Float):
                raise TypeError(f'expected a real number argument, got {arg}')
            return fn(arg, ctx=ctx)
        else:
            match e:
                case Not():
                    return self._apply_not(e.arg, ctx)
                case Range():
                    return self._apply_range(e.arg, ctx)
                case Dim():
                    return self._apply_dim(e.arg, ctx)
                case Enumerate():
                    return self._apply_enumerate(e.arg, ctx)
                case Sum():
                    return self._apply_sum(e.arg, ctx)
                case _:
                    raise RuntimeError('unknown operator', e)

    def _visit_binaryop(self, e: BinaryOp, ctx: Context):
        fn = _BINARY_TABLE.get(type(e))
        if fn is not None:
            first = self._visit_expr(e.first, ctx)
            second = self._visit_expr(e.second, ctx)
            if not isinstance(first, Float):
                raise TypeError(f'expected a real number argument, got {first}')
            if not isinstance(second, Float):
                raise TypeError(f'expected a real number argument, got {second}')
            return fn(first, second, ctx=ctx)
        else:
            match e:
                case Size():
                    return self._apply_size(e.first, e.second, ctx)
                case _:
                    raise RuntimeError('unknown operator', e)

    def _visit_ternaryop(self, e: TernaryOp, ctx: Context):
        fn = _TERNARY_TABLE.get(type(e))
        if fn is not None:
            first = self._visit_expr(e.first, ctx)
            second = self._visit_expr(e.second, ctx)
            third = self._visit_expr(e.third, ctx)
            if not isinstance(first, Float):
                raise TypeError(f'expected a real number argument, got {first}')
            if not isinstance(second, Float):
                raise TypeError(f'expected a real number argument, got {second}')
            if not isinstance(third, Float):
                raise TypeError(f'expected a real number argument, got {third}')
            return fn(first, second, third, ctx=ctx)
        else:
            raise RuntimeError('unknown operator', e)

    def _visit_naryop(self, e: NaryOp, ctx: Context):
        match e:
            case And():
                return self._apply_and(e.args, ctx)
            case Or():
                return self._apply_or(e.args, ctx)
            case Zip():
                return self._apply_zip(e.args, ctx)
            case Min():
                return self._apply_min(e.args, ctx)
            case Max():
                return self._apply_max(e.args, ctx)
            case _:
                raise RuntimeError('unknown operator', e)

    def _visit_call(self, e: Call, ctx: Context):
        match e.func:
            case NamedId():
                fn = self.foreign[e.func.base]
            case ForeignAttribute():
                fn = self._visit_foreign_attr(e.func, ctx)
            case _:
                raise RuntimeError('unreachable', e.func)

        args = [self._visit_expr(arg, ctx) for arg in e.args]
        match fn:
            case Function():
                # calling FPy function
                rt = _Interpreter(fn.env, override_ctx=self.override_ctx)
                return rt.eval(fn.ast, args, ctx)
            case Primitive():
                # calling FPy primitive
                return fn(*args, ctx=ctx)
            case _:
                # calling foreign function
                # only `print` is allowed
                if fn == print:
                    print(*args)
                    # TODO: should we allow `None` to return
                    return None
                else:
                    raise RuntimeError(f'attempting to call a Python function: `{fn}` at `{e.format()}`')

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
        lhs = self._visit_expr(e.args[0], ctx)
        for op, arg in zip(e.ops, e.args[1:]):
            rhs = self._visit_expr(arg, ctx)
            if not self._apply_cmp2(op, lhs, rhs):
                return False
            lhs = rhs
        return True

    def _visit_tuple_expr(self, e: TupleExpr, ctx: Context):
        return tuple(self._visit_expr(x, ctx) for x in e.args)

    def _visit_list_expr(self, e: ListExpr, ctx: Context):
        return [self._visit_expr(x, ctx) for x in e.args]

    def _visit_list_ref(self, e: ListRef, ctx: Context):
        arr = self._visit_expr(e.value, ctx)
        if not isinstance(arr, list):
            raise TypeError(f'expected a list, got {arr}')

        idx = self._visit_expr(e.index, ctx)
        if not isinstance(idx, Float):
            raise TypeError(f'expected a real number index, got {idx}')
        if not idx.is_integer():
            raise TypeError(f'expected an integer index, got {idx}')
        return arr[int(idx)]

    def _visit_list_slice(self, e: ListSlice, ctx: Context):
        arr = self._visit_expr(e.value, ctx)
        if not isinstance(arr, list):
            raise TypeError(f'expected a list, got {arr}')

        if e.start is None:
            start = 0
        else:
            val = self._visit_expr(e.start, ctx)
            if not isinstance(val, Float):
                raise TypeError(f'expected a real number start index, got {val}')
            if not val.is_integer():
                raise TypeError(f'expected an integer start index, got {val}')
            start = int(val)

        if e.stop is None:
            stop = len(arr)
        else:
            val = self._visit_expr(e.stop, ctx)
            if not isinstance(val, Float):
                raise TypeError(f'expected a real number stop index, got {val}')
            if not val.is_integer():
                raise TypeError(f'expected an integer stop index, got {val}')
            stop = int(val)

        if start < 0 or stop > len(arr):
            return []
        else:
            return [arr[i] for i in range(start, stop)]

    def _visit_list_set(self, e: ListSet, ctx: Context):
        value = self._visit_expr(e.array, ctx)
        if not isinstance(value, list):
            raise TypeError(f'expected a list, got {value}')
        array = copy.deepcopy(value) # make a copy

        slices = []
        for s in e.slices:
            val = self._visit_expr(s, ctx)
            if not isinstance(val, Float):
                raise TypeError(f'expected a real number slice, got {val}')
            if not val.is_integer():
                raise TypeError(f'expected an integer slice, got {val}')
            slices.append(int(val))

        val = self._visit_expr(e.value, ctx)
        for idx in slices[:-1]:
            if not isinstance(array, list):
                raise TypeError(f'index {idx} is out of bounds for `{array}`')
            array = array[idx]

        array[slices[-1]] = val
        return array


    def _apply_comp(
        self,
        bindings: list[tuple[Id | TupleBinding, Expr]],
        elt: Expr,
        ctx: Context,
        elts: list[Any]
    ):
        if bindings == []:
            elts.append(self._visit_expr(elt, ctx))
        else:
            target, iterable = bindings[0]
            array = self._visit_expr(iterable, ctx)
            if not isinstance(array, list):
                raise TypeError(f'expected a list, got {array}')
            for val in array:
                match target:
                    case NamedId():
                        self.env[target] = val
                    case TupleBinding():
                        self._unpack_tuple(target, val, ctx)
                self._apply_comp(bindings[1:], elt, ctx, elts)

    def _visit_list_comp(self, e: ListComp, ctx: Context):
        # evaluate comprehension
        elts: list[Any] = []
        bindings = list(zip(e.targets, e.iterables))
        self._apply_comp(bindings, e.elt, ctx, elts)
        return elts

    def _visit_if_expr(self, e: IfExpr, ctx: Context):
        cond = self._visit_expr(e.cond, ctx)
        if not isinstance(cond, bool):
            raise TypeError(f'expected a boolean, got {cond}')
        return self._visit_expr(e.ift if cond else e.iff, ctx)

    def _unpack_tuple(self, binding: TupleBinding, val: list, ctx: Context) -> None:
        if not isinstance(val, tuple):
            raise TypeError(f'can only unpack tuples, got `{val}` for `{binding}`')
        if len(binding.elts) != len(val):
            raise NotImplementedError(f'unpacking {len(val)} values into {len(binding.elts)}')
        for elt, v in zip(binding.elts, val):
            match elt:
                case NamedId():
                    self.env[elt] = v
                case TupleBinding():
                    self._unpack_tuple(elt, v, ctx)

    def _visit_assign(self, stmt: Assign, ctx: Context) -> None:
        val = self._visit_expr(stmt.expr, ctx)
        match stmt.binding:
            case NamedId():
                self.env[stmt.binding] = val
            case TupleBinding():
                self._unpack_tuple(stmt.binding, val, ctx)

    def _visit_indexed_assign(self, stmt: IndexedAssign, ctx: Context) -> None:
        # lookup the array
        array = self._lookup(stmt.var)

        # evaluate indices
        slices: list[int] = []
        for slice in stmt.slices:
            val = self._visit_expr(slice, ctx)
            if not isinstance(val, Float):
                raise TypeError(f'expected a real number slice, got {val}')
            if not val.is_integer():
                raise TypeError(f'expected an integer slice, got {val}')
            slices.append(int(val))

        # evaluate and update array
        val = self._visit_expr(stmt.expr, ctx)
        for idx in slices[:-1]:
            if not isinstance(array, list):
                raise TypeError(f'index {idx} is out of bounds for `{array}`')
            array = array[idx]
        array[slices[-1]] = val

    def _visit_if1(self, stmt: If1Stmt, ctx: Context):
        # evaluate the condition
        cond = self._visit_expr(stmt.cond, ctx)
        if not isinstance(cond, bool):
            raise TypeError(f'expected a boolean, got {cond}')
        elif cond:
            # evaluate the if-true branch
            self._visit_block(stmt.body, ctx)

    def _visit_if(self, stmt: IfStmt, ctx: Context) -> None:
        cond = self._visit_expr(stmt.cond, ctx)
        if not isinstance(cond, bool):
            raise TypeError(f'expected a boolean, got {cond}')

        if cond:
            # evaluate the if-true branch
            self._visit_block(stmt.ift, ctx)
        else:
            # evaluate the if-false branch
            self._visit_block(stmt.iff, ctx)

    def _visit_while(self, stmt: WhileStmt, ctx: Context) -> None:
        # evaluate the condition
        cond = self._visit_expr(stmt.cond, ctx)
        if not isinstance(cond, bool):
            raise TypeError(f'expected a boolean, got {cond}')

        while cond:
            # evaluate the while body
            self._visit_block(stmt.body, ctx)
            # evaluate the condition
            cond = self._visit_expr(stmt.cond, ctx)
            if not isinstance(cond, bool):
                raise TypeError(f'expected a boolean, got {cond}')

    def _visit_for(self, stmt: ForStmt, ctx: Context) -> None:
        # evaluate the iterable data
        iterable = self._visit_expr(stmt.iterable, ctx)
        if not isinstance(iterable, list):
            raise TypeError(f'expected a list, got {iterable}')
        # iterate over each element
        for val in iterable:
            match stmt.target:
                case NamedId():
                    self.env[stmt.target] = val
                case TupleBinding():
                    self._unpack_tuple(stmt.target, val, ctx)
            # evaluate the body of the loop
            self._visit_block(stmt.body, ctx)

    def _visit_foreign_attr(self, e: ForeignAttribute, ctx: Context):
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

    def _visit_context(self, stmt: ContextStmt, ctx: Context):
        match stmt.ctx:
            case ForeignAttribute():
                round_ctx = self._visit_foreign_attr(stmt.ctx, ctx)
            case _:
                round_ctx = self._visit_expr(stmt.ctx, ctx)
        self._visit_block(stmt.body, round_ctx)

    def _visit_assert(self, stmt: AssertStmt, ctx: Context):
        test = self._visit_expr(stmt.test, ctx)
        if not isinstance(test, bool):
            raise TypeError(f'expected a boolean, got {test}')
        if not test:
            raise AssertionError(stmt.msg)

    def _visit_effect(self, stmt: EffectStmt, ctx: Context):
        self._visit_expr(stmt.expr, ctx)

    def _visit_return(self, stmt: ReturnStmt, ctx: Context):
        x = self._visit_expr(stmt.expr, ctx)
        raise FunctionReturnException(x)

    def _visit_block(self, block: StmtBlock, ctx: Context):
        for stmt in block.stmts:
            self._visit_statement(stmt, ctx)

    def _visit_function(self, func: FuncDef, ctx: Context):
        raise NotImplementedError('do not call directly')


class DefaultInterpreter(Interpreter):
    """
    Standard interpreter for FPy programs.

    Values:
     - booleans are Python `bool` values,
     - real numbers are FPy `Float` values,
     - lists are Python `list` values.

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
