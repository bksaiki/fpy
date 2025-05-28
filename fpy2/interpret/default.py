"""
FPy runtime backed by the Titanic library.
"""

from fractions import Fraction
from typing import Any, Callable, Optional, Sequence, TypeAlias

from titanfp.titanic.ndarray import NDArray

from .. import math

from ..ast import *
from ..fpc_context import FPCoreContext
from ..number import Context, Float, IEEEContext, RM
from ..number.gmp import mpfr_constant
from ..env import ForeignEnv
from ..function import Function
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

def _isfinite(x: Float, _: Context) -> bool:
    return x.is_finite()

def _isinf(x: Float, _: Context) -> bool:
    return x.isinf

def _isnan(x: Float, _: Context) -> bool:
    return x.isnan

def _isnormal(x: Float, _: Context) -> bool:
    # TODO: should all Floats have this property?
    return True

def _signbit(x: Float, _: Context) -> bool:
    # TODO: should all Floats have this property?
    return x.s

_unary_table: dict[UnaryOpKind, Callable[[Float, Context], Any]] = {
    UnaryOpKind.FABS: math.fabs,
    UnaryOpKind.SQRT: math.sqrt,
    UnaryOpKind.NEG: math.neg,
    UnaryOpKind.CBRT: math.cbrt,
    UnaryOpKind.CEIL: math.ceil,
    UnaryOpKind.FLOOR: math.floor,
    UnaryOpKind.NEARBYINT: math.nearbyint,
    UnaryOpKind.ROUND: math.round,
    UnaryOpKind.TRUNC: math.trunc,
    UnaryOpKind.ACOS: math.acos,
    UnaryOpKind.ASIN: math.asin,
    UnaryOpKind.ATAN: math.atan,
    UnaryOpKind.COS: math.cos,
    UnaryOpKind.SIN: math.sin,
    UnaryOpKind.TAN: math.tan,
    UnaryOpKind.ACOSH: math.acosh,
    UnaryOpKind.ASINH: math.asinh,
    UnaryOpKind.ATANH: math.atanh,
    UnaryOpKind.COSH: math.cosh,
    UnaryOpKind.SINH: math.sinh,
    UnaryOpKind.TANH: math.tanh,
    UnaryOpKind.EXP: math.exp,
    UnaryOpKind.EXP2: math.exp2,
    UnaryOpKind.EXPM1: math.expm1,
    UnaryOpKind.LOG: math.log,
    UnaryOpKind.LOG10: math.log10,
    UnaryOpKind.LOG1P: math.log1p,
    UnaryOpKind.LOG2: math.log2,
    UnaryOpKind.ERF: math.erf,
    UnaryOpKind.ERFC: math.erfc,
    UnaryOpKind.LGAMMA: math.lgamma,
    UnaryOpKind.TGAMMA: math.tgamma,
    UnaryOpKind.ISFINITE: _isfinite,
    UnaryOpKind.ISINF: _isinf,
    UnaryOpKind.ISNAN: _isnan,
    UnaryOpKind.ISNORMAL: _isnormal,
    UnaryOpKind.SIGNBIT: _signbit,
}

_binary_table: dict[BinaryOpKind, Callable[[Float, Float, Context], Any]] = {
    BinaryOpKind.ADD: math.add,
    BinaryOpKind.SUB: math.sub,
    BinaryOpKind.MUL: math.mul,
    BinaryOpKind.DIV: math.div,
    BinaryOpKind.COPYSIGN: math.copysign,
    BinaryOpKind.FDIM: math.fdim,
    BinaryOpKind.FMAX: math.fmax,
    BinaryOpKind.FMIN: math.fmin,
    BinaryOpKind.FMOD: math.fmod,
    BinaryOpKind.REMAINDER: math.remainder,
    BinaryOpKind.HYPOT: math.hypot,
    BinaryOpKind.ATAN2: math.atan2,
    BinaryOpKind.POW: math.pow,
}

_ternary_table: dict[TernaryOpKind, Callable[[Float, Float, Float, Context], Any]] = {
    TernaryOpKind.FMA: math.fma,
}

_Env: TypeAlias = dict[NamedId, ScalarVal | TensorVal]

_PY_CTX = IEEEContext(11, 64, RM.RNE)
"""the native Python floating-point context"""


class _Interpreter(DefaultAstVisitor):
    """Single-use interpreter for a function"""

    foreign: ForeignEnv
    """foreign environment"""
    override_ctx: Optional[Context]
    """optional overriding context"""
    env: _Env
    """Environment mapping variable names to values"""

    def __init__(
        self, 
        foreign: ForeignEnv,
        *,
        override_ctx: Optional[Context] = None,
        env: Optional[_Env] = None,
    ):
        if env is None:
            env = {}

        self.foreign = foreign
        self.override_ctx = override_ctx
        self.env = env

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

    # TODO: what are the semantics of arguments
    def _arg_to_mpmf(self, arg: Any, ctx: Context):
        match arg:
            case int():
                return Float.from_int(arg, ctx=ctx)
            case float():
                return Float.from_float(arg, ctx=ctx)
            case Float():
                return arg.round(ctx)
            case tuple() | list():
                return NDArray([self._arg_to_mpmf(x, ctx) for x in arg])
            case _:
                return arg

    def eval(
        self,
        func: FuncDef,
        args: Sequence[Any],
        ctx: Optional[Context] = None
    ):
        # check arity
        args = tuple(args)
        if len(args) != len(func.args):
            raise TypeError(f'Expected {len(func.args)} arguments, got {len(args)}')

        # determine context if `None` is specified
        if ctx is None:
            ctx = _PY_CTX

        # possibly override the context
        ctx = self._eval_ctx(ctx)
        assert isinstance(ctx, Context)

        # process arguments and add to environment
        for val, arg in zip(args, func.args):
            match arg.type:
                case AnyTypeAnn():
                    x = self._arg_to_mpmf(val, ctx)
                    if isinstance(arg.name, NamedId):
                        self.env[arg.name] = x
                case RealTypeAnn():
                    x = self._arg_to_mpmf(val, ctx)
                    if not isinstance(x, Float):
                        raise NotImplementedError(f'argument is a scalar, got data {val}')
                    if isinstance(arg.name, NamedId):
                        self.env[arg.name] = x
                case _:
                    raise NotImplementedError(f'unknown argument type {arg.type}')

        # process free variables
        for var in func.free_vars:
            x = self._arg_to_mpmf(self.foreign[var.base], ctx)
            self.env[var] = x

        # evaluation
        try:
            self._visit_block(func.body, ctx)
            raise RuntimeError('no return statement encountered')
        except FunctionReturnException as e:
            return e.value

    def _lookup(self, name: NamedId):
        if name not in self.env:
            raise RuntimeError(f'unbound variable {name}')
        return self.env[name]

    def _visit_var(self, e: Var, ctx: Context):
        return self._lookup(e.name)

    def _visit_bool(self, e: BoolVal, ctx: Any):
        return e.val

    def _visit_foreign(self, e: ForeignVal, ctx: None):
        return e.val

    def _visit_decnum(self, e: Decnum, ctx: Context):
        x = decnum_to_fraction(e.val)
        return ctx.round(x)

    def _visit_integer(self, e: Integer, ctx: Context):
        return ctx.round(e.val)

    def _visit_hexnum(self, e: Hexnum, ctx: Context):
        x = hexnum_to_fraction(e.val)
        return ctx.round(x)

    def _visit_rational(self, e: Rational, ctx: Context):
        x = Fraction(e.p, e.q)
        return ctx.round(x)

    def _visit_constant(self, e: Constant, ctx: Context):
        prec, _ = ctx.round_params()
        assert isinstance(prec, int) # TODO: not every context produces has a known precision
        x = mpfr_constant(e.val, prec=prec)
        return ctx.round(x)

    def _visit_digits(self, e: Digits, ctx: Context):
        x = digits_to_fraction(e.m, e.e, e.b)
        return ctx.round(x)

    def _apply_method(self, fn: Callable[..., Any], args: Sequence[Expr], ctx: Context):
        vals: list[Float] = []
        for arg in args:
            val = self._visit_expr(arg, ctx)
            if not isinstance(val, Float):
                raise TypeError(f'expected a real number argument, got {val}')
            vals.append(val)
        # compute the result
        return fn(*vals, ctx=ctx)

    def _apply_cast(self, arg: Expr, ctx: Context):
        x = self._visit_expr(arg, ctx)
        if not isinstance(x, Float):
            raise TypeError(f'expected a real number argument, got {x}')
        return ctx.round(x)

    def _apply_not(self, arg: Expr, ctx: Context):
        arg = self._visit_expr(arg, ctx)
        if not isinstance(arg, bool):
            raise TypeError(f'expected a boolean argument, got {arg}')
        return not arg

    def _apply_and(self, args: Sequence[Expr], ctx: Context):
        vals: list[bool] = []
        for arg in args:
            val = self._visit_expr(arg, ctx)
            if not isinstance(val, bool):
                raise TypeError(f'expected a boolean argument, got {val}')
            vals.append(val)
        return all(vals)

    def _apply_or(self, args: Sequence[Expr], ctx: Context):
        vals: list[bool] = []
        for arg in args:
            val = self._visit_expr(arg, ctx)
            if not isinstance(val, bool):
                raise TypeError(f'expected a boolean argument, got {val}')
            vals.append(val)
        return any(vals)

    def _apply_shape(self, arg: Expr, ctx: Context):
        v = self._visit_expr(arg, ctx)
        if not isinstance(v, NDArray):
            raise TypeError(f'expected a tensor, got {v}')
        return NDArray([ctx.round(x) for x in v.shape])

    def _apply_range(self, arg: Expr, ctx: Context):
        stop = self._visit_expr(arg, ctx)
        if not isinstance(stop, Float):
            raise TypeError(f'expected a real number argument, got {stop}')
        if not stop.is_integer():
            raise TypeError(f'expected an integer argument, got {stop}')

        elts: list[Float] = []
        for i in range(int(stop)):
            elts.append(Float.from_int(i, ctx=ctx))
        return NDArray(elts)

    def _apply_dim(self, arg: Expr, ctx: Context):
        v = self._visit_expr(arg, ctx)
        if not isinstance(v, NDArray):
            raise TypeError(f'expected a tensor, got {v}')
        return Float.from_int(len(v.shape), ctx=ctx)

    def _apply_size(self, arr: Expr, idx: Expr, ctx: Context):
        v = self._visit_expr(arr, ctx)
        if not isinstance(v, NDArray):
            raise TypeError(f'expected a tensor, got {v}')
        dim = self._visit_expr(idx, ctx)
        if not isinstance(dim, Float):
            raise TypeError(f'expected a real number argument, got {dim}')
        if not dim.is_integer():
            raise TypeError(f'expected an integer argument, got {dim}')
        return Float.from_int(v.shape[int(dim)], ctx=ctx)

    def _apply_zip(self, args: Sequence[Expr], ctx: Context):
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

    def _visit_unaryop(self, e: UnaryOp, ctx: Context):
        fn = _unary_table.get(e.op)
        if fn is not None:
            return self._apply_method(fn, (e.arg,), ctx)
        else:
            match e.op:
                case UnaryOpKind.CAST:
                    return self._apply_cast(e.arg, ctx)
                case UnaryOpKind.NOT:
                    return self._apply_not(e.arg, ctx)
                case UnaryOpKind.RANGE:
                    return self._apply_range(e.arg, ctx)
                case UnaryOpKind.SHAPE:
                    return self._apply_shape(e.arg, ctx)
                case UnaryOpKind.DIM:
                    return self._apply_dim(e.arg, ctx)
                case _:
                    raise RuntimeError('unknown operator', e.op)

    def _visit_binaryop(self, e: BinaryOp, ctx: Context):
        fn = _binary_table.get(e.op)
        if fn is not None:
            return self._apply_method(fn, (e.left, e.right), ctx)
        else:
            match e.op:
                case BinaryOpKind.SIZE:
                    return self._apply_size(e.left, e.right, ctx)
                case _:
                    raise RuntimeError('unknown operator', e.op)

    def _visit_ternaryop(self, e: TernaryOp, ctx: Context):
        fn = _ternary_table.get(e.op)
        if fn is not None:
            return self._apply_method(fn, (e.arg0, e.arg1, e.arg2), ctx)
        else:
            raise RuntimeError('unknown operator', e.op)

    def _visit_naryop(self, e: NaryOp, ctx: Context):
        match e.op:
            case NaryOpKind.AND:
                return self._apply_and(e.args, ctx)
            case NaryOpKind.OR:
                return self._apply_or(e.args, ctx)
            case NaryOpKind.ZIP:
                return self._apply_zip(e.args, ctx)
            case _:
                raise RuntimeError('unknown operator', e.op)

    def _visit_call(self, e: Call, ctx: Context):
        args = [self._visit_expr(arg, ctx) for arg in e.args]
        fn = self.foreign[e.op]
        if isinstance(fn, Function):
            # calling FPy function
            rt = _Interpreter(fn.env, override_ctx=self.override_ctx)
            return rt.eval(fn.ast, args, ctx)
        elif callable(fn):
            # calling foreign function
            return fn(*args)
        else:
            raise RuntimeError(f'not a function {fn}')

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
        return NDArray([self._visit_expr(x, ctx) for x in e.args])

    def _visit_tuple_ref(self, e: TupleRef, ctx: Context):
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

    def _visit_tuple_set(self, e: TupleSet, ctx: Context):
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
        ctx: Context,
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
                        self.env[target] = val
                    case TupleBinding():
                        self._unpack_tuple(target, val, ctx)
                    case _:
                        raise RuntimeError('unreachable', target)
                self._apply_comp(bindings[1:], elt, ctx, elts)

    def _visit_comp_expr(self, e: CompExpr, ctx: Context):
        # evaluate comprehension
        elts: list[Any] = []
        bindings = list(zip(e.targets, e.iterables))
        self._apply_comp(bindings, e.elt, ctx, elts)

        # remove temporarily bound variables
        for target in e.targets:
            match target:
                case NamedId():
                    del self.env[target]
                case TupleBinding():
                    for var in target.names():
                        del self.env[var]

        return NDArray(elts)

    def _visit_if_expr(self, e: IfExpr, ctx: Context):
        cond = self._visit_expr(e.cond, ctx)
        if not isinstance(cond, bool):
            raise TypeError(f'expected a boolean, got {cond}')
        return self._visit_expr(e.ift if cond else e.iff, ctx)

    def _visit_simple_assign(self, stmt: SimpleAssign, ctx: Context) -> None:
        val = self._visit_expr(stmt.expr, ctx)
        match stmt.var:
            case NamedId():
                self.env[stmt.var] = val
            case UnderscoreId():
                pass
            case _:
                raise NotImplementedError('unknown variable', stmt.var)

    def _unpack_tuple(self, binding: TupleBinding, val: NDArray, ctx: Context) -> None:
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

    def _visit_tuple_unpack(self, stmt: TupleUnpack, ctx: Context) -> None:
        val = self._visit_expr(stmt.expr, ctx)
        if not isinstance(val, NDArray):
            raise TypeError(f'expected a tuple, got {val}')
        self._unpack_tuple(stmt.binding, val, ctx)

    def _visit_index_assign(self, stmt: IndexAssign, ctx: Context) -> None:
        # lookup array
        array = self._lookup(stmt.var)

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

    def _visit_if1(self, stmt: If1Stmt, ctx: Context):
        cond = self._visit_expr(stmt.cond, ctx)
        if not isinstance(cond, bool):
            raise TypeError(f'expected a boolean, got {cond}')
        elif cond:
            self._visit_block(stmt.body, ctx)
            for phi in stmt.phis:
                self.env[phi.name] = self.env[phi.rhs]
        else:
            for phi in stmt.phis:
                self.env[phi.name] = self.env[phi.lhs]

    def _visit_if(self, stmt: IfStmt, ctx: Context) -> None:
        cond = self._visit_expr(stmt.cond, ctx)
        if not isinstance(cond, bool):
            raise TypeError(f'expected a boolean, got {cond}')

        if cond:
            self._visit_block(stmt.ift, ctx)
            for phi in stmt.phis:
                self.env[phi.name] = self.env[phi.lhs]
        else:
            self._visit_block(stmt.iff, ctx)
            for phi in stmt.phis:
                self.env[phi.name] = self.env[phi.rhs]

    def _visit_while(self, stmt: WhileStmt, ctx: Context) -> None:
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

    def _visit_for(self, stmt: ForStmt, ctx: Context) -> None:
        for phi in stmt.phis:
            self.env[phi.name] = self.env[phi.lhs]
            del self.env[phi.lhs]

        iterable = self._visit_expr(stmt.iterable, ctx)
        if not isinstance(iterable, NDArray):
            raise TypeError(f'expected a tensor, got {iterable}')

        for val in iterable:
            match stmt.target:
                case NamedId():
                    self.env[stmt.target] = val
                case TupleBinding():
                    self._unpack_tuple(stmt.target, val, ctx)
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
                    if isinstance(v, Float) and v.is_integer():
                        # HACK: keeps things as specific as possible
                        args.append(int(v))
                    else:
                        args.append(v)

        kwargs: dict[str, Any] = {}
        for k, v in e.kwargs:
            match v:
                case ForeignAttribute():
                    kwargs[k] = self._visit_foreign_attr(v)
                case _:
                    v = self._visit_expr(v, ctx)
                    if isinstance(v, Float) and v.is_integer():
                        kwargs[k] = int(v)
                    else:
                        kwargs[k] = v

        return ctor(*args, **kwargs)

    def _visit_context(self, stmt: ContextStmt, ctx: Context):
        ctx = self._visit_expr(stmt.ctx, ctx)
        return self._visit_block(stmt.body, self._eval_ctx(ctx))

    def _visit_assert(self, stmt: AssertStmt, ctx: Context):
        test = self._visit_expr(stmt.test, ctx)
        if not isinstance(test, bool):
            raise TypeError(f'expected a boolean, got {test}')
        if not test:
            raise AssertionError(stmt.msg)
        return ctx

    def _visit_effect(self, stmt: EffectStmt, ctx: Context):
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

    # override typing hint
    def _visit_statement(self, stmt, ctx: Context) -> None:
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
        return rt.eval(func.ast, args, ctx)

    def eval_expr(self, expr: Expr, env: _Env, ctx: Context):
        rt = _Interpreter(ForeignEnv.empty(), override_ctx=self.ctx, env=env)
        return rt._visit_expr(expr, ctx)
