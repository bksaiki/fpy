"""
FPy runtime backed by the Titanic library.
"""

from fractions import Fraction
from typing import Any, Callable, Optional, Sequence, TypeAlias

from titanfp.titanic.ndarray import NDArray

from .. import math

from ..fpc_context import FPCoreContext
from ..number import Context, Float, IEEEContext, RM
from ..number.gmp import mpfr_constant
from ..runtime.trace import ExprTraceEntry
from ..runtime.env import ForeignEnv
from ..function import Function
from ..ir import *
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

_method_table: dict[str, Callable[..., Any]] = {
    '+': math.add,
    '-': math.sub,
    '*': math.mul,
    '/': math.div,
    'fabs': math.fabs,
    'sqrt': math.sqrt,
    'fma': math.fma,
    'neg': math.neg,
    'copysign': math.copysign,
    'fdim': math.fdim,
    'fmax': math.fmax,
    'fmin': math.fmin,
    'fmod': math.fmod,
    'remainder': math.remainder,
    'hypot': math.hypot,
    'cbrt': math.cbrt,
    'ceil': math.ceil,
    'floor': math.floor,
    'nearbyint': math.nearbyint,
    'round': math.round,
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
    'exp2': math.exp2,
    'expm1': math.expm1,
    'log': math.log,
    'log10': math.log10,
    'log1p': math.log1p,
    'log2': math.log2,
    'pow': math.pow,
    'erf': math.erf,
    'erfc': math.erfc,
    'lgamma': math.lgamma,
    'tgamma': math.tgamma,
    'isfinite': _isfinite,
    'isinf': _isinf,
    'isnan': _isnan,
    'isnormal': _isnormal,
    'signbit': _signbit,
}

_Env: TypeAlias = dict[NamedId, ScalarVal | TensorVal]

_PY_CTX = IEEEContext(11, 64, RM.RNE)
"""the native Python floating-point context"""


class _Interpreter(ReduceVisitor):
    """Single-use interpreter for a function"""

    foreign: ForeignEnv
    """foreign environment"""
    override_ctx: Optional[Context]
    """optional overriding context"""
    env: _Env
    """Environment mapping variable names to values"""
    trace: list[ExprTraceEntry]
    """expression trace"""
    enable_trace: bool
    """expression tracing enabled?"""

    def __init__(
        self, 
        foreign: ForeignEnv,
        *,
        override_ctx: Optional[Context] = None,
        env: Optional[_Env] = None,
        enable_trace: bool = False
    ):
        if env is None:
            env = {}

        self.foreign = foreign
        self.override_ctx = override_ctx
        self.env = env
        self.trace = []
        self.enable_trace = enable_trace

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
            match arg.ty:
                case AnyType():
                    x = self._arg_to_mpmf(val, ctx)
                    if isinstance(arg.name, NamedId):
                        self.env[arg.name] = x
                case RealType():
                    x = self._arg_to_mpmf(val, ctx)
                    if not isinstance(x, Float):
                        raise NotImplementedError(f'argument is a scalar, got data {val}')
                    if isinstance(arg.name, NamedId):
                        self.env[arg.name] = x
                case _:
                    raise NotImplementedError(f'unknown argument type {arg.ty}')

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

    def _apply_method(self, e: NaryExpr, ctx: Context):
        fn = _method_table[e.name]
        args: list[Float] = []
        for arg in e.children:
            val = self._visit_expr(arg, ctx)
            if not isinstance(val, Float):
                raise TypeError(f'expected a real number argument for {e.name}, got {val}')
            args.append(val)

        # compute the result
        return fn(*args, ctx=ctx)

    def _apply_cast(self, e: Cast, ctx: Context):
        x = self._visit_expr(e.children[0], ctx)
        if not isinstance(x, Float):
            raise TypeError(f'expected a real number argument, got {x}')
        return ctx.round(x)

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

    def _apply_shape(self, e: Shape, ctx: Context):
        v = self._visit_expr(e.children[0], ctx)
        if not isinstance(v, NDArray):
            raise TypeError(f'expected a tensor, got {v}')
        return NDArray([ctx.round(x) for x in v.shape])

    def _apply_range(self, e: Range, ctx: Context):
        stop = self._visit_expr(e.children[0], ctx)
        if not isinstance(stop, Float):
            raise TypeError(f'expected a real number argument, got {stop}')
        if not stop.is_integer():
            raise TypeError(f'expected an integer argument, got {stop}')

        elts: list[Float] = []
        for i in range(int(stop)):
            elts.append(Float.from_int(i, ctx=ctx))
        return NDArray(elts)

    def _apply_dim(self, e: Dim, ctx: Context):
        v = self._visit_expr(e.children[0], ctx)
        if not isinstance(v, NDArray):
            raise TypeError(f'expected a tensor, got {v}')
        return Float.from_int(len(v.shape), ctx=ctx)

    def _apply_size(self, e: Size, ctx: Context):
        v = self._visit_expr(e.children[0], ctx)
        if not isinstance(v, NDArray):
            raise TypeError(f'expected a tensor, got {v}')
        dim = self._visit_expr(e.children[1], ctx)
        if not isinstance(dim, Float):
            raise TypeError(f'expected a real number argument, got {dim}')
        if not dim.is_integer():
            raise TypeError(f'expected an integer argument, got {dim}')
        return Float.from_int(v.shape[int(dim)], ctx=ctx)

    def _visit_nary_expr(self, e: NaryExpr, ctx: Context):
        if e.name in _method_table:
            return self._apply_method(e, ctx)
        elif isinstance(e, Cast):
            return self._apply_cast(e, ctx)
        elif isinstance(e, Not):
            return self._apply_not(e, ctx)
        elif isinstance(e, And):
            return self._apply_and(e, ctx)
        elif isinstance(e, Or):
            return self._apply_or(e, ctx)
        elif isinstance(e, Shape):
            return self._apply_shape(e, ctx)
        elif isinstance(e, Range):
            return self._apply_range(e, ctx)
        elif isinstance(e, Dim):
            return self._apply_dim(e, ctx)
        elif isinstance(e, Size):
            return self._apply_size(e, ctx)
        else:
            raise NotImplementedError('unknown n-ary expression', e)

    def _visit_unknown(self, e: UnknownCall, ctx: Context):
        args = [self._visit_expr(arg, ctx) for arg in e.children]
        fn = self.foreign[e.name]
        if isinstance(fn, Function):
            # calling FPy function
            rt = _Interpreter(fn.env, override_ctx=self.override_ctx)
            return rt.eval(fn.to_ir(), args, ctx)
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
        lhs = self._visit_expr(e.children[0], ctx)
        for op, arg in zip(e.ops, e.children[1:]):
            rhs = self._visit_expr(arg, ctx)
            if not self._apply_cmp2(op, lhs, rhs):
                return False
            lhs = rhs
        return True

    def _visit_tuple_expr(self, e: TupleExpr, ctx: Context):
        return NDArray([self._visit_expr(x, ctx) for x in e.children])

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
            if not isinstance(array, NDArray):
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

        return NDArray(elts)

    def _visit_if_expr(self, e: IfExpr, ctx: Context):
        cond = self._visit_expr(e.cond, ctx)
        if not isinstance(cond, bool):
            raise TypeError(f'expected a boolean, got {cond}')
        return self._visit_expr(e.ift if cond else e.iff, ctx)

    def _visit_simple_assign(self, stmt: SimpleAssign, ctx: Context) -> None:
        val = self._visit_expr(stmt.expr, ctx)
        if self.enable_trace:
            entry = ExprTraceEntry(stmt.expr, val, dict(self.env), ctx)
            self.trace.append(entry)

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

        if self.enable_trace:
            entry = ExprTraceEntry(stmt.expr, val, dict(self.env), ctx)
            self.trace.append(entry)

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

        if self.enable_trace:
            entry = ExprTraceEntry(stmt.cond, cond, dict(self.env), ctx)
            self.trace.append(entry)

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

        if self.enable_trace:
            entry = ExprTraceEntry(stmt.cond, cond, dict(self.env), ctx)
            self.trace.append(entry)

        while cond:
            self._visit_block(stmt.body, ctx)
            for phi in stmt.phis:
                self.env[phi.name] = self.env[phi.rhs]
                del self.env[phi.rhs]

            cond = self._visit_expr(stmt.cond, ctx)
            if not isinstance(cond, bool):
                raise TypeError(f'expected a boolean, got {cond}')

            if self.enable_trace:
                entry = ExprTraceEntry(stmt.cond, cond, dict(self.env), ctx)
                self.trace.append(entry)


    def _visit_for(self, stmt: ForStmt, ctx: Context) -> None:
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
        val = self._visit_expr(stmt.expr, ctx)
        if self.enable_trace:
            entry = ExprTraceEntry(stmt.expr, val, dict(self.env), ctx)
            self.trace.append(entry)
        return val

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
        return rt.eval(func.to_ir(), args, ctx)

    def eval_with_trace(self, func: Function, args: Sequence[Any], ctx = None):
        rt = _Interpreter(func.env, override_ctx=self.ctx, enable_trace=True)
        result = rt.eval(func.to_ir(), args, ctx)
        return result, rt.trace

    def eval_expr(self, expr: Expr, env: _Env, ctx: Context):
        rt = _Interpreter(ForeignEnv.empty(), override_ctx=self.ctx, env=env)
        return rt._visit_expr(expr, ctx)
