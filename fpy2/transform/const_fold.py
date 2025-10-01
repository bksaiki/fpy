"""
Constant folding.
"""

import inspect

from fractions import Fraction
from typing import Any, Callable, TypeAlias

from ..analysis import DefineUse, DefineUseAnalysis, Definition
from ..ast.fpyast import *
from ..ast.visitor import DefaultTransformVisitor
from ..env import ForeignEnv
from ..fpc_context import FPCoreContext
from ..number import Float, RealFloat, REAL
from ..utils import is_dyadic

from .. import ops

ScalarValue: TypeAlias = bool | Float | Fraction | Context
TupleValue: TypeAlias = tuple['Value', ...]
Value: TypeAlias = ScalarValue | TupleValue

# Table from `fpy2/interpret/default.py`
_NULLARY_TABLE: dict[type[NullaryOp], Callable[..., Float]] = {
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

_UNARY_TABLE: dict[type[UnaryOp], Callable[..., Float | bool]] = {
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
    RoundExact: ops.round_exact
}

_BINARY_TABLE: dict[type[BinaryOp], Callable[..., Float]] = {
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

_TERNARY_TABLE: dict[type[TernaryOp], Callable[..., Float]] = {
    Fma: ops.fma,
}

class _ConstFoldInstance(DefaultTransformVisitor):
    """
    Constant folding instance for a function.
    """

    func: FuncDef
    env: ForeignEnv
    def_use: DefineUseAnalysis
    enable_context: bool
    enable_op: bool

    vals: dict[Definition, Value]
    remap: dict[Var, Var]

    def __init__(
        self,
        func: FuncDef,
        env: ForeignEnv,
        def_use: DefineUseAnalysis,
        enable_context: bool,
        enable_op: bool,
    ):
        self.func = func
        self.env = env
        self.def_use = def_use
        self.enable_context = enable_context
        self.enable_op = enable_op
        self.vals = {}
        self.remap = {}

    def _is_value(self, e: Expr) -> bool:
        match e:
            case Var():
                d = self.def_use.find_def_from_use(self.remap.get(e, e))
                return d in self.vals
            case BoolVal() | RationalVal() | ForeignVal():
                return True
            case Attribute():
                return self._is_value(e.value)
            case TupleExpr():
                return all(self._is_value(elt) for elt in e.elts)
            case _:
                return False

    def _value(self, e: Expr) -> Value:
        match e:
            case Var():
                d = self.def_use.find_def_from_use(self.remap.get(e, e))
                if d not in self.vals:
                    raise RuntimeError(f'variable `{e.name}` is not a constant')
                return self.vals[d]
            case BoolVal() | ForeignVal():
                return e.val
            case RationalVal():
                return e.as_rational()
            case Attribute():
                val = self._value(e.value)
                if hasattr(val, e.attr):
                    return getattr(val, e.attr)
                elif isinstance(val, dict) and e.attr in val:
                    return val[e.attr]
                else:
                    raise RuntimeError(f'unknown attribute `{e.attr}` for `{val}`')
            case TupleExpr():
                return tuple(self._value(elt) for elt in e.elts)
            case _:
                raise NotImplementedError(e)

    def _rational_as_ast(self, x: Fraction, loc: Location | None) -> Expr:
        if x.denominator == 1:
            return Integer(int(x), loc)
        else:
            # TODO: emitting a rational node requires a name:
            # could be `rational` or `fp.rational` or `fpy2.rational`
            func = Attribute(Var(NamedId('fp'), loc), 'rational', loc)
            return Rational(func, x.numerator, x.denominator, loc)

    def _cvt_float(self, x: Value) -> Float:
        match x:
            case Float():
                return x
            case Fraction():
                if not is_dyadic(x):
                    raise TypeError(f'expected a dyadic rational, got `{x}`')
                return Float.from_rational(x, ctx=REAL)
            case _:
                raise TypeError(f'expected a real number, got `{x}`')

    def _cvt_context_arg(self, name: str, arg: Any, ty: type):
        if ty is int:
            # convert to int
            val = self._cvt_float(arg)
            if not val.is_integer():
                raise TypeError(f'expected an integer argument for `{name}={arg}`')
            return int(val)
        elif ty is float:
            # convert to float
            raise NotImplementedError(arg, ty)
        elif ty is RealFloat:
            # convert to RealFloat
            raise NotImplementedError(arg, ty)
        else:
            # don't apply a conversion
            return arg

    def _construct_context(self, cls: type[Context], args: list[Value], kwargs: dict[str, Value]):
        sig = inspect.signature(cls.__init__)
        params = list(sig.parameters)

        ctor_args = []
        for arg, name in zip(args, params[1:]):
            param = sig.parameters[name]
            ctor_arg = self._cvt_context_arg(name, arg, param.annotation)
            ctor_args.append(ctor_arg)

        ctor_kwargs = {}
        for name, val in kwargs.items():
            if name not in sig.parameters:
                raise TypeError(f'unknown parameter {name} for constructor {cls}')
            param = sig.parameters[name]
            ctor_kwargs[name] = self._cvt_context_arg(name, val, param.annotation)

        return cls(*ctor_args, **ctor_kwargs)

    def _visit_var(self, e: Var, ctx: Context | None):
        e2 = super()._visit_var(e, ctx)
        self.remap[e2] = e
        return e2

    def _visit_attribute(self, e: Attribute, ctx: Context | None):
        value = self._visit_expr(e.value, ctx)
        if self._is_value(value):
            val = self._value(value)
            if hasattr(val, e.attr):
                return ForeignVal(getattr(val, e.attr), e.loc)
            elif isinstance(val, dict) and e.attr in val:
                return ForeignVal(val[e.attr], e.loc)
            else:
                raise RuntimeError(f'unknown attribute `{e.attr}` for `{val}`')
        else:
            # otherwise, just visit the value
            return Attribute(value, e.attr, e.loc)

    def _visit_nullaryop(self, e: NullaryOp, ctx: Context | None):
        cls = type(e)
        if not self.enable_op or ctx is None or cls not in _NULLARY_TABLE:
            # one of the following is true:
            # - constant folding for operations is disabled
            # - context is unknown so we cannot evaluate
            # - operation is unknown
            return cls(e.func, e.loc)

        # evaluate the operation
        fn = _NULLARY_TABLE[cls]
        val = fn(ctx=ctx)
        return self._rational_as_ast(val.as_rational(), e.loc)

    def _visit_unaryop(self, e: UnaryOp, ctx: Context | None):
        cls = type(e)
        arg = self._visit_expr(e.arg, ctx)
        if not self.enable_op or ctx is None or cls not in _UNARY_TABLE or not self._is_value(arg):
            # one of the following is true:
            # - constant folding for operations is disabled
            # - context is unknown so we cannot evaluate
            # - operation is unknown
            # - argument is not a constant
            if isinstance(e, NamedUnaryOp):
                return type(e)(e.func, arg, e.loc)
            else:
                return type(e)(arg, e.loc)

        # real -> real or real -> bool operation
        arg_val = self._value(arg)
        if not isinstance(arg_val, (Float, Fraction)):
            raise TypeError(f'expected a real number, got `{arg_val}`')

        # evaluate the operation
        fn = _UNARY_TABLE[cls]
        val = fn(self._cvt_float(arg_val), ctx=ctx)
        match val:
            case Float():
                return self._rational_as_ast(val.as_rational(), e.loc)
            case bool():
                return BoolVal(val, e.loc)
            case _:
                raise RuntimeError(f'unexpected return value `{val}` from `{fn}`')

    def _visit_binaryop(self, e: BinaryOp, ctx: Context | None):
        cls = type(e)
        first = self._visit_expr(e.first, ctx)
        second = self._visit_expr(e.second, ctx)
        if not self.enable_op or ctx is None or cls not in _BINARY_TABLE or not (self._is_value(first) and self._is_value(second)):
            # one of the following is true:
            # - constant folding for operations is disabled
            # - context is unknown so we cannot evaluate
            # - operation is unknown
            # - argument is not a constant
            return cls(first, second, e.loc)

        # real -> real operation
        first_val = self._value(first)
        second_val = self._value(second)
        if not isinstance(first_val, (Float, Fraction)) or not isinstance(second_val, (Float, Fraction)):
            raise TypeError(f'expected real numbers, got `{first_val}` and `{second_val}`')

        # evaluate the operation
        fn = _BINARY_TABLE[cls]
        val = fn(self._cvt_float(first_val), self._cvt_float(second_val), ctx=ctx)
        return self._rational_as_ast(val.as_rational(), e.loc)

    def _visit_ternaryop(self, e: TernaryOp, ctx: Context | None):
        cls = type(e)
        first = self._visit_expr(e.first, ctx)
        second = self._visit_expr(e.second, ctx)
        third = self._visit_expr(e.third, ctx)
        if not self.enable_op or ctx is None or cls not in _TERNARY_TABLE or not (self._is_value(first) and self._is_value(second) and self._is_value(third)):
            # one of the following is true:
            # - constant folding for operations is disabled
            # - context is unknown so we cannot evaluate
            # - operation is unknown
            # - argument is not a constant
            return cls(first, second, third, e.loc)

        # real -> real operation
        first_val = self._value(first)
        second_val = self._value(second)
        third_val = self._value(third)
        if not isinstance(first_val, (Float, Fraction)) or not isinstance(second_val, (Float, Fraction)) or not isinstance(third_val, (Float, Fraction)):
            raise TypeError(f'expected real numbers, got `{first_val}`, `{second_val}`, and `{third_val}`')

        # evaluate the operation
        fn = _TERNARY_TABLE[cls]
        val = fn(self._cvt_float(first_val), self._cvt_float(second_val), self._cvt_float(third_val), ctx=ctx)
        return self._rational_as_ast(val.as_rational(), e.loc)

    def _visit_call(self, e: Call, ctx: Context | None):
        args = [ self._visit_expr(arg, ctx) for arg in e.args ]
        kwargs = [ (k, self._visit_expr(v, ctx)) for k, v in e.kwargs ]
        if (
            self.enable_context
            and isinstance(e.fn, type)
            and issubclass(e.fn, Context)
            and all(self._is_value(arg) for arg in args)
            and all(self._is_value(v) for _, v in kwargs)
        ):
            # context constructor
            vals = [ self._value(arg) for arg in args ]
            kwvals = { k: self._value(v) for k, v in kwargs }
            val = self._construct_context(e.fn, vals, kwvals)
            return ForeignVal(val, e.loc)
        else:
            # otherwise
            return Call(e.func, e.fn, args, kwargs, e.loc)

    def _visit_context(self, stmt: ContextStmt, ctx: Context | None):
        ctx_e = self._visit_expr(stmt.ctx, ctx)
        if isinstance(ctx_e, ForeignVal) and isinstance(ctx_e.val, Context):
            # we can determine the context
            new_ctx = ctx_e.val
        else:
            # otherwise, we cannot
            new_ctx = None

        body, _ = self._visit_block(stmt.body, new_ctx)
        s = ContextStmt(stmt.target, ctx_e, body, stmt.loc)
        return s, ctx

    def _visit_function(self, func: FuncDef, ctx: None):
        # extract overriding context
        match func.ctx:
            case None:
                fctx: Context | None = None
            case FPCoreContext():
                fctx = func.ctx.to_context()
            case Context():
                fctx = func.ctx
            case _:
                raise RuntimeError(f'unreachable: {func.ctx}')

        # bind foreign values
        for name in func.free_vars:
            d = self.def_use.find_def_from_site(name, func)
            self.vals[d] = self.env[name.base]

        return super()._visit_function(func, fctx)

    def apply(self) -> FuncDef:
        return self._visit_function(self.func, None)


class ConstFold:
    """
    Constant folding.

    This transform evaluates expressions that can be determined statically:
    - constants constructors
    - operations (math, lists, etc.)
    """

    @staticmethod
    def apply(
        func: FuncDef,
        env: ForeignEnv,
        *,
        def_use: DefineUseAnalysis | None = None,
        enable_context: bool = True,
        enable_op: bool = True,
    ) -> FuncDef:
        """
        Applies constant folding.

        Optionally, specify:
        - `enable_context`: whether to enable constant folding for context constructors [default: True]
        - `enable_op`: whether to enable constant folding for operations [default: True]
        """
        if not isinstance(func, FuncDef):
            raise TypeError(f'Expected `FuncDef`, got {type(func)} for {func}')
        if def_use is None:
            def_use = DefineUse.analyze(func)
        inst = _ConstFoldInstance(func, env, def_use, enable_context, enable_op)
        return inst.apply()
