"""
MPFR-backed engine for round-to-odd arithmetic.

This engine wraps the gmpy2/MPFR library to provide
round-to-odd arithmetic operations.
"""

import enum
import math
from collections.abc import Callable
from fractions import Fraction

import gmpy2 as gmp

from ...utils import enum_repr
from ..context import Context
from ..gmputils import float_to_mpfr, mpfr_call
from ..number import Float
from .engine import Engine, EngineArg, EngineRes


@enum_repr
class _Constant(enum.Enum):
    """
    All constants defined in C99 standard `math.h`.
    """
    E = 0
    LOG2E = 1
    LOG10E = 2
    LN2 = 3
    LN10 = 4
    PI = 5
    PI_2 = 6
    PI_4 = 7
    M_1_PI = 8
    M_2_PI = 9
    M_2_SQRTPI = 10
    SQRT2 = 11
    SQRT1_2 = 12


def _gmp_neg(x: gmp.mpfr) -> gmp.mpfr:
    return -x

def _gmp_abs(x: gmp.mpfr) -> gmp.mpfr:
    return abs(x)

def _gmp_pow(x: gmp.mpfr, y: gmp.mpfr) -> gmp.mpfr:
    return x ** y

def _gmp_lgamma(x: gmp.mpfr) -> gmp.mpfr:
    y, _ = gmp.lgamma(x)
    return y

def _mpfr_eval(
    gmp_fn: Callable[..., gmp.mpfr],
    *args: Float,
    prec: int | None = None,
    n: int | None = None) -> Float:
    """
    Evaluates `gmp_fn(*args)` such that the result may be safely re-rounded.
    Either specify:
    - `prec`: the number of digits, or
    - `n`: the first unrepresentable digit
    If both are specified, `prec` takes precedence.
    """
    gmp_args = tuple(float_to_mpfr(x) for x in args)
    return mpfr_call(gmp_fn, gmp_args, prec=prec, n=n)


# From `titanfp` package
# TODO: some of these are unsafe
_constant_exprs: dict[_Constant, Callable[[], gmp.mpfr]] = {
    _Constant.E : lambda : gmp.exp(1),
    _Constant.LOG2E : lambda: gmp.log2(gmp.exp(1)), # TODO: may be inaccurate
    _Constant.LOG10E : lambda: gmp.log10(gmp.exp(1)), # TODO: may be inaccurate
    _Constant.LN2 : gmp.const_log2,
    _Constant.LN10 : lambda: gmp.log(10),
    _Constant.PI : gmp.const_pi,
    _Constant.PI_2 : lambda: gmp.const_pi() / 2, # division by 2 is exact
    _Constant.PI_4 : lambda: gmp.const_pi() / 4, # division by 4 is exact
    _Constant.M_1_PI : lambda: 1 / gmp.const_pi(), # TODO: may be inaccurate
    _Constant.M_2_PI : lambda: 2 / gmp.const_pi(), # TODO: may be inaccurate
    _Constant.M_2_SQRTPI : lambda: 2 / gmp.sqrt(gmp.const_pi()), # TODO: may be inaccurate
    _Constant.SQRT2: lambda: gmp.sqrt(2),
    _Constant.SQRT1_2: lambda: gmp.sqrt(gmp.div(gmp.mpfr(1), gmp.mpfr(2))),
}


def _mpfr_constant(x: _Constant, *, prec: int | None = None, n: int | None = None) -> Float:
    """
    Computes constant `x` such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    try:
        fn = _constant_exprs[x]
        return mpfr_call(fn, (), prec=prec, n=n)
    except KeyError as e:
        raise ValueError(f'unknown constant {e.args[0]!r}') from None


def _apply(gmp_fn: Callable[..., gmp.mpfr], ctx: Context, *args: EngineArg) -> EngineRes:
    """`gmp_fn(*args)`, to be re-rounded under *ctx*; `None` for an exact
    operand or an unbounded context, which this engine does not handle."""
    xs = [x for x in args if isinstance(x, Float)]
    if len(xs) < len(args):
        return None
    prec, n = ctx.round_params()
    if prec is None and n is None:
        return None
    return _mpfr_eval(gmp_fn, *xs, prec=prec, n=n)


def _constant(c: _Constant, ctx: Context) -> EngineRes:
    """The constant *c*, to be re-rounded under *ctx*; `None` for an unbounded
    context."""
    prec, n = ctx.round_params()
    if prec is None and n is None:
        return None
    return _mpfr_constant(c, prec=prec, n=n)


_mpfr_engine_inst = None
"""single instance of MPFR engine"""


class MPFREngine(Engine):
    """
    Engine that uses MPFR (via gmpy2) and round-to-odd arithmetic.

    This engine can handle all operations except exact arithmetic, and
    `ceil`/`floor`/`trunc`/`roundint`, whose gmpy2 results are integers.
    """

    @staticmethod
    def instance() -> 'MPFREngine':
        """Returns the singleton instance of the MPFR engine."""
        global _mpfr_engine_inst
        if _mpfr_engine_inst is None:
            _mpfr_engine_inst = MPFREngine()
        return _mpfr_engine_inst

    def acos(self, x: EngineArg, ctx: Context) -> EngineRes:
        return _apply(gmp.acos, ctx, x)

    def acosh(self, x: EngineArg, ctx: Context) -> EngineRes:
        return _apply(gmp.acosh, ctx, x)

    def asin(self, x: EngineArg, ctx: Context) -> EngineRes:
        return _apply(gmp.asin, ctx, x)

    def asinh(self, x: EngineArg, ctx: Context) -> EngineRes:
        return _apply(gmp.asinh, ctx, x)

    def atan(self, x: EngineArg, ctx: Context) -> EngineRes:
        return _apply(gmp.atan, ctx, x)

    def atanh(self, x: EngineArg, ctx: Context) -> EngineRes:
        return _apply(gmp.atanh, ctx, x)

    def cbrt(self, x: EngineArg, ctx: Context) -> EngineRes:
        return _apply(gmp.cbrt, ctx, x)

    def cos(self, x: EngineArg, ctx: Context) -> EngineRes:
        return _apply(gmp.cos, ctx, x)

    def cosh(self, x: EngineArg, ctx: Context) -> EngineRes:
        return _apply(gmp.cosh, ctx, x)

    def erf(self, x: EngineArg, ctx: Context) -> EngineRes:
        return _apply(gmp.erf, ctx, x)

    def erfc(self, x: EngineArg, ctx: Context) -> EngineRes:
        return _apply(gmp.erfc, ctx, x)

    def exp(self, x: EngineArg, ctx: Context) -> EngineRes:
        return _apply(gmp.exp, ctx, x)

    def exp2(self, x: EngineArg, ctx: Context) -> EngineRes:
        return _apply(gmp.exp2, ctx, x)

    def exp10(self, x: EngineArg, ctx: Context) -> EngineRes:
        return _apply(gmp.exp10, ctx, x)

    def expm1(self, x: EngineArg, ctx: Context) -> EngineRes:
        return _apply(gmp.expm1, ctx, x)

    def fabs(self, x: EngineArg, ctx: Context) -> EngineRes:
        return _apply(_gmp_abs, ctx, x)

    def lgamma(self, x: EngineArg, ctx: Context) -> EngineRes:
        return _apply(_gmp_lgamma, ctx, x)

    def log(self, x: EngineArg, ctx: Context) -> EngineRes:
        return _apply(gmp.log, ctx, x)

    def log10(self, x: EngineArg, ctx: Context) -> EngineRes:
        return _apply(gmp.log10, ctx, x)

    def log1p(self, x: EngineArg, ctx: Context) -> EngineRes:
        return _apply(gmp.log1p, ctx, x)

    def log2(self, x: EngineArg, ctx: Context) -> EngineRes:
        return _apply(gmp.log2, ctx, x)

    def neg(self, x: EngineArg, ctx: Context) -> EngineRes:
        return _apply(_gmp_neg, ctx, x)

    def sin(self, x: EngineArg, ctx: Context) -> EngineRes:
        return _apply(gmp.sin, ctx, x)

    def sinh(self, x: EngineArg, ctx: Context) -> EngineRes:
        return _apply(gmp.sinh, ctx, x)

    def sqrt(self, x: EngineArg, ctx: Context) -> EngineRes:
        return _apply(gmp.sqrt, ctx, x)

    def tan(self, x: EngineArg, ctx: Context) -> EngineRes:
        return _apply(gmp.tan, ctx, x)

    def tanh(self, x: EngineArg, ctx: Context) -> EngineRes:
        return _apply(gmp.tanh, ctx, x)

    def tgamma(self, x: EngineArg, ctx: Context) -> EngineRes:
        return _apply(gmp.gamma, ctx, x)

    def add(self, x: EngineArg, y: EngineArg, ctx: Context) -> EngineRes:
        return _apply(gmp.add, ctx, x, y)

    def atan2(self, y: EngineArg, x: EngineArg, ctx: Context) -> EngineRes:
        return _apply(gmp.atan2, ctx, y, x)

    def copysign(self, x: EngineArg, y: EngineArg, ctx: Context) -> EngineRes:
        return _apply(gmp.copy_sign, ctx, x, y)

    def div(self, x: EngineArg, y: EngineArg, ctx: Context) -> EngineRes:
        return _apply(gmp.div, ctx, x, y)

    def _fdim(self, x: Float, y: Float, prec: int | None, n: int | None) -> Float:
        if x.isnan or y.isnan:
            # C reference: if either argument is NaN, NaN is returned
            return Float(isnan=True)
        elif x > y:
            # if `x > y`, returns `x - y`
            return _mpfr_eval(gmp.sub, x, y, prec=prec, n=n)
        else:
            # otherwise, returns +0
            return Float()

    def fdim(self, x: EngineArg, y: EngineArg, ctx: Context) -> EngineRes:
        if isinstance(x, Fraction) or isinstance(y, Fraction):
            return None
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return self._fdim(x, y, prec, n)

    def fmod(self, x: EngineArg, y: EngineArg, ctx: Context) -> EngineRes:
        return _apply(gmp.fmod, ctx, x, y)

    def fmax(self, x: EngineArg, y: EngineArg, ctx: Context) -> EngineRes:
        return _apply(gmp.maxnum, ctx, x, y)

    def fmin(self, x: EngineArg, y: EngineArg, ctx: Context) -> EngineRes:
        return _apply(gmp.minnum, ctx, x, y)

    def hypot(self, x: EngineArg, y: EngineArg, ctx: Context) -> EngineRes:
        return _apply(gmp.hypot, ctx, x, y)

    def _mod(self, x: Float, y: Float) -> Float:
        if x.isnan or y.isnan:
            # if either argument is NaN, NaN is returned
            return Float(isnan=True)
        elif x.isinf:
            # if x is infinite, NaN is returned
            return Float(isnan=True)
        elif y.isinf:
            # if y is infinite, ...
            if x.is_zero():
                # if x is +/-0, returns copysign(x, y)
                return Float(x=x, s=y.s)
            elif x.s == y.s:
                # same sign => returns x
                return x
            else:
                # different sign => returns y
                return y
        elif y.is_zero():
            # if y is zero, NaN is returned
            return Float(isnan=True)
        elif x.is_zero():
            # if x is zero, +/-0 is returned
            return Float(x=x, s=y.s)
        else:
            # x, y are both finite and non-zero
            # manually compute `x - floor(x / y) * y`

            # step 1. compute `floor(x / y)`
            q = math.floor(_mpfr_eval(gmp.div, x, y, n=-1))

            # step 2. compute `x - q * y`
            return x - q * y

    def mod(self, x: EngineArg, y: EngineArg, ctx: Context) -> EngineRes:
        if isinstance(x, Fraction) or isinstance(y, Fraction):
            return None
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return self._mod(x, y)

    def mul(self, x: EngineArg, y: EngineArg, ctx: Context) -> EngineRes:
        return _apply(gmp.mul, ctx, x, y)

    def pow(self, x: EngineArg, y: EngineArg, ctx: Context) -> EngineRes:
        return _apply(_gmp_pow, ctx, x, y)

    def remainder(self, x: EngineArg, y: EngineArg, ctx: Context) -> EngineRes:
        return _apply(gmp.remainder, ctx, x, y)

    def sub(self, x: EngineArg, y: EngineArg, ctx: Context) -> EngineRes:
        return _apply(gmp.sub, ctx, x, y)

    def fma(self, x: EngineArg, y: EngineArg, z: EngineArg, ctx: Context) -> EngineRes:
        return _apply(gmp.fma, ctx, x, y, z)

    def const_e(self, ctx: Context) -> EngineRes:
        return _constant(_Constant.E, ctx)

    def const_log2e(self, ctx: Context) -> EngineRes:
        return _constant(_Constant.LOG2E, ctx)

    def const_log10e(self, ctx: Context) -> EngineRes:
        return _constant(_Constant.LOG10E, ctx)

    def const_ln2(self, ctx: Context) -> EngineRes:
        return _constant(_Constant.LN2, ctx)

    def const_ln10(self, ctx: Context) -> EngineRes:
        return _constant(_Constant.LN10, ctx)

    def const_pi(self, ctx: Context) -> EngineRes:
        return _constant(_Constant.PI, ctx)

    def const_pi_2(self, ctx: Context) -> EngineRes:
        return _constant(_Constant.PI_2, ctx)

    def const_pi_4(self, ctx: Context) -> EngineRes:
        return _constant(_Constant.PI_4, ctx)

    def const_1_pi(self, ctx: Context) -> EngineRes:
        return _constant(_Constant.M_1_PI, ctx)

    def const_2_pi(self, ctx: Context) -> EngineRes:
        return _constant(_Constant.M_2_PI, ctx)

    def const_2_sqrt_pi(self, ctx: Context) -> EngineRes:
        return _constant(_Constant.M_2_SQRTPI, ctx)

    def const_sqrt2(self, ctx: Context) -> EngineRes:
        return _constant(_Constant.SQRT2, ctx)

    def const_sqrt1_2(self, ctx: Context) -> EngineRes:
        return _constant(_Constant.SQRT1_2, ctx)

