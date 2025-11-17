"""
MPFR-backed engine for round-to-odd arithmetic.

This engine wraps the gmpy2/MPFR library to provide
round-to-odd arithmetic operations.
"""

import enum
import gmpy2 as gmp
import math

from fractions import Fraction
from typing import Callable

from ...utils import enum_repr
from ..context import Context
from ..number import Float
from ..gmp import float_to_mpfr, mpfr_call

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
    INFINITY = 13
    NAN = 14


def _gmp_neg(x):
    return -x

def _gmp_abs(x):
    return abs(x)

def _gmp_pow(x, y):
    return x ** y

def _gmp_lgamma(x):
    y, _ = gmp.lgamma(x)
    return y

def _mpfr_eval(
    gmp_fn: Callable[..., gmp.mpfr],
    *args: Float,
    prec: int | None = None,
    n: int | None = None):
    """
    Evaluates `gmp_fn(*args)` such that the result may be safely re-rounded.
    Either specify:
    - `prec`: the number of digits, or
    - `n`: the first unrepresentable digit
    If both are specified, `prec` takes precedence.
    """
    match prec, n:
        case int(), _:
            n = None
        case None, int():
            pass
        case _:
            raise ValueError('Either `prec` or `n` must be specified')

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


def _mpfr_constant(x: _Constant, *, prec: int | None = None, n: int | None = None):
    """
    Computes constant `x` such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    try:
        fn = _constant_exprs[x]
        return mpfr_call(fn, (), prec=prec, n=n)
    except KeyError as e:
        raise ValueError(f'unknown constant {e.args[0]!r}') from None


_mpfr_engine_inst = None
"""single instance of MPFR engine"""


class MPFREngine(Engine):
    """
    Engine that uses MPFR (via gmpy2) and round-to-odd arithmetic.

    This engine can handle all operations except exact arithmetic.
    """

    @staticmethod
    def instance() -> 'MPFREngine':
        """Returns the singleton instance of the MPFR engine."""
        global _mpfr_engine_inst
        if _mpfr_engine_inst is None:
            _mpfr_engine_inst = MPFREngine()
        return _mpfr_engine_inst

    def acos(self, x: EngineArg, ctx: Context) -> EngineRes:
        if isinstance(x, Fraction):
            return None
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_eval(gmp.acos, x, prec=prec, n=n)

    def acosh(self, x: EngineArg, ctx: Context) -> EngineRes:
        if isinstance(x, Fraction):
            return None
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_eval(gmp.acosh, x, prec=prec, n=n)

    def asin(self, x: EngineArg, ctx: Context) -> EngineRes:
        if isinstance(x, Fraction):
            return None
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_eval(gmp.asin, x, prec=prec, n=n)

    def asinh(self, x: EngineArg, ctx: Context) -> EngineRes:
        if isinstance(x, Fraction):
            return None
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_eval(gmp.asinh, x, prec=prec, n=n)

    def atan(self, x: EngineArg, ctx: Context) -> EngineRes:
        if isinstance(x, Fraction):
            return None
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_eval(gmp.atan, x, prec=prec, n=n)

    def atanh(self, x: EngineArg, ctx: Context) -> EngineRes:
        if isinstance(x, Fraction):
            return None
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_eval(gmp.atanh, x, prec=prec, n=n)

    def cbrt(self, x: EngineArg, ctx: Context) -> EngineRes:
        if isinstance(x, Fraction):
            return None
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_eval(gmp.cbrt, x, prec=prec, n=n)

    # we don't use MPFR's ceil/floor/trunc/roundint
    # since their return value is strange
    def ceil(self, x: EngineArg, ctx: Context) -> EngineRes:
        return None

    def cos(self, x: EngineArg, ctx: Context) -> EngineRes:
        if isinstance(x, Fraction):
            return None
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_eval(gmp.cos, x, prec=prec, n=n)

    def cosh(self, x: EngineArg, ctx: Context) -> EngineRes:
        if isinstance(x, Fraction):
            return None
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_eval(gmp.cosh, x, prec=prec, n=n)

    def erf(self, x: EngineArg, ctx: Context) -> EngineRes:
        if isinstance(x, Fraction):
            return None
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_eval(gmp.erf, x, prec=prec, n=n)

    def erfc(self, x: EngineArg, ctx: Context) -> EngineRes:
        if isinstance(x, Fraction):
            return None
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_eval(gmp.erfc, x, prec=prec, n=n)

    def exp(self, x: EngineArg, ctx: Context) -> EngineRes:
        if isinstance(x, Fraction):
            return None
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_eval(gmp.exp, x, prec=prec, n=n)

    def exp2(self, x: EngineArg, ctx: Context) -> EngineRes:
        if isinstance(x, Fraction):
            return None
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_eval(gmp.exp2, x, prec=prec, n=n)

    def exp10(self, x: EngineArg, ctx: Context) -> EngineRes:
        if isinstance(x, Fraction):
            return None
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_eval(gmp.exp10, x, prec=prec, n=n)

    def expm1(self, x: EngineArg, ctx: Context) -> EngineRes:
        if isinstance(x, Fraction):
            return None
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_eval(gmp.expm1, x, prec=prec, n=n)

    def fabs(self, x: EngineArg, ctx: Context) -> EngineRes:
        if isinstance(x, Fraction):
            return None
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_eval(_gmp_abs, x, prec=prec, n=n)

    # we don't use MPFR's ceil/floor/trunc/roundint
    # since their return value is strange
    def floor(self, x: EngineArg, ctx: Context) -> EngineRes:
        return None

    def lgamma(self, x: EngineArg, ctx: Context) -> EngineRes:
        if isinstance(x, Fraction):
            return None
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_eval(_gmp_lgamma, x, prec=prec, n=n)

    def log(self, x: EngineArg, ctx: Context) -> EngineRes:
        if isinstance(x, Fraction):
            return None
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_eval(gmp.log, x, prec=prec, n=n)

    def log10(self, x: EngineArg, ctx: Context) -> EngineRes:
        if isinstance(x, Fraction):
            return None
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_eval(gmp.log10, x, prec=prec, n=n)

    def log1p(self, x: EngineArg, ctx: Context) -> EngineRes:
        if isinstance(x, Fraction):
            return None
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_eval(gmp.log1p, x, prec=prec, n=n)

    def log2(self, x: EngineArg, ctx: Context) -> EngineRes:
        if isinstance(x, Fraction):
            return None
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_eval(gmp.log2, x, prec=prec, n=n)

    def neg(self, x: EngineArg, ctx: Context) -> EngineRes:
        if isinstance(x, Fraction):
            return None
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_eval(_gmp_neg, x, prec=prec, n=n)

    # we don't use MPFR's ceil/floor/trunc/roundint
    # since their return value is strange
    def roundint(self, x: EngineArg, ctx: Context) -> EngineRes:
        return None

    def sin(self, x: EngineArg, ctx: Context) -> EngineRes:
        if isinstance(x, Fraction):
            return None
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_eval(gmp.sin, x, prec=prec, n=n)

    def sinh(self, x: EngineArg, ctx: Context) -> EngineRes:
        if isinstance(x, Fraction):
            return None
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_eval(gmp.sinh, x, prec=prec, n=n)

    def sqrt(self, x: EngineArg, ctx: Context) -> EngineRes:
        if isinstance(x, Fraction):
            return None
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_eval(gmp.sqrt, x, prec=prec, n=n)

    def tan(self, x: EngineArg, ctx: Context) -> EngineRes:
        if isinstance(x, Fraction):
            return None
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_eval(gmp.tan, x, prec=prec, n=n)

    def tanh(self, x: EngineArg, ctx: Context) -> EngineRes:
        if isinstance(x, Fraction):
            return None
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_eval(gmp.tanh, x, prec=prec, n=n)

    def tgamma(self, x: EngineArg, ctx: Context) -> EngineRes:
        if isinstance(x, Fraction):
            return None
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_eval(gmp.gamma, x, prec=prec, n=n)

    # we don't use MPFR's ceil/floor/trunc/roundint
    # since their return value is strange
    def trunc(self, x: EngineArg, ctx: Context) -> EngineRes:
        return None

    def add(self, x: EngineArg, y: EngineArg, ctx: Context) -> EngineRes:
        if isinstance(x, Fraction) or isinstance(y, Fraction):
            return None
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_eval(gmp.add, x, y, prec=prec, n=n)

    def atan2(self, y: EngineArg, x: EngineArg, ctx: Context) -> EngineRes:
        if isinstance(y, Fraction) or isinstance(x, Fraction):
            return None
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_eval(gmp.atan2, y, x, prec=prec, n=n)

    def copysign(self, x: EngineArg, y: EngineArg, ctx: Context) -> EngineRes:
        if isinstance(x, Fraction) or isinstance(y, Fraction):
            return None
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_eval(gmp.copy_sign, x, y, prec=prec, n=n)

    def div(self, x: EngineArg, y: EngineArg, ctx: Context) -> EngineRes:
        if isinstance(x, Fraction) or isinstance(y, Fraction):
            return None
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_eval(gmp.div, x, y, prec=prec, n=n)

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
        if isinstance(x, Fraction) or isinstance(y, Fraction):
            return None
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_eval(gmp.fmod, x, y, prec=prec, n=n)

    def fmax(self, x: EngineArg, y: EngineArg, ctx: Context) -> EngineRes:
        if isinstance(x, Fraction) or isinstance(y, Fraction):
            return None
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_eval(gmp.maxnum, x, y, prec=prec, n=n)

    def fmin(self, x: EngineArg, y: EngineArg, ctx: Context) -> EngineRes:
        if isinstance(x, Fraction) or isinstance(y, Fraction):
            return None
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_eval(gmp.minnum, x, y, prec=prec, n=n)

    def hypot(self, x: EngineArg, y: EngineArg, ctx: Context) -> EngineRes:
        if isinstance(x, Fraction) or isinstance(y, Fraction):
            return None
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_eval(gmp.hypot, x, y, prec=prec, n=n)

    def _mod(self, x: Float, y: Float, ctx: Context) -> Float:
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
        return self._mod(x, y, ctx)

    def mul(self, x: EngineArg, y: EngineArg, ctx: Context) -> EngineRes:
        if isinstance(x, Fraction) or isinstance(y, Fraction):
            return None
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_eval(gmp.mul, x, y, prec=prec, n=n)

    def pow(self, x: EngineArg, y: EngineArg, ctx: Context) -> EngineRes:
        if isinstance(x, Fraction) or isinstance(y, Fraction):
            return None
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_eval(_gmp_pow, x, y, prec=prec, n=n)

    def remainder(self, x: EngineArg, y: EngineArg, ctx: Context) -> EngineRes:
        if isinstance(x, Fraction) or isinstance(y, Fraction):
            return None
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_eval(gmp.remainder, x, y, prec=prec, n=n)

    def sub(self, x: EngineArg, y: EngineArg, ctx: Context) -> EngineRes:
        if isinstance(x, Fraction) or isinstance(y, Fraction):
            return None
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_eval(gmp.sub, x, y, prec=prec, n=n)

    def fma(self, x: EngineArg, y: EngineArg, z: EngineArg, ctx: Context) -> EngineRes:
        if isinstance(x, Fraction) or isinstance(y, Fraction) or isinstance(z, Fraction):
            return None
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_eval(gmp.fma, x, y, z, prec=prec, n=n)

    def const_e(self, ctx: Context) -> EngineRes:
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_constant(_Constant.E, prec=prec, n=n)

    def const_log2e(self, ctx: Context) -> EngineRes:
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_constant(_Constant.LOG2E, prec=prec, n=n)

    def const_log10e(self, ctx: Context) -> EngineRes:
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_constant(_Constant.LOG10E, prec=prec, n=n)

    def const_ln2(self, ctx: Context) -> EngineRes:
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_constant(_Constant.LN2, prec=prec, n=n)

    def const_ln10(self, ctx: Context) -> EngineRes:
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_constant(_Constant.LN10, prec=prec, n=n)

    def const_pi(self, ctx: Context) -> EngineRes:
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_constant(_Constant.PI, prec=prec, n=n)

    def const_pi_2(self, ctx: Context) -> EngineRes:
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_constant(_Constant.PI_2, prec=prec, n=n)

    def const_pi_4(self, ctx: Context) -> EngineRes:
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_constant(_Constant.PI_4, prec=prec, n=n)

    def const_1_pi(self, ctx: Context) -> EngineRes:
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_constant(_Constant.M_1_PI, prec=prec, n=n)

    def const_2_pi(self, ctx: Context) -> EngineRes:
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_constant(_Constant.M_2_PI, prec=prec, n=n)

    def const_2_sqrtpi(self, ctx: Context) -> EngineRes:
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_constant(_Constant.M_2_SQRTPI, prec=prec, n=n)

    def const_sqrt2(self, ctx: Context) -> EngineRes:
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_constant(_Constant.SQRT2, prec=prec, n=n)

    def const_sqrt1_2(self, ctx: Context) -> EngineRes:
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_constant(_Constant.SQRT1_2, prec=prec, n=n)

