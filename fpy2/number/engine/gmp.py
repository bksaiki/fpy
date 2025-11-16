"""
MPFR-backed engine for round-to-odd arithmetic.

This engine wraps the gmpy2/MPFR library to provide
round-to-odd arithmetic operations.
"""

import gmpy2 as gmp
from typing import Callable

from ..context import Context
from ..number import Float, RealFloat
from ..gmp import Constant
from .engine import Engine


_MPFR_EMIN = gmp.get_emin_min()
_MPFR_EMAX = gmp.get_emax_max()


def _gmp_neg(x):
    return -x

def _gmp_abs(x):
    return abs(x)

def _gmp_pow(x, y):
    return x ** y

def _gmp_lgamma(x):
    y, _ = gmp.lgamma(x)
    return y


def _round_odd(x: gmp.mpfr, inexact: bool):
    """Applies the round-to-odd fix up."""
    s = x.is_signed()
    if x.is_nan():
        return Float(s=s, isnan=True)
    elif x.is_infinite():
        # check for inexactness => only occurs when MPFR overflows
        # TODO: awkward to use interval information for an infinity
        if inexact:
            interval_size = 0
            interval_down = not s
            interval_closed = False
            return Float(
                s=s,
                isinf=True,
                interval_size=interval_size,
                interval_down=interval_down,
                interval_closed=interval_closed
            )
        else:
             return Float(s=s, isinf=True)
    elif x.is_zero():
        # check for inexactness => only occurs when MPFR overflows
        # TODO: generate a reasonable inexact value
        if inexact:
            exp = gmp.get_emin_min() - 1
            return Float(s=s, exp=exp, c=1)
        else:
            return Float(s=s)
    else:
        # extract mantissa and exponent
        m_, exp_ = x.as_mantissa_exp()
        c = int(abs(m_))
        exp = int(exp_)

        # round to odd => sticky bit = last bit | inexact
        if c % 2 == 0 and inexact:
            c += 1
        return Float(s=s, c=c, exp=exp)


def _float_to_mpfr(x: RealFloat | Float):
    """Converts `x` into an MPFR type exactly."""
    if isinstance(x, Float):
        if x.isnan:
            # drops sign bit
            return gmp.nan()
        elif x.isinf:
            return gmp.set_sign(gmp.inf(), x.s)

    s_fmt = '-' if x.s else '+'
    fmt = f'{s_fmt}{hex(x.c)}p{x.exp}'
    return gmp.mpfr(fmt, precision=x.p, base=16)


def _mpfr_call_with_prec(prec: int, fn: Callable[..., gmp.mpfr], args: tuple[gmp.mpfr, ...]):
    """
    Calls an MPFR method `fn` with arguments `args` using `prec` digits
    of precision and round towards zero (RTZ).
    """
    with gmp.context(
        precision=prec,
        emin=_MPFR_EMIN,
        emax=_MPFR_EMAX,
        trap_underflow=False,
        trap_overflow=False,
        trap_inexact=False,
        trap_divzero=False,
        round=gmp.RoundToZero,
    ):
        return fn(*args)


def _mpfr_call(fn: Callable[..., gmp.mpfr], args: tuple[gmp.mpfr, ...], prec: int | None = None, n: int | None = None):
    """
    Evalutes `fn(args)` such that the result may be safely re-rounded.
    Either specify:
    - `prec`: the number of digits, or
    - `n`: the first unrepresentable digit
    """
    if prec is None:
        # computing to re-round safely up to the `n`th absolute digit
        if n is None:
            raise ValueError('Either `prec` or `n` must be specified')

        # compute with 2 digits of precision
        result = _mpfr_call_with_prec(2, fn, args)

        # special cases: NaN, Inf, or 0
        if result.is_nan() or result.is_infinite() or result.is_zero():
            return _round_odd(result, result.rc != 0)

        # extract the normalized exponent of `y`
        # gmp has a messed up definition of exponent
        e = gmp.get_exp(result) - 1

        # all digits are at or below the `n`th digit, so we can round safely
        # we at least have two digits of precision, so we can round safely
        if e <= n:
            return _round_odd(result, result.rc != 0)

        # need to re-compute with the correct precision
        # `e - n`` are the number of digits above the `n`th digit
        # add two digits for the rounding bits
        prec = e - n
        result = _mpfr_call_with_prec(prec + 2, fn, args)
        return _round_odd(result, result.rc != 0)
    else:
        # computing to re-round safely to `prec` digits
        # if `n` is set, we ignore it since having too much precision is okay
        result = _mpfr_call_with_prec(prec + 2, fn, args)
        return _round_odd(result, result.rc != 0)


def _mpfr_eval(gmp_fn: Callable[..., gmp.mpfr], *args: Float, prec: int | None = None, n: int | None = None):
    """
    Evaluates `gmp_fn(*args)` such that the result may be safely re-rounded.
    Either specify:
    - `prec`: the number of digits, or
    - `n`: the first unrepresentable digit
    """
    if prec is not None and n is not None:
        raise ValueError('Either `prec` or `n` must be specified, not both')
    gmp_args = tuple(_float_to_mpfr(x) for x in args)
    return _mpfr_call(gmp_fn, gmp_args, prec=prec, n=n)


# From `titanfp` package
# TODO: some of these are unsafe
_constant_exprs: dict[Constant, Callable[[], gmp.mpfr]] = {
    Constant.E : lambda : gmp.exp(1),
    Constant.LOG2E : lambda: gmp.log2(gmp.exp(1)), # TODO: may be inaccurate
    Constant.LOG10E : lambda: gmp.log10(gmp.exp(1)), # TODO: may be inaccurate
    Constant.LN2 : gmp.const_log2,
    Constant.LN10 : lambda: gmp.log(10),
    Constant.PI : gmp.const_pi,
    Constant.PI_2 : lambda: gmp.const_pi() / 2, # division by 2 is exact
    Constant.PI_4 : lambda: gmp.const_pi() / 4, # division by 4 is exact
    Constant.M_1_PI : lambda: 1 / gmp.const_pi(), # TODO: may be inaccurate
    Constant.M_2_PI : lambda: 2 / gmp.const_pi(), # TODO: may be inaccurate
    Constant.M_2_SQRTPI : lambda: 2 / gmp.sqrt(gmp.const_pi()), # TODO: may be inaccurate
    Constant.SQRT2: lambda: gmp.sqrt(2),
    Constant.SQRT1_2: lambda: gmp.sqrt(gmp.div(gmp.mpfr(1), gmp.mpfr(2))),
}


def _mpfr_constant(x: Constant, *, prec: int | None = None, n: int | None = None):
    """
    Computes constant `x` such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    try:
        fn = _constant_exprs[x]
        return _mpfr_call(fn, (), prec=prec, n=n)
    except KeyError as e:
        raise ValueError(f'unknown constant {e.args[0]!r}') from None



class MPFREngine(Engine):
    """
    Engine that uses MPFR (via gmpy2) for round-to-odd arithmetic.
    
    This engine can handle all operations except exact arithmetic (when both prec and n are None).
    """

    def acos(self, x: Float, ctx: Context) -> Float | None:
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_eval(gmp.acos, x, prec=prec, n=n)

    def acosh(self, x: Float, ctx: Context) -> Float | None:
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_eval(gmp.acosh, x, prec=prec, n=n)

    def asin(self, x: Float, ctx: Context) -> Float | None:
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_eval(gmp.asin, x, prec=prec, n=n)

    def asinh(self, x: Float, ctx: Context) -> Float | None:
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_eval(gmp.asinh, x, prec=prec, n=n)

    def atan(self, x: Float, ctx: Context) -> Float | None:
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_eval(gmp.atan, x, prec=prec, n=n)

    def atanh(self, x: Float, ctx: Context) -> Float | None:
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_eval(gmp.atanh, x, prec=prec, n=n)

    def cbrt(self, x: Float, ctx: Context) -> Float | None:
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_eval(gmp.cbrt, x, prec=prec, n=n)

    def cos(self, x: Float, ctx: Context) -> Float | None:
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_eval(gmp.cos, x, prec=prec, n=n)

    def cosh(self, x: Float, ctx: Context) -> Float | None:
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_eval(gmp.cosh, x, prec=prec, n=n)

    def erf(self, x: Float, ctx: Context) -> Float | None:
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_eval(gmp.erf, x, prec=prec, n=n)

    def erfc(self, x: Float, ctx: Context) -> Float | None:
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_eval(gmp.erfc, x, prec=prec, n=n)

    def exp(self, x: Float, ctx: Context) -> Float | None:
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_eval(gmp.exp, x, prec=prec, n=n)

    def exp2(self, x: Float, ctx: Context) -> Float | None:
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_eval(gmp.exp2, x, prec=prec, n=n)

    def exp10(self, x: Float, ctx: Context) -> Float | None:
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_eval(gmp.exp10, x, prec=prec, n=n)

    def expm1(self, x: Float, ctx: Context) -> Float | None:
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_eval(gmp.expm1, x, prec=prec, n=n)

    def fabs(self, x: Float, ctx: Context) -> Float | None:
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_eval(_gmp_abs, x, prec=prec, n=n)

    def lgamma(self, x: Float, ctx: Context) -> Float | None:
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_eval(_gmp_lgamma, x, prec=prec, n=n)

    def log(self, x: Float, ctx: Context) -> Float | None:
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_eval(gmp.log, x, prec=prec, n=n)

    def log10(self, x: Float, ctx: Context) -> Float | None:
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_eval(gmp.log10, x, prec=prec, n=n)

    def log1p(self, x: Float, ctx: Context) -> Float | None:
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_eval(gmp.log1p, x, prec=prec, n=n)

    def log2(self, x: Float, ctx: Context) -> Float | None:
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_eval(gmp.log2, x, prec=prec, n=n)

    def neg(self, x: Float, ctx: Context) -> Float | None:
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_eval(_gmp_neg, x, prec=prec, n=n)

    def sin(self, x: Float, ctx: Context) -> Float | None:
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_eval(gmp.sin, x, prec=prec, n=n)

    def sinh(self, x: Float, ctx: Context) -> Float | None:
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_eval(gmp.sinh, x, prec=prec, n=n)

    def sqrt(self, x: Float, ctx: Context) -> Float | None:
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_eval(gmp.sqrt, x, prec=prec, n=n)

    def tan(self, x: Float, ctx: Context) -> Float | None:
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_eval(gmp.tan, x, prec=prec, n=n)

    def tanh(self, x: Float, ctx: Context) -> Float | None:
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_eval(gmp.tanh, x, prec=prec, n=n)

    def tgamma(self, x: Float, ctx: Context) -> Float | None:
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_eval(gmp.gamma, x, prec=prec, n=n)

    def add(self, x: Float, y: Float, ctx: Context) -> Float | None:
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_eval(gmp.add, x, y, prec=prec, n=n)

    def atan2(self, y: Float, x: Float, ctx: Context) -> Float | None:
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_eval(gmp.atan2, y, x, prec=prec, n=n)

    def copysign(self, x: Float, y: Float, ctx: Context) -> Float | None:
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_eval(gmp.copy_sign, x, y, prec=prec, n=n)

    def div(self, x: Float, y: Float, ctx: Context) -> Float | None:
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_eval(gmp.div, x, y, prec=prec, n=n)

    def fdim(self, x: Float, y: Float, ctx: Context) -> Float | None:
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        if x.isnan or y.isnan:
            # C reference: if either argument is NaN, NaN is returned
            return Float(isnan=True)
        elif x > y:
            # if `x > y`, returns `x - y`
            return _mpfr_eval(gmp.sub, x, y, prec=prec, n=n)
        else:
            # otherwise, returns +0
            return Float()

    def fmod(self, x: Float, y: Float, ctx: Context) -> Float | None:
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_eval(gmp.fmod, x, y, prec=prec, n=n)

    def fmax(self, x: Float, y: Float, ctx: Context) -> Float | None:
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_eval(gmp.maxnum, x, y, prec=prec, n=n)

    def fmin(self, x: Float, y: Float, ctx: Context) -> Float | None:
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_eval(gmp.minnum, x, y, prec=prec, n=n)

    def hypot(self, x: Float, y: Float, ctx: Context) -> Float | None:
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_eval(gmp.hypot, x, y, prec=prec, n=n)

    def mod(self, x: Float, y: Float, ctx: Context) -> Float | None:
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
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
            import math
            
            # step 1. compute `floor(x / y)`
            q = math.floor(_mpfr_eval(gmp.div, x, y, n=-1))

            # step 2. compute `x - q * y`
            return x - q * y

    def mul(self, x: Float, y: Float, ctx: Context) -> Float | None:
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_eval(gmp.mul, x, y, prec=prec, n=n)

    def pow(self, x: Float, y: Float, ctx: Context) -> Float | None:
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_eval(_gmp_pow, x, y, prec=prec, n=n)

    def remainder(self, x: Float, y: Float, ctx: Context) -> Float | None:
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_eval(gmp.remainder, x, y, prec=prec, n=n)

    def sub(self, x: Float, y: Float, ctx: Context) -> Float | None:
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_eval(gmp.sub, x, y, prec=prec, n=n)

    def fma(self, x: Float, y: Float, z: Float, ctx: Context) -> Float | None:
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_eval(gmp.fma, x, y, z, prec=prec, n=n)

    def const_e(self, ctx: Context) -> Float | None:
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_constant(Constant.E, prec=prec, n=n)

    def const_log2e(self, ctx: Context) -> Float | None:
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_constant(Constant.LOG2E, prec=prec, n=n)

    def const_log10e(self, ctx: Context) -> Float | None:
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_constant(Constant.LOG10E, prec=prec, n=n)

    def const_ln2(self, ctx: Context) -> Float | None:
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_constant(Constant.LN2, prec=prec, n=n)

    def const_ln10(self, ctx: Context) -> Float | None:
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_constant(Constant.LN10, prec=prec, n=n)

    def const_pi(self, ctx: Context) -> Float | None:
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_constant(Constant.PI, prec=prec, n=n)

    def const_pi_2(self, ctx: Context) -> Float | None:
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_constant(Constant.PI_2, prec=prec, n=n)

    def const_pi_4(self, ctx: Context) -> Float | None:
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_constant(Constant.PI_4, prec=prec, n=n)

    def const_1_pi(self, ctx: Context) -> Float | None:
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_constant(Constant.M_1_PI, prec=prec, n=n)

    def const_2_pi(self, ctx: Context) -> Float | None:
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_constant(Constant.M_2_PI, prec=prec, n=n)

    def const_2_sqrtpi(self, ctx: Context) -> Float | None:
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_constant(Constant.M_2_SQRTPI, prec=prec, n=n)

    def const_sqrt2(self, ctx: Context) -> Float | None:
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_constant(Constant.SQRT2, prec=prec, n=n)

    def const_sqrt1_2(self, ctx: Context) -> Float | None:
        prec, n = ctx.round_params()
        if prec is None and n is None:
            return None
        return _mpfr_constant(Constant.SQRT1_2, prec=prec, n=n)

