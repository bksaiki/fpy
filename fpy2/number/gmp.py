"""
Nicer interface to gmpy2 / MPFR.

The interface centers around round-to-odd,
a special rounding mode that ensures that re-rounding
at less precision is safe.
"""

import enum
import gmpy2 as gmp

from typing import Callable

from .number import RealFloat, Float
from ..utils import enum_repr

def _bool_to_sign(b: bool):
    return '-' if b else '+'

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

def _gmp_neg(x):
    return -x

def _gmp_abs(x):
    return abs(x)

def _gmp_pow(x, y):
    return x ** y

def _gmp_lgamma(x):
    y, _ = gmp.lgamma(x)
    return y


def float_to_mpfr(x: RealFloat | Float):
    """
    Converts `x` into an MPFR type exactly.
    """
    if isinstance(x, Float) and x.is_nar():
        if x.isnan:
            s = '-' if x.s else '+'
            return gmp.mpfr(f'{s}nan')
        else: # x.isinf
            s = '-' if x.s else '+'
            return gmp.mpfr(f'{s}inf')
    else:
        fmt = f'{_bool_to_sign(x.s)}{hex(x.c)}p{x.exp}'
        return gmp.mpfr(fmt, precision=x.p, base=16)

def mpfr_to_float(x):
    """
    Converts `x` into Float type exactly.

    The precision of the result is the same as the precision of `x`.
    """
    return _round_odd(x, False)


def _mpfr_call_with_prec(prec: int, fn: Callable[..., gmp.mpfr], args: tuple[gmp.mpfr]):
    """
    Calls an MPFR method `fn` with arguments `args` using `prec` digits
    of precision and round towards zero (RTZ).
    """
    with gmp.context(
        precision=prec,
        emin=gmp.get_emin_min(),
        emax=gmp.get_emax_max(),
        trap_underflow=False,
        trap_overflow=False,
        trap_inexact=False,
        trap_divzero=False,
        round=gmp.RoundToZero,
    ):
        return fn(*args)

def _mpfr_call(fn: Callable[..., gmp.mpfr], *args: gmp.mpfr, prec: int | None = None, n: int | None = None):
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

def mpfr_value(x, *, prec: int | None = None, n: int | None = None):
    """
    Converts `x` into an MPFR type such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_call(gmp.mpfr, x, prec=prec, n=n)

@enum_repr
class Constant(enum.Enum):
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


# From `titanfp` package
# TODO: some of these are unsafe
# TODO: should these be indexed by string or enum?
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

def mpfr_constant(x: Constant, *, prec: int | None = None, n: int | None = None):
    """
    Converts `x` into an MPFR type such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    try:
        fn = _constant_exprs[x]
        return _mpfr_call(fn, prec=prec, n=n)
    except KeyError as e:
        raise ValueError(f'unknown constant {e.args[0]!r}') from None

def _mpfr_eval(gmp_fn: Callable[..., gmp.mpfr], *args: Float, prec: int | None = None, n: int | None = None):
    """
    Evaluates `gmp_fn(*args)` such that the result may be safely re-rounded.
    Either specify:
    - `prec`: the number of digits, or
    - `n`: the first unrepresentable digit
    """
    if prec is not None and n is not None:
        raise ValueError('Either `prec` or `n` must be specified, not both')
    gmp_args = tuple(float_to_mpfr(x) for x in args)
    return _mpfr_call(gmp_fn, *gmp_args, prec=prec, n=n)

#####################################################################
# General operations

def mpfr_acos(x: Float, *, prec: int | None = None, n: int | None = None):
    """
    Computes `acos(x)` using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_eval(gmp.acos, x, prec=prec, n=n)

def mpfr_acosh(x: Float, *, prec: int | None = None, n: int | None = None):
    """
    Computes `acosh(x)` using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_eval(gmp.acosh, x, prec=prec, n=n)

def mpfr_add(x: Float, y: Float, *, prec: int | None = None, n: int | None = None):
    """
    Computes `x + y` using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_eval(gmp.add, x, y, prec=prec, n=n)

def mpfr_asin(x: Float, *, prec: int | None = None, n: int | None = None):
    """
    Computes `asin(x)` using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_eval(gmp.asin, x, prec=prec, n=n)

def mpfr_asinh(x: Float, *, prec: int | None = None, n: int | None = None):
    """
    Computes `asinh(x)` using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_eval(gmp.asinh, x, prec=prec, n=n)

def mpfr_atan(x: Float, *, prec: int | None = None, n: int | None = None):
    """
    Computes `atan(x)` using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_eval(gmp.atan, x, prec=prec, n=n)

def mpfr_atan2(y: Float, x: Float, *, prec: int | None = None, n: int | None = None):
    """
    Computes `atan2(y, x)` using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_eval(gmp.atan2, y, x, prec=prec, n=n)

def mpfr_atanh(x: Float, *, prec: int | None = None, n: int | None = None):
    """
    Computes `atanh(x)` using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_eval(gmp.atanh, x, prec=prec, n=n)

def mpfr_cbrt(x: Float, *, prec: int | None = None, n: int | None = None):
    """
    Computes `cbrt(x)` using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_eval(gmp.cbrt, x, prec=prec, n=n)

def mpfr_copysign(x: Float, y: Float, *, prec: int | None = None, n: int | None = None):
    """
    Returns `x` with the sign of `y` using MPFR such that it may be
    safely re-rounded accurately to `prec` digits of precision.
    """
    return _mpfr_eval(gmp.copy_sign, x, y, prec=prec, n=n)

def mpfr_cos(x: Float, *, prec: int | None = None, n: int | None = None):
    """
    Computes `cos(x)` using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_eval(gmp.cos, x, prec=prec, n=n)

def mpfr_cosh(x: Float, *, prec: int | None = None, n: int | None = None):
    """
    Computes `cosh(x)` using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_eval(gmp.cosh, x, prec=prec, n=n)

def mpfr_div(x: Float, y: Float, *, prec: int | None = None, n: int | None = None):
    """
    Computes `x / y` using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_eval(gmp.div, x, y, prec=prec, n=n)

def mpfr_erf(x: Float, *, prec: int | None = None, n: int | None = None):
    """
    Computes `erf(x)` using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_eval(gmp.erf, x, prec=prec, n=n)

def mpfr_erfc(x: Float, *, prec: int | None = None, n: int | None = None):
    """
    Computes `erfc(x)` using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_eval(gmp.erfc, x, prec=prec, n=n)

def mpfr_exp(x: Float, *, prec: int | None = None, n: int | None = None):
    """
    Computes `exp(x)` using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_eval(gmp.exp, x, prec=prec, n=n)

def mpfr_exp2(x: Float, *, prec: int | None = None, n: int | None = None):
    """
    Computes `2**x` using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_eval(gmp.exp2, x, prec=prec, n=n)

def mpfr_exp10(x: Float, *, prec: int | None = None, n: int | None = None):
    """
    Computes `10**x` using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_eval(gmp.exp10, x, prec=prec, n=n)

def mpfr_expm1(x: Float, *, prec: int | None = None, n: int | None = None):
    """
    Computes `exp(x) - 1` using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_eval(gmp.expm1, x, prec=prec, n=n)

def mpfr_fabs(x: Float, *, prec: int | None = None, n: int | None = None):
    """
    Computes `abs(x)` using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.

    This is the same as computing `abs(x)` exactly and
    then rounding the result to the desired.
    """
    return _mpfr_eval(_gmp_abs, x, prec=prec, n=n)

def mpfr_fdim(x: Float, y: Float, *, prec: int | None = None, n: int | None = None):
    """
    Computes `max(x - y, 0)` using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    if x.isnan or y.isnan:
        # C reference: if either argument is NaN, NaN is returned
        return Float(isnan=True)
    elif x > y:
        # if `x > y`, returns `x - y`
        return mpfr_sub(x, y, prec=prec, n=n)
    else:
        # otherwise, returns +0
        return Float()

def mpfr_fma(x: Float, y: Float, z: Float, *, prec: int | None = None, n: int | None = None):
    """
    Computes `x * y + z` using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_eval(gmp.fma, x, y, z, prec=prec, n=n)

def mpfr_fmod(x: Float, y: Float, *, prec: int | None = None, n: int | None = None):
    """
    Computes the remainder of `x / y`, where the remainder has
    the same sign as `x`, using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.

    The remainder is exactly `x - iquot * y`, where `iquot` is the
    `x / y` with its fractional part truncated.
    """
    return _mpfr_eval(gmp.fmod, x, y, prec=prec, n=n)

def mpfr_fmax(x: Float, y: Float, *, prec: int | None = None, n: int | None = None):
    """
    Computes `max(x, y)` using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.

    This is the same as computing `max(x, y)` exactly and
    then rounding the result to the desired precision.
    """
    return _mpfr_eval(gmp.maxnum, x, y, prec=prec, n=n)

def mpfr_fmin(x: Float, y: Float, *, prec: int | None = None, n: int | None = None):
    """
    Computes `min(x, y)` using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.

    This is the same as computing `min(x, y)` exactly and 
    then rounding the result to the desired precision.
    """
    return _mpfr_eval(gmp.minnum, x, y, prec=prec, n=n)

def mpfr_hypot(x: Float, y: Float, *, prec: int | None = None, n: int | None = None):
    """
    Computes `sqrt(x * x + y * y)` using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_eval(gmp.hypot, x, y, prec=prec, n=n)

def mpfr_lgamma(x: Float, *, prec: int | None = None, n: int | None = None):
    """
    Computes `lgamma(x)` using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_eval(_gmp_lgamma, x, prec=prec, n=n)

def mpfr_log(x: Float, *, prec: int | None = None, n: int | None = None):
    """
    Computes `ln(x)` using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_eval(gmp.log, x, prec=prec, n=n)

def mpfr_log10(x: Float, *, prec: int | None = None, n: int | None = None):
    """
    Computes `log10(x)` using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_eval(gmp.log10, x, prec=prec, n=n)

def mpfr_log1p(x: Float, *, prec: int | None = None, n: int | None = None):
    """
    Computes `log(1 + x)` using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_eval(gmp.log1p, x, prec=prec, n=n)

def mpfr_log2(x: Float, *, prec: int | None = None, n: int | None = None):
    """
    Computes `log2(x)` using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_eval(gmp.log2, x, prec=prec, n=n)

def mpfr_mul(x: Float, y: Float, *, prec: int | None = None, n: int | None = None):
    """
    Computes `x * y` using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_eval(gmp.mul, x, y, prec=prec, n=n)

def mpfr_neg(x: Float, *, prec: int | None = None, n: int | None = None):
    """
    Computes `-x` using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_eval(_gmp_neg, x, prec=prec, n=n)

def mpfr_pow(x: Float, y: Float, *, prec: int | None = None, n: int | None = None):
    """
    Computes `x ** y` using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_eval(_gmp_pow, x, y, prec=prec, n=n)

def mpfr_remainder(x: Float, y: Float, *, prec: int | None = None, n: int | None = None):
    """
    Computes the remainder of `x / y` using MPFR such that it may be
    safely re-rounded accurately to `prec` digits of precision.

    The remainder is exactly `x - quo * y`, where `quo` is the
    integral value nearest the exact value of `x / y`.
    """
    return _mpfr_eval(gmp.remainder, x, y, prec=prec, n=n)

def mpfr_sin(x: Float, *, prec: int | None = None, n: int | None = None):
    """
    Computes `sin(x)` using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_eval(gmp.sin, x, prec=prec, n=n)

def mpfr_sinh(x: Float, *, prec: int | None = None, n: int | None = None):
    """
    Computes `sinh(x)` using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_eval(gmp.sinh, x, prec=prec, n=n)

def mpfr_sqrt(x: Float, *, prec: int | None = None, n: int | None = None):
    """
    Computes `sqrt(x)` using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_eval(gmp.sqrt, x, prec=prec, n=n)

def mpfr_sub(x: Float, y: Float, *, prec: int | None = None, n: int | None = None):
    """
    Computes `x - y` using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_eval(gmp.sub, x, y, prec=prec, n=n)

def mpfr_tan(x: Float, *, prec: int | None = None, n: int | None = None):
    """
    Computes `tan(x)` using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_eval(gmp.tan, x, prec=prec, n=n)

def mpfr_tanh(x: Float, *, prec: int | None = None, n: int | None = None):
    """
    Computes `tanh(x)` using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_eval(gmp.tanh, x, prec=prec, n=n)

def mpfr_tgamma(x: Float, *, prec: int | None = None, n: int | None = None):
    """
    Computes `tgamma(x)` using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_eval(gmp.gamma, x, prec=prec, n=n)
