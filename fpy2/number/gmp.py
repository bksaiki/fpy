"""
Nicer interface to gmpy2 / MPFR.

The interface centers around round-to-odd,
a special rounding mode that ensures that re-rounding
at less precision is safe.
"""

import gmpy2 as gmp

from typing import Any, Callable

from .number import Float
from .real import RealFloat

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


def mpfr_value(x, prec: int):
    """
    Converts `x` into an MPFR type such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    with gmp.context(
        precision=prec+2,
        emin=gmp.get_emin_min(),
        emax=gmp.get_emax_max(),
        trap_underflow=False,
        trap_overflow=False,
        trap_inexact=False,
        trap_divzero=False,
        round=gmp.RoundToZero,
    ):
        y = gmp.mpfr(x)
        return _round_odd(y, y.rc != 0)

# From `titanfp` package
# TODO: some of these are unsafe
# TODO: should these be indexed by string or enum?
_constant_exprs = {
    'E' : lambda : gmp.exp(1),
    'LOG2E' : lambda: gmp.log2(gmp.exp(1)), # TODO: may be inaccurate
    'LOG10E' : lambda: gmp.log10(gmp.exp(1)), # TODO: may be inaccurate
    'LN2' : gmp.const_log2,
    'LN10' : lambda: gmp.log(10),
    'PI' : gmp.const_pi,
    'PI_2' : lambda: gmp.const_pi() / 2, # division by 2 is exact
    'PI_4' : lambda: gmp.const_pi() / 4, # division by 4 is exact
    'M_1_PI' : lambda: 1 / gmp.const_pi(), # TODO: may be inaccurate
    'M_2_PI' : lambda: 2 / gmp.const_pi(), # TODO: may be inaccurate
    'M_2_SQRTPI' : lambda: 2 / gmp.sqrt(gmp.const_pi()), # TODO: may be inaccurate
    'SQRT2': lambda: gmp.sqrt(2),
    'SQRT1_2': lambda: gmp.sqrt(gmp.div(gmp.mpfr(1), gmp.mpfr(2))),
    'INFINITY': gmp.inf,
    'NAN': gmp.nan,
}

def mpfr_constant(x: str, prec: int):
    """
    Converts `x` into an MPFR type such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    if not isinstance(x, str):
        raise TypeError(f'Expected a string, got {type(x)}')

    with gmp.context(
        precision=prec+2,
        emin=gmp.get_emin_min(),
        emax=gmp.get_emax_max(),
        trap_underflow=False,
        trap_overflow=False,
        trap_inexact=False,
        trap_divzero=False,
        round=gmp.RoundToZero,
    ):
        try:
            y = _constant_exprs[x]()
            return _round_odd(y, y.rc != 0)
        except KeyError as e:
            raise ValueError(f'unknown constant {e.args[0]!r}') from None

def _mpfr_1ary(gmp_fn: Callable[[Any], Any], x: Float, prec: int):
    xf = float_to_mpfr(x)
    with gmp.context(
        precision=prec+2,
        emin=gmp.get_emin_min(),
        emax=gmp.get_emax_max(),
        trap_underflow=False,
        trap_overflow=False,
        trap_inexact=False,
        trap_divzero=False,
        round=gmp.RoundToZero,
    ):
        r = gmp_fn(xf)
        return _round_odd(r, r.rc != 0)

def _mpfr_2ary(gmp_fn: Callable[[Any, Any], Any], x: Float, y: Float, prec: int):
    """Applies a 2-argument MPFR function with expected ternary."""
    xf = float_to_mpfr(x)
    yf = float_to_mpfr(y)
    with gmp.context(
        precision=prec+2,
        emin=gmp.get_emin_min(),
        emax=gmp.get_emax_max(),
        trap_underflow=False,
        trap_overflow=False,
        trap_inexact=False,
        trap_divzero=False,
        round=gmp.RoundToZero,
    ):
        r = gmp_fn(xf, yf)
        return _round_odd(r, r.rc != 0)

def _mpfr_3ary(gmp_fn: Callable[[Any, Any, Any], Any], x: Float, y: Float, z: Float, prec: int):
    """Applies a 3-argument MPFR function with expected ternary."""
    xf = float_to_mpfr(x)
    yf = float_to_mpfr(y)
    zf = float_to_mpfr(z)
    with gmp.context(
        precision=prec+2,
        emin=gmp.get_emin_min(),
        emax=gmp.get_emax_max(),
        trap_underflow=True,
        trap_overflow=True,
        trap_inexact=False,
        trap_divzero=False,
        round=gmp.RoundToZero,
    ):
        r = gmp_fn(xf, yf, zf)
        return _round_odd(r, r.rc != 0)

#####################################################################
# General operations

def mpfr_acos(x: Float, prec: int):
    """
    Computes `acos(x)` using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_1ary(gmp.acos, x, prec)

def mpfr_acosh(x: Float, prec: int):
    """
    Computes `acosh(x)` using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_1ary(gmp.acosh, x, prec)

def mpfr_add(x: Float, y: Float, prec: int):
    """
    Computes `x + y` using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_2ary(gmp.add, x, y, prec)

def mpfr_asin(x: Float, prec: int):
    """
    Computes `asin(x)` using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_1ary(gmp.asin, x, prec)

def mpfr_asinh(x: Float, prec: int):
    """
    Computes `asinh(x)` using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_1ary(gmp.asinh, x, prec)

def mpfr_atan(x: Float, prec: int):
    """
    Computes `atan(x)` using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_1ary(gmp.atan, x, prec)

def mpfr_atan2(y: Float, x: Float, prec: int):
    """
    Computes `atan2(y, x)` using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_2ary(gmp.atan2, y, x, prec)

def mpfr_atanh(x: Float, prec: int):
    """
    Computes `atanh(x)` using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_1ary(gmp.atanh, x, prec)

def mpfr_cbrt(x: Float, prec: int):
    """
    Computes `cbrt(x)` using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_1ary(gmp.cbrt, x, prec)

def mpfr_copysign(x: Float, y: Float, prec: int):
    """
    Returns `x` with the sign of `y` using MPFR such that it may be
    safely re-rounded accurately to `prec` digits of precision.
    """
    return _mpfr_2ary(gmp.copy_sign, x, y, prec)

def mpfr_cos(x: Float, prec: int):
    """
    Computes `cos(x)` using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_1ary(gmp.cos, x, prec)

def mpfr_cosh(x: Float, prec: int):
    """
    Computes `cosh(x)` using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_1ary(gmp.cosh, x, prec)

def mpfr_div(x: Float, y: Float, prec: int):
    """
    Computes `x / y` using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_2ary(gmp.div, x, y, prec)

def mpfr_erf(x: Float, prec: int):
    """
    Computes `erf(x)` using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_1ary(gmp.erf, x, prec)

def mpfr_erfc(x: Float, prec: int):
    """
    Computes `erfc(x)` using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_1ary(gmp.erfc, x, prec)

def mpfr_exp(x: Float, prec: int):
    """
    Computes `exp(x)` using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_1ary(gmp.exp, x, prec)

def mpfr_exp2(x: Float, prec: int):
    """
    Computes `2**x` using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_1ary(gmp.exp2, x, prec)

def mpfr_exp10(x: Float, prec: int):
    """
    Computes `10**x` using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_1ary(gmp.exp10, x, prec)

def mpfr_expm1(x: Float, prec: int):
    """
    Computes `exp(x) - 1` using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_1ary(gmp.expm1, x, prec)

def mpfr_fabs(x: Float, prec: int):
    """
    Computes `abs(x)` using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.

    This is the same as computing `abs(x)` exactly and
    then rounding the result to the desired.
    """
    return _mpfr_1ary(_gmp_abs, x, prec)

def mpfr_fdim(x: Float, y: Float, prec: int):
    """
    Computes `max(x - y, 0)` using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    if x.isnan or y.isnan:
        # C reference: if either argument is NaN, NaN is returned
        return Float(isnan=True)
    elif x > y:
        # if `x > y`, returns `x - y`
        return mpfr_sub(x, y, prec)
    else:
        # otherwise, returns +0
        return Float()

def mpfr_fma(x: Float, y: Float, z: Float, prec: int):
    """
    Computes `x * y + z` using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_3ary(gmp.fma, x, y, z, prec)

def mpfr_fmod(x: Float, y: Float, prec: int):
    """
    Computes the remainder of `x / y`, where the remainder has
    the same sign as `x`, using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.

    The remainder is exactly `x - iquot * y`, where `iquot` is the
    `x / y` with its fractional part truncated.
    """
    return _mpfr_2ary(gmp.fmod, x, y, prec)

def mpfr_fmax(x: Float, y: Float, prec: int):
    """
    Computes `max(x, y)` using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.

    This is the same as computing `max(x, y)` exactly and
    then rounding the result to the desired precision.
    """
    return _mpfr_2ary(gmp.maxnum, x, y, prec)

def mpfr_fmin(x: Float, y: Float, prec: int):
    """
    Computes `min(x, y)` using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.

    This is the same as computing `max(x, y)` exactly and 
    then rounding the result to the desired precision.
    """
    return _mpfr_2ary(gmp.minnum, x, y, prec)

def mpfr_hypot(x: Float, y: Float, prec: int):
    """
    Computes `sqrt(x * x + y * y)` using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_2ary(gmp.hypot, x, y, prec)

def mpfr_lgamma(x: Float, prec: int):
    """
    Computes `lgamma(x)` using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_1ary(_gmp_lgamma, x, prec)

def mpfr_log(x: Float, prec: int):
    """
    Computes `ln(x)` using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_1ary(gmp.log, x, prec)

def mpfr_log10(x: Float, prec: int):
    """
    Computes `log10(x)` using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_1ary(gmp.log10, x, prec)

def mpfr_log1p(x: Float, prec: int):
    """
    Computes `log(1 + x)` using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_1ary(gmp.log1p, x, prec)

def mpfr_log2(x: Float, prec: int):
    """
    Computes `log2(x)` using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_1ary(gmp.log2, x, prec)

def mpfr_mul(x: Float, y: Float, prec: int):
    """
    Computes `x * y` using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_2ary(gmp.mul, x, y, prec)

def mpfr_neg(x: Float, prec: int):
    """
    Computes `-x` using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_1ary(_gmp_neg, x, prec)

def mpfr_pow(x: Float, y: Float, prec: int):
    """
    Computes `x ** y` using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_2ary(_gmp_pow, x, y, prec)

def mpfr_remainder(x: Float, y: Float, prec: int):
    """
    Computes the remainder of `x / y` using MPFR such that it may be
    safely re-rounded accurately to `prec` digits of precision.

    The remainder is exactly `x - quo * y`, where `quo` is the
    integral value nearest the exact value of `x / y`.
    """
    return _mpfr_2ary(gmp.remainder, x, y, prec)

def mpfr_sin(x: Float, prec: int):
    """
    Computes `sin(x)` using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_1ary(gmp.sin, x, prec)

def mpfr_sinh(x: Float, prec: int):
    """
    Computes `sinh(x)` using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_1ary(gmp.sinh, x, prec)

def mpfr_sqrt(x: Float, prec: int):
    """
    Computes `sqrt(x)` using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_1ary(gmp.sqrt, x, prec)

def mpfr_sub(x: Float, y: Float, prec: int):
    """
    Computes `x - y` using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_2ary(gmp.sub, x, y, prec)

def mpfr_tan(x: Float, prec: int):
    """
    Computes `tan(x)` using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_1ary(gmp.tan, x, prec)

def mpfr_tanh(x: Float, prec: int):
    """
    Computes `tanh(x)` using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_1ary(gmp.tanh, x, prec)

def mpfr_tgamma(x: Float, prec: int):
    """
    Computes `tgamma(x)` using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_1ary(gmp.gamma, x, prec)
