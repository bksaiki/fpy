"""
Nicer interface to gmpy2 / MPFR.

The interface centers around round-to-odd,
a special rounding mode that ensures that re-rounding
at less precision is safe.
"""

import gmpy2 as gmp

from typing import Any, Callable

from .float import Float



def _round_odd(x: gmp.mpfr, inexact: bool):
    """Applies the round-to-odd fix up."""
    s = x.is_signed()
    if x.is_nan():
        return Float(s=s, isnan=True)
    elif x.is_infinite():
        return Float(s=s, isinf=True)
    elif x.is_zero():
        if inexact:
            raise ValueError(f'zero is unexpectedly inexact x={x}, inexact={inexact}')
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

def _bool_to_sign(b: bool):
    return '-' if b else '+'

def _to_mpfr(x: Float):
    fmt = f'{_bool_to_sign(x.s)}{hex(x.c)}p{x.exp}'
    return gmp.mpfr(fmt, precision=x.p, base=16)

def mpfr_constant(x, prec: int):
    """
    Converts `x` into an MPFR type such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
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
        y = gmp.mpfr(x)
        return _round_odd(y, y.rc != 0)

def _mpfr_1ary(gmp_fn: Callable[[Any], Any], x: Float, prec: int):
    xf = _to_mpfr(x)
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
        r = gmp_fn(xf)
        return _round_odd(r, r.rc != 0)

def _mpfr_2ary(gmp_fn: Callable[[Any, Any], Any], x: Float, y: Float, prec: int):
    """Applies a 2-argument MPFR function with expected ternary."""
    xf = _to_mpfr(x)
    yf = _to_mpfr(y)
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
        r = gmp_fn(xf, yf)
        return _round_odd(r, r.rc != 0)

def mpfr_add(x: Float, y: Float, prec: int):
    """
    Adds two Floats using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_2ary(gmp.add, x, y, prec)

def mpfr_sub(x: Float, y: Float, prec: int):
    """
    Subtracts two Floats using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_2ary(gmp.sub, x, y, prec)

def mpfr_mul(x: Float, y: Float, prec: int):
    """
    Multiplies two Floats using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_2ary(gmp.mul, x, y, prec)

def mpfr_div(x: Float, y: Float, prec: int):
    """
    Divides two Floats using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_2ary(gmp.div, x, y, prec)

def mpfr_sqrt(x: Float, prec: int):
    """
    Computes the square root of a Float using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_1ary(gmp.sqrt, x, prec)

def mpfr_exp(x: Float, prec: int):
    """
    Computes the exponential of a Float using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_1ary(gmp.exp, x, prec)

def mpfr_log(x: Float, prec: int):
    """
    Computes the natural logarithm of a Float using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_1ary(gmp.log, x, prec)

def mpfr_pow(x: Float, y: Float, prec: int):
    """
    Computes `x ** y` using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_2ary(gmp.pow, x, y, prec)

def mpfr_sin(x: Float, prec: int):
    """
    Computes the sine of a Float using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_1ary(gmp.sin, x, prec)

def mpfr_cos(x: Float, prec: int):
    """
    Computes the cosine of a Float using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_1ary(gmp.cos, x, prec)

def mpfr_tan(x: Float, prec: int):
    """
    Computes the tangent of a Float using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_1ary(gmp.tan, x, prec)

def mpfr_asin(x: Float, prec: int):
    """
    Computes the arcsine of a Float using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_1ary(gmp.asin, x, prec)

def mpfr_acos(x: Float, prec: int):
    """
    Computes the arccosine of a Float using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_1ary(gmp.acos, x, prec)

def mpfr_atan(x: Float, prec: int):
    """
    Computes the arctangent of a Float using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_1ary(gmp.atan, x, prec)

def mpfr_sinh(x: Float, prec: int):
    """
    Computes the hyperbolic sine of a Float using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_1ary(gmp.sinh, x, prec)

def mpfr_cosh(x: Float, prec: int):
    """
    Computes the hyperbolic cosine of a Float using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_1ary(gmp.cosh, x, prec)

def mpfr_tanh(x: Float, prec: int):
    """
    Computes the hyperbolic tangent of a Float using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_1ary(gmp.tanh, x, prec)

def mpfr_asinh(x: Float, prec: int):
    """
    Computes the hyperbolic arcsine of a Float using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_1ary(gmp.asinh, x, prec)

def mpfr_acosh(x: Float, prec: int):
    """
    Computes the hyperbolic arccosine of a Float using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_1ary(gmp.acosh, x, prec)

def mpfr_atanh(x: Float, prec: int):
    """
    Computes the hyperbolic arctangent of a Float using MPFR such that it may be safely re-rounded
    accurately to `prec` digits of precision.
    """
    return _mpfr_1ary(gmp.atanh, x, prec)
