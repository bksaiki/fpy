"""
Nicer interface to gmpy2 / MPFR.

This interface provides conversions between
the `Float` type and the `gmpy2.mpfr` type,
as well as wrappers for round-to-odd arithmetic.
"""

import gmpy2 as gmp

from typing import Callable

from .number import RealFloat, Float

__all__ = [
    'MPFR_EMIN',
    'MPFR_EMAX',
    'float_to_mpfr',
    'mpfr_to_float',
    'mpfr_call',
    'mpfr_value',
]

###########################################################
# Limits

MPFR_EMIN = gmp.get_emin_min()
"""Maximum exponent for MPFR numbers."""

MPFR_EMAX = gmp.get_emax_max()
"""Minimum exponent for MPFR numbers."""

###########################################################
# Conversions between Float and gmp.mpfr

def float_to_mpfr(x: RealFloat | Float):
    """
    Converts `x` into an MPFR type exactly.
    """
    if isinstance(x, Float):
        if x.isnan:
            # drops sign bit
            return gmp.nan()
        elif x.isinf:
            return gmp.set_sign(gmp.inf(), x.s)

    s_fmt = '-' if x.s else '+'
    fmt = f'{s_fmt}{hex(x.c)}p{x.exp}'
    return gmp.mpfr(fmt, precision=x.p, base=16)

def mpfr_to_float(x):
    """
    Converts `x` into Float type exactly.

    The precision of the result is the same as the precision of `x`.
    """
    return _round_odd(x, False)

###########################################################
# MPFR round-to-odd arithmetic wrappers

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

def _mpfr_call_with_prec(prec: int, fn: Callable[..., gmp.mpfr], args: tuple[gmp.mpfr, ...]):
    """
    Calls an MPFR method `fn` with arguments `args` using `prec` digits
    of precision and round towards zero (RTZ).
    """
    with gmp.context(
        precision=prec,
        emin=MPFR_EMIN,
        emax=MPFR_EMAX,
        trap_underflow=False,
        trap_overflow=False,
        trap_inexact=False,
        trap_divzero=False,
        round=gmp.RoundToZero,
    ):
        return fn(*args)

def mpfr_call(
    fn: Callable[..., gmp.mpfr],
    args: tuple[gmp.mpfr, ...],
    prec: int | None = None,
    n: int | None = None
):
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
    return mpfr_call(gmp.mpfr, (x,), prec=prec, n=n)
