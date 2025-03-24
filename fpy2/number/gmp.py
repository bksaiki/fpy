"""
This module defines a nicer interface to the gmpy2 / MPFR.
"""

import gmpy2 as gmp

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


def from_mpfr(x, prec: int):
    """
    Converts `x` into an MPFR type such that it may be re-rounded
    accurately to `prec` digits of precision.

    Safe re-rounding is guaranteed by employing round-to-odd with
    two extra digits of precision.
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
