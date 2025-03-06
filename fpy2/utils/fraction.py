"""
Helper methods for fractions.
"""

import re

from fractions import Fraction

_HEXNUM_PATTERN = re.compile(r'([+-])?0x(([0-9a-f]+)(\.([0-9a-f]+))?|\.[0-9a-f]+)(p([-+]?[0-9]+))?')

def digits_to_fraction(m: int, e: int, b: int):
    """Converts a mantissa, exponent, and base to a fraction."""
    return Fraction(m) * Fraction(b) ** e

def hexnum_to_fraction(s: str):
    """
    Converts a hexadecimal number to a fraction.

    Works for both integers and floating-point.
    """
    m = re.match(_HEXNUM_PATTERN, s)
    if not m:
        raise ValueError(f'invalid hexadecimal number: {s}')

    # all relevant components
    s = m.group(1)
    i = m.group(3)
    f = m.group(5)
    e = m.group(7)

    # sign (optional)
    if s is not None and s == '-':
        sign = -1
    else:
        sign = +1

    # integer component (required)
    ipart = int(i, 16)

    # fraction (optional)
    if f is not None:
        fpart = int(f, 16)
        efrac = -len(f)
    else:
        fpart = 1
        efrac = 0

    # exponent (optional)
    if e is not None:
        exp = int(e)
    else:
        exp = 0

    # combine the parts
    return sign * ipart * (fpart * Fraction(16) ** efrac) * (Fraction(16) ** exp)
