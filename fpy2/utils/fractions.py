"""
Helper methods for fractions.
"""

import re

from fractions import Fraction
from typing import Optional

_DECIMAL_PATTERN = re.compile(r'([+-])?([0-9]+)(\.([0-9]+))?([eE]([-+]?[0-9]+))?')
_HEXNUM_PATTERN = re.compile(r'([+-])?0x(([0-9a-f]+)(\.([0-9a-f]+))?|\.[0-9a-f]+)(p([-+]?[0-9]+))?')

def fraction(numerator: int, denominator: int):
    """Creates a fraction from a numerator and denominator."""
    if not isinstance(numerator, int):
        raise TypeError(f'Expected \'int\', got \'{type(numerator)}\' for numerator={numerator}')
    if not isinstance(denominator, int):
        raise TypeError(f'Expected \'int\', got \'{type(denominator)}\' for denominator={denominator}')
    if denominator == 0:
        raise ZeroDivisionError('denominator cannot be zero')
    return Fraction(numerator, denominator)

def digits_to_fraction(m: int, e: int, b: int):
    """Converts a mantissa, exponent, and base to a fraction."""
    if not isinstance(m, int):
        raise TypeError(f'Expected \'int\', got \'{type(m)}\' for m={m}')
    if not isinstance(e, int):
        raise TypeError(f'Expected \'int\', got \'{type(e)}\' for e={e}')
    if not isinstance(b, int):
        raise TypeError(f'Expected \'int\', got \'{type(b)}\' for b={b}')
    return Fraction(m) * Fraction(b) ** e

def _sci_to_fraction(
    s: Optional[str],
    i: str,
    f: Optional[str],
    e: Optional[str],
    b: int
) -> Fraction:
    """
    Converts a number in base `b` to a fraction.

    :param s: sign (as a string)
    :param i: integer part (as a string)
    :param f: fraction part (as a string)
    :param e: exponent part (as a string)
    :param b: base
    """
    assert b >= 2, f'base must be >= 2: b={b}'

    # sign (optional)
    if s is not None and s == '-':
        sign = -1
    else:
        sign = +1

    # integer component (required)
    ipart = int(i, b)

    # fraction (optional)
    if f is not None:
        fpart = int(f, b)
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
    x = ipart * Fraction(b) ** exp
    x *= fpart * Fraction(b) ** (exp + efrac)
    x *= sign
    return x


def decnum_to_fraction(s: str):
    """
    Converts a decimal number to a fraction.

    Works for both integers and floating-point.
    """
    if not isinstance(s, str):
        raise TypeError(f'Expected \'str\', got \'{type(s)}\' for s={s}')

    m = re.match(_DECIMAL_PATTERN, s)
    if not m:
        raise ValueError(f'invalid decimal number: {s}')

    # all relevant components
    s = m.group(1)
    i = m.group(2)
    f = m.group(4)
    e = m.group(6)

    return _sci_to_fraction(s, i, f, e, 10)


def hexnum_to_fraction(s: str):
    """
    Converts a hexadecimal number to a fraction.

    Works for both integers and floating-point.
    """
    if not isinstance(s, str):
        raise TypeError(f'Expected \'str\', got \'{type(s)}\' for s={s}')

    m = re.match(_HEXNUM_PATTERN, s)
    if not m:
        raise ValueError(f'invalid hexadecimal number: {s}')

    # all relevant components
    s = m.group(1)
    i = m.group(3)
    f = m.group(5)
    e = m.group(7)

    return _sci_to_fraction(s, i, f, e, 16)
