"""
Helper methods for fractions.
"""

import numbers
import re

from fractions import Fraction

from .bits import is_power_of_two

_DECIMAL_PATTERN = re.compile(r'([-+])?([0-9]+(\.[0-9]+)?|\.[0-9]+)(e([-+]?[0-9]+))?')
_HEXNUM_PATTERN = re.compile(r'([-+])?0x([0-9a-f]+(\.[0-9a-f]+)?|\.[0-9a-f]+)(p([-+]?[0-9]+))?')

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
    s: str | None,
    i: str,
    f: str | None,
    e: str | None,
    base: int,
    b: int,
) -> Fraction:
    """
    Converts a number in base `b` to a fraction.

    :param s: sign (as a string)
    :param i: integer part (as a string)
    :param f: fraction part (as a string)
    :param e: exponent part (as a string)
    :param b: base
    """
    assert base >= 2, f'base must be >= 2: base={base}'
    assert b >= 2, f'base must be >= 2: b={b}'

    # sign (optional)
    if s is not None and s == '-':
        sign = -1
    else:
        sign = +1

    # integer component (required)
    ipart = int(i, base)

    # fraction (optional)
    if f is not None:
        fpart = int(f, base)
        efrac = -len(f)
    else:
        fpart = 0
        efrac = 0

    # exponent (optional)
    if e is not None:
        exp = int(e)
    else:
        exp = 0

    # combine the parts
    return sign * (ipart + fpart * Fraction(base) ** efrac) * (Fraction(b) ** exp)


def decnum_to_fraction(s: str):
    """
    Converts a decimal number to a fraction.

    Works for both integers and floating-point.
    """
    if not isinstance(s, str):
        raise TypeError(f'Expected \'str\', got \'{type(s)}\' for s={s}')

    # apply the regex to extract components
    m = re.fullmatch(_DECIMAL_PATTERN, s.strip())
    if not m:
        raise ValueError(f'invalid decimal number: {s}')

    sign = m.group(1)
    mant = m.group(2)
    exp = m.group(5)

    if '.' in mant:
        parts = mant.split('.')
        assert len(parts) == 2
        i = '0' if parts[0] == '' else parts[0]
        f = parts[1]
    else:
        i = mant
        f = None

    return _sci_to_fraction(sign, i, f, exp, 10, 10)


def hexnum_to_fraction(s: str):
    """
    Converts a hexadecimal number to a fraction.

    Works for both integers and floating-point.
    """
    if not isinstance(s, str):
        raise TypeError(f'Expected \'str\', got \'{type(s)}\' for s={s}')

    m = re.fullmatch(_HEXNUM_PATTERN, s.strip())
    if not m:
        raise ValueError(f'invalid hexadecimal number: {s}')

    sign = m.group(1)
    mant = m.group(2)
    exp = m.group(5)

    if '.' in mant:
        parts = mant.split('.')
        assert len(parts) == 2
        i = parts[0]
        f = parts[1]
    else:
        i = mant
        f = None

    return _sci_to_fraction(sign, i, f, exp, 16, 2)

def is_dyadic(x: numbers.Rational) -> bool:
    """
    Check if the fraction is dyadic, i.e., can be expressed as a
    fraction with a power of two in the denominator.
    """
    return is_power_of_two(abs(int(x.denominator)))
