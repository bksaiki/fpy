"""
Error-free transformation (EFT) benchmarks.
"""

from fpy2 import fpy
from fpy2.typing import *


def two_sum(a: Real, b: Real):
    """
    Error-free transformation of the sum of two floating-point numbers.

    This is the classic implementation due to Knuth.
    Must be computed with a nearest rounding mode.
    If `rnd(a + b)` is finite, then `x + y = a + b`.
    """
    x = a + b
    z = x - a
    y = (a - (x - z)) + b - z
    return x, y

def fast_two_sum(a: Real, b: Real):
    """
    Error-free transformation of the sum of two floating-point numbers.

    This is a faster implementation due to Dekker.
    Must be computed with a nearest rounding mode.
    If `rnd(a + b)` is finite, then `x + y = a + b`.

    Assumes that `|a| >= |b|`.
    """
    assert abs(a) >= abs(b)
    x = a + b
    y = (a - x) + b
    return (x, y)

def split(a: Real):
    """
    Split a floating-point number into its high and low parts.

    This is the classic implementation due to Dekker.
    If `a` is finite, then `a = a1 = a2` where `a1`
    and `a2` are non-overlapping with `|a2| <= |a1|`.
    """
    s = 27 # s = ceil(p/2)

    f = pow(2, s) + 1 # 2^s + 1
    c = f * a
    a1 = c - (c - a)
    a2 = a - a1
    return a1, a2

def two_prod(a: Real, b: Real):
    """
    Error-free transformation of the product of two floating-point numbers.

    This is the classic implementation due to Veltkamp.
    Must be computed with a nearest rounding mode.
    If `rnd(a * b)` is finite, then `x * y = a * b`.
    """
    x = a * b
    a1, a2 = split(a)
    b1, b2 = split(b)
    y = (a2 * b) - (((x - a1 * b1) - a1 * b2) - a2 * b1)
    return x, y

def two_prod_fma(a: Real, b: Real):
    """
    Error-free transformation of the product of two floating-point numbers.

    This is a faster implementation due to Ogita, Rump, and Oishi.
    Must be computed with a nearest rounding mode.
    If `rnd(a * b)` is finite, then `x * y = a * b`.
    """
    x = a * b
    y = fma(a, b, -x)
    return x, y

def three_fma(a: Real, b: Real, c: Real):
    """
    Error-free transformation of the sum of three floating-point numbers.

    This implementation is due to Boldo and Muller.
    Must be computed with a nearest rounding mode.
    If `rnd(a * b + c)` is finite, then `x * y + z = a * b + c`.
    """
    x = fma(a, b, c)
    u1, u2 = two_prod_fma(a, b)
    a1, z = two_sum(c, u2)
    (b1, b2) = two_sum(u1, a1)
    y = ((b1 - x) + b2)
    return x, y, z
