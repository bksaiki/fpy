"""
Error-free transformations
"""

from . import base as fp
from . import core

###########################################################
# Splitting

@fp.fpy
def veltkamp_split(x: fp.Real, s: fp.Real):
    """
    Splits a floating-point number into a high and low part
    such that the high part is representable in `prec(x) - s` digits
    and the low part is representable in `s` digits.
    This algorithm is due to Veltkamp.
    """

    C = fp.pow(2, s) + 1
    g = C * x
    e = x - g
    h = g + e
    l = x - h

    return h, l


###########################################################
# Addition

@fp.fpy
def ideal_2sum(a: fp.Real, b: fp.Real):
    """
    Error-free transformation of the sum of two floating-point numbers.

    Returns a tuple (s, t) where:
    - `s` is the floating-point sum of `a` and `b`;
    - `t` is the error term such that `s + t = a + b`.

    This implementation is "ideal" since the error term
    may not be representable in the caller's rounding context.
    """

    s = a + b
    with fp.REAL:
        t = (a + b) - s
    return s, t


@fp.fpy
def fast_2sum(a: fp.Real, b: fp.Real):
    """
    Error-free transformation of the sum of two floating-point numbers.
    This algorithm is due to Dekker (1971).

    Returns a tuple (s, t) where:
    - `s` is the floating-point sum of `a` and `b`;
    - `t` is the error term such that `s + t = a + b`.

    Assumes that:
    - `|a| >= |b|`;
    - the rounding context is floating point;
    - the rounding mode is round-nearest.
    """

    assert core.isnar(a) or core.isnar(b) or abs(a) >= abs(b)

    s = a + b
    z = s - a
    t = b - z
    return s, t


@fp.fpy
def classic_2sum(a: fp.Real, b: fp.Real):
    """
    Computes the sum of two floating-point numbers with error-free transformation.
    This algorithm is due to Knuth and Moller.

    Returns a tuple (s, t) where:
    - `s` is the floating-point sum of `a` and `b`;
    - `t` is the error term such that `s + t = a + b`.

    Assumes that:
    - the rounding context is floating point;
    - the rounding mode is round-nearest.
    """ 

    s = a + b
    aa = s - b
    bb = s - a
    ea = a - aa
    eb = b - bb
    t = ea + eb
    return s, t


@fp.fpy
def priest_2sum(a: fp.Real, b: fp.Real):
    """
    Computes the sum of two floating-point numbers with error-free transformation.
    This algorithm is due to Priest.

    Returns a tuple (s, t) where:
    - `s` is the faithfully-rounded sum of `a` and `b`;
    - `t` is the error term such that `s + t = a + b`.

    Assumes that:
    - the rounding context is floating point
    """

    if abs(a) < abs(b):
        a, b = b, a

    c = a + b
    e = c - a
    g = c - e
    h = g - a
    f = b - h
    d = f - e
    if d + e != f:
        c = a
        d = b

    return c, d

###########################################################
# Multiplication

@fp.fpy
def ideal_2mul(a: fp.Real, b: fp.Real):
    """
    Error-free transformation of the product of two floating-point numbers.

    Returns a tuple (p, t) where:
    - `s` is the floating-point product of `a` and `b`;
    - `t` is the error term such that `s + t = a * b`.

    This implementation is "ideal" since the error term
    may not be representable in the caller's rounding context.
    """

    s = a * b
    with fp.REAL:
        t = (a * b) - s
    return s, t

@fp.fpy
def classic_2mul(a: fp.Real, b: fp.Real):
    """
    Computes the product of two floating-point numbers with error-free transformation.
    This algorithm is due to Dekker.

    Returns a tuple (s, t) where:
    - `s` is the floating-point product of `a` and `b`;
    - `t` is the error term such that `s + t = a * b`.

    Assumes that:
    - the rounding context is floating point;
    - the rounding mode is round-nearest.
    """

    with fp.INTEGER:
        p = core.max_p()
        s = fp.ceil(p / 2)

    ah, al = veltkamp_split(a, s)
    bh, bl = veltkamp_split(b, s)

    r1 = a * b
    t1 = -r1 + ah * bh
    t2 = t1 + ah * bl
    t3 = t2 + al * bh
    r2 = t3 + al * bl

    return r1, r2
