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

    C = fp.pow(fp.R(2), s) + fp.R(1)
    g = C * x
    e = x - g
    s = g + e
    t = x - s

    return s, t


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


@fp.fpy
def fast_2mul(a: fp.Real, b: fp.Real):
    """
    Error-free transformation of the product of two floating-point numbers.

    Returns a tuple (s, t) where:
    - `s` is the floating-point product of `a` and `b`;
    - `t` is the error term such that `s + t = a * b`.

    Assumes that:
    - the rounding context is floating point;
    """

    r1 = a * b
    r2 = fp.fma(a, b, -r1)
    return r1, r2

###########################################################
# FMA

@fp.fpy
def ideal_fma(a: fp.Real, b: fp.Real, c: fp.Real):
    """
    Error-free transformation of the fused multiply-add operation.

    Returns a tuple (r, t) where:
    - `r` is the floating-point result of `a * b + c`;
    - `t` is the error term such that `r + t = a * b + c`.

    This implementation is "ideal" since the error term
    may not be representable in the caller's rounding context.
    """

    r = fp.fma(a, b, c)
    with fp.REAL:
        t = fp.fma(a, b, c) - r
    return r, t

@fp.fpy
def classic_2fma(a: fp.Real, b: fp.Real, c: fp.Real):
    """
    Computes the fused multiply-add operation with error-free transformation.
    This algorithm is due to Boldo and Muller.

    Returns a tuple (r, t) where:
    - `r1` is the floating-point result of `a * b + c`;
    - `r2` and `r3` are the error terms: `a * b + c = r1 + r2 + r3`.

    Assumes that:
    - the rounding context is floating point;
    - the rounding mode is round-nearest.
    """

    r1 = fp.fma(a, b, c)
    u1, u2 = fast_2mul(a, b)
    a1, a2 = classic_2sum(c, u2)
    b1, b2 = classic_2sum(u1, a1)
    g = (b1 - r1) + b2
    r2, r3 = fast_2sum(g, a2)

    return r1, r2, r3
