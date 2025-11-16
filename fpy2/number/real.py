"""
This module defines real computation.
"""

from fractions import Fraction

from .context import REAL
from .gmp import mpfr_value
from .number import Float, RealFloat
from .round import RoundingMode

#####################################################################
# Helpers

def _is_nan(x: Float | Fraction) -> bool:
    return isinstance(x, Float) and x.isnan

def _is_inf(x: Float | Fraction) -> bool:
    return isinstance(x, Float) and x.isinf

def _is_zero(x: Float | Fraction) -> bool:
    return x.is_zero() if isinstance(x, Float) else x == 0

def _signbit(x: Float | Fraction) -> bool:
    return x.s if isinstance(x, Float) else (x < 0)

#####################################################################
# Real methods

def real_neg(x: Float | Fraction) -> Float | Fraction:
    """
    Negate a real number, exactly.
    """
    match x:
        case Float():
            return Float(s=not x.s, x=x, ctx=REAL)
        case Fraction():
            return -x
        case _:
            raise TypeError(f'Expected \'Float\' or \'Fraction\', got \'{type(x)}\' for x={x}')


def real_abs(x: Float | Fraction) -> Float | Fraction:
    """
    Absolute value of a real number, exactly.
    """
    match x:
        case Float():
            return Float(s=False, x=x, ctx=REAL)
        case Fraction():
            return abs(x)
        case _:
            raise TypeError(f'Expected \'Float\' or \'Fraction\', got \'{type(x)}\' for x={x}')


def real_add(x: Float | Fraction, y: Float | Fraction) -> Float | Fraction:
    """
    Add two real numbers, exactly.
    """
    if not isinstance(x, Float | Fraction):
        raise TypeError(f'Expected \'Float\' or \'Fraction\', got \'{type(x)}\' for x={x}')
    if not isinstance(y, Float | Fraction):
        raise TypeError(f'Expected \'Float\' or \'Fraction\', got \'{type(y)}\' for y={y}')

    if _is_nan(x) or _is_nan(y):
        # either is NaN
        return Float(isnan=True, ctx=REAL)
    elif _is_inf(x):
        # x is Inf
        if _is_inf(y):
            # y is also Inf
            if _signbit(x) == _signbit(y):
                # Inf + Inf = Inf
                return Float(s=_signbit(x), isinf=True, ctx=REAL)
            else:
                # Inf + -Inf = NaN
                return Float(isnan=True, ctx=REAL)
        else:
            # y is finite: Inf + y = Inf
            return Float(s=_signbit(x), isinf=True, ctx=REAL)
    elif _is_inf(y):
        # y is Inf, x is finite: x + Inf = Inf
        return Float(s=_signbit(y), isinf=True, ctx=REAL)
    else:
        # both are finite
        match x, y:
            case Float(), Float():
                r = x.as_real() + y.as_real()
                return Float(x=r, ctx=REAL)
            case Fraction(), Fraction():
                return x + y
            case Fraction(), Float():
                return x + y.as_rational()
            case Float(), Fraction():
                return x.as_rational() + y
            case _:
                raise RuntimeError('unreachable', x, y)


def real_sub(x: Float | Fraction, y: Float | Fraction) -> Float | Fraction:
    return real_add(x, real_neg(y))


def real_mul(x: Float | Fraction, y: Float | Fraction) -> Float | Fraction:
    """
    Multiply two real numbers, exactly.
    """
    if not isinstance(x, Float | Fraction):
        raise TypeError(f'Expected \'Float\' or \'Fraction\', got \'{type(x)}\' for x={x}')
    if not isinstance(y, Float | Fraction):
        raise TypeError(f'Expected \'Float\' or \'Fraction\', got \'{type(y)}\' for y={y}')

    if _is_nan(x) or _is_nan(y):
        # either is NaN
        return Float(isnan=True, ctx=REAL)
    elif _is_inf(x):
        # x is Inf
        if _is_zero(y):
            # Inf * 0 = NaN
            return Float(isnan=True, ctx=REAL)
        else:
            # Inf * y = Inf
            s = _signbit(x) != _signbit(y)
            return Float(s=s, isinf=True, ctx=REAL)
    elif _is_inf(y):
        # y is Inf
        if _is_zero(x):
            # 0 * Inf = NaN
            return Float(isnan=True, ctx=REAL)
        else:
            # x * Inf = Inf
            s = _signbit(x) != _signbit(y)
            return Float(s=s, isinf=True, ctx=REAL)
    else:
        # both are finite
        match x, y:
            case Float(), Float():
                r = x.as_real() * y.as_real()
                return Float(x=r, ctx=REAL)
            case Fraction(), Fraction():
                return x * y
            case Fraction(), Float():
                return x * y.as_rational()
            case Float(), Fraction():
                return x.as_rational() * y
            case _:
                raise RuntimeError('unreachable', x, y)


def real_fma(x: Float | Fraction, y: Float | Fraction, z: Float | Fraction) -> Float | Fraction:
    """
    Fused multiply-add operation for real numbers, exactly.
    Computes x * y + z.
    """
    if not isinstance(x, Float | Fraction):
        raise TypeError(f'Expected \'Float\' or \'Fraction\', got \'{type(x)}\' for x={x}')
    if not isinstance(y, Float | Fraction):
        raise TypeError(f'Expected \'Float\' or \'Fraction\', got \'{type(y)}\' for y={y}')
    if not isinstance(z, Float | Fraction):
        raise TypeError(f'Expected \'Float\' or \'Fraction\', got \'{type(z)}\' for z={z}')
    return real_add(real_mul(x, y), z)


def _real_rint(x: Float | Fraction, rm: RoundingMode) -> Float:
    """
    Round a real number to the nearest integer under a specific rounding mode.
    """
    match x:
        case Float():
            if x.is_nar():
                # special value
                return Float(x=x, ctx=REAL)
            else:
                # finite value
                r = x.as_real().round(None, -1, rm)
        case Fraction():
            y = mpfr_value(x, n=-1)
            r = y.as_real().round(None, -1, rm)
        case _:
            raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')

    return Float(x=r, ctx=REAL)


def real_ceil(x: Float | Fraction):
    """
    Round a real number up to the nearest integer.
    """
    return _real_rint(x, RoundingMode.RTP)

def real_floor(x: Float | Fraction):
    """
    Round a real number down to the nearest integer.
    """
    return _real_rint(x, RoundingMode.RTN)

def real_trunc(x: Float | Fraction):
    """
    Rounds a real number towards the nearest integer
    with smaller or equal magnitude to `x`.
    """
    return _real_rint(x, RoundingMode.RTZ)


def real_roundint(x: Float | Fraction):
    """
    Round a real number to the nearest integer,
    rounding ties away from zero in halfway cases.
    """
    return _real_rint(x, RoundingMode.RNA)


#####################################################################

def rto_recip(x: RealFloat, p: int) -> RealFloat:
    """Computes `1/x` to `p` bits using round-to-odd mode."""
    if x == 0:
        raise ZeroDivisionError('1/0')

    s = x.s # result sign
    e = -x.e # result normalized exponent
    exp = e - p + 1 # result unnormalized exponent

    c = x.c # argument significand
    one = 1 << (x.p - 1) # representation of 1.0 in fixed<-M, M+1>

    if c == one:
        # special case: c = 1 => q = 1.0
        q = 1 << (p - 1)
    else:
        # general case: c > 1 => q \in (0.5, 1.0)
        # step 1. digit recurrence for p - 1 digits
        # trick: skip first iteration since we always extract 0
        q = 0 # quotient
        r = one << 1 # remainder (fold first iter)
        for _ in range(p - 1):
            q <<= 1
            if r >= c:
                q |= 1
                r -= c
            r <<= 1
            print(bin(q), r)

        # step 2. generate last digit by inexactness
        q <<= 1
        if r > 0:
            q |= 1

        # step 3. adjust exponent so that q \in [1.0, 2.0)
        exp -= 1

    # result
    return RealFloat(s, exp, q)


def rto_recip_sqr(x: RealFloat, p: int) -> RealFloat:
    """Computes `1/x^2` to `p` bits using round-to-odd mode."""
    assert x != 0

    # square the argument
    x = x * x

    e = -x.e # result normalized exponent
    exp = e - p + 1 # result unnormalized exponent

    m = x.c # argument significand (in 1.m)
    one = 1 << (x.p - 1) # representation of 1.0 in fixed<-M, M+1>

    if m == one:
        # special case: c = 1 => q = 1.0
        q = 1 << (p - 1)
    else:
        # general case: c > 1 => q \in (0.5, 1.0)
        # step 1. digit recurrence algorithm
        # trick: skip first iteration since we always extract 0
        q = 0 # quotient
        r = one << 1 # remainder (constant fold first iter)
        for _ in range(1, p): # compute p - 1 bits
            q <<= 1
            if r >= m:
                q |= 1
                r -= m
            r <<= 1

        # step 2. generate last digit by inexactness
        q <<= 1
        if r > 0:
            q |= 1

        # step 3. adjust exponent so that q \in [1.0, 2.0)
        exp -= 1

    # result (always non-negative)
    return RealFloat(False, exp, q)


def rto_sqrt(x: RealFloat, p: int) -> RealFloat:
    """Computes `sqrt(x)` to `p` bits using round-to-odd mode."""
    assert x >= 0

    s = x.s # result sign
    e = x.e # argument normalized exponent

    # adjust exponent parity
    if e % 2 == 1:
        e -= 1
        c = x.c << 1
        xp = x.p + 1
    else:
        c = x.c
        xp = x.p

    e //= 2
    exp = e - (p - 1) # result unnormalized exponent

    m = c # argument significand (in 1.m)
    one = 1 << (xp - 1) # representation of 1.0 in fixed<-M, M+1>

    if m == one:
        # special case: c = 1 => q = 1.0
        q = 1 << (p - 1)
    else:
        # general case: c > 1 => q \in (0.5, 1.0)
        # step 1. digit recurrence algorithm
        # trick: skip first iteration since we always extract 0
        q = 0 # quotient
        r = m # remainder
        for _ in range(1, p): # compute p - 1 bits
            r <<= 2
            d = 2 * q + 1
            if r >= d:
                q = 2 * q + 1
                r -= d
            else:
                q = 2 * q

        # step 2. generate last digit by inexactness
        q <<= 1
        if r > 0:
            q |= 1

    # result
    return RealFloat(s, exp, q)
