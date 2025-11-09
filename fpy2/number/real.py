"""
This module defines real computation.
"""

from fractions import Fraction

from .context import REAL
from .gmp import mpfr_value
from .number import Float
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
