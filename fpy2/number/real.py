"""
This module defines the rounding context for real numbers.
"""

from fractions import Fraction

from ..utils import default_repr

from .context import Context
from .number import Float, RealFloat
from .round import RoundingMode

@default_repr
class RealContext(Context):
    """
    Rounding context for real numbers.

    The rounding function under this context is the identity function.
    Values are never rounded under this context.
    """

    def with_rm(self, rm: RoundingMode):
        raise RuntimeError('cannot set rounding mode for real context')

    def is_representable(self, x: RealFloat | Float):
        if not isinstance(x, RealFloat | Float):
            raise TypeError(f'Expected \'RealFloat\' or \'Float\', got \'{type(x)}\' for x={x}')
        return True

    def is_canonical(self, x: Float):
        if not isinstance(x, Float):
            raise TypeError(f'Expected a representable \'Float\', got \'{type(x)}\' for x={x}')
        return True

    def normalize(self, x: Float) -> Float:
        return Float(x=x, ctx=self)

    def is_normal(self, x: Float) -> bool:
        if not isinstance(x, Float):
            raise TypeError(f'Expected a representable \'Float\', got \'{type(x)}\' for x={x}')
        return x.is_nonzero()

    def round_params(self):
        return (None, None)

    def round(self, x):
        match x:
            case Float() | RealFloat():
                return Float(x=x, ctx=self)
            case int():
                return Float.from_int(x, ctx=self)
            case float():
                return Float.from_float(x, ctx=self)
            case str() | Fraction():
                # TODO: implement
                raise NotImplementedError
            case _:
                raise TypeError(f'not valid argument x={x}')

    def round_at(self, x, n):
        raise RuntimeError('cannot round at a specific position in real context')


def real_neg(x: Float) -> Float:
    """
    Negate a real number, exactly.
    """
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    return Float(s=not x.s, x=x, ctx=RealContext())

def real_abs(x: Float) -> Float:
    """
    Absolute value of a real number, exactly.
    """
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    return Float(s=False, x=x, ctx=RealContext())

def real_add(x: Float, y: Float) -> Float:
    """
    Add two real numbers, exactly.
    """
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(y, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(y)}\' for y={y}')

    if x.isnan or y.isnan:
        # either is NaN
        return Float(isnan=True, ctx=RealContext())
    elif x.isinf:
        # x is Inf
        if y.isinf:
            # y is also Inf
            if x.s == y.s:
                return Float(s=x.s, isinf=True, ctx=RealContext())
            else:
                return Float(isnan=True, ctx=RealContext())
        else:
            # y is not Inf
            return Float(s=x.s, isinf=True, ctx=RealContext())
    elif y.isinf:
        return Float(s=y.s, isinf=True, ctx=RealContext())
    else:
        # both are finite
        r = x.as_real() + y.as_real()
        return Float(x=r, ctx=RealContext())

def real_sub(x: Float, y: Float) -> Float:
    return real_add(x, real_neg(y))

def real_mul(x: Float, y: Float) -> Float:
    """
    Multiply two real numbers, exactly.
    """
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(y, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(y)}\' for y={y}')

    if x.isnan or y.isnan:
        # either is NaN
        return Float(isnan=True, ctx=RealContext())
    elif x.isinf:
        # x is Inf
        if y.is_zero():
            # Inf * 0 = NaN
            return Float(isnan=True, ctx=RealContext())
        else:
            # Inf * y = Inf
            return Float(s=x.s != y.s, isinf=True, ctx=RealContext())
    elif y.isinf:
        # y is Inf
        if x.is_zero():
            # 0 * Inf = NaN
            return Float(isnan=True, ctx=RealContext())
        else:
            # x * Inf = Inf
            return Float(s=x.s != y.s, isinf=True, ctx=RealContext())
    else:
        # both are finite
        r = x.as_real() * y.as_real()
        return Float(x=r, ctx=RealContext())

def real_ceil(x: Float) -> Float:
    """
    Round a real number up to the nearest integer.
    """
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    
    if x.is_nar():
        # special value
        return Float(x=x, ctx=RealContext())
    else:
        # finite value
        r = x.as_real().round(None, -1, RoundingMode.RTP)
        return Float(x=r, ctx=RealContext())

def real_floor(x: Float) -> Float:
    """
    Round a real number down to the nearest integer.
    """
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    
    if x.is_nar():
        # special value
        return Float(x=x, ctx=RealContext())
    else:
        # finite value
        r = x.as_real().round(None, -1, RoundingMode.RTN)
        return Float(x=r, ctx=RealContext())

def real_trunc(x: Float) -> Float:
    """
    Rounds a real number towards the nearest integer
    with smaller or equal magnitude to `x`.
    """
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    
    if x.is_nar():
        # special value
        return Float(x=x, ctx=RealContext())
    else:
        # finite value
        r = x.as_real().round(None, -1, RoundingMode.RTZ)
        return Float(x=r, ctx=RealContext())

def real_round(x: Float) -> Float:
    """
    Round a real number to the nearest integer,
    rounding ties away from zero in halfway cases.
    """
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    
    if x.is_nar():
        # special value
        return Float(x=x, ctx=RealContext())
    else:
        # finite value
        r = x.as_real().round(None, -1, RoundingMode.RNA)
        return Float(x=r, ctx=RealContext())
