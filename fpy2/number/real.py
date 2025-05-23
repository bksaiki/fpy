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
