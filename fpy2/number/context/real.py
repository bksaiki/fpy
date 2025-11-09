"""
This module defines the rounding context for real numbers.
"""

from ..number import Float, RealFloat
from ...utils import default_repr

from .context import Context

#####################################################################
# Real rounding context

@default_repr
class RealContext(Context):
    """
    Rounding context for real numbers.

    The rounding function under this context is the identity function.
    Values are never rounded under this context.
    """

    def __eq__(self, other):
        return isinstance(other, RealContext)

    def __hash__(self):
        return hash(self.__class__)

    def with_params(self, **kwargs) -> 'RealContext':
        if kwargs:
            raise TypeError(f'Unexpected parameters {kwargs} for RealContext')
        return self

    def is_stochastic(self) -> bool:
        return False

    def is_equiv(self, other: Context) -> bool:
        if not isinstance(other, Context):
            raise TypeError(f'Expected \'Context\', got \'{type(other)}\' for other={other}')
        return isinstance(other, RealContext)

    def representable_under(self, x: RealFloat | Float):
        if not isinstance(x, RealFloat | Float):
            raise TypeError(f'Expected \'RealFloat\' or \'Float\', got \'{type(x)}\' for x={x}')
        return True

    def canonical_under(self, x: Float):
        if not isinstance(x, Float):
            raise TypeError(f'Expected a representable \'Float\', got \'{type(x)}\' for x={x}')
        return True

    def normalize(self, x: Float) -> Float:
        return Float(x=x, ctx=self)

    def normal_under(self, x: Float) -> bool:
        if not isinstance(x, Float):
            raise TypeError(f'Expected a representable \'Float\', got \'{type(x)}\' for x={x}')
        return x.is_nonzero()

    def round_params(self):
        return (None, None)

    def round(self, x, *, exact: bool = False):
        xr = self._round_prepare(x)
        return Float(x=xr, ctx=self)

    def round_at(self, x, n: int, *, exact: bool = False):
        raise RuntimeError('cannot round at a specific position in real context')

