"""
This module defines the rounding context for real numbers.
"""

from typing import Any, NoReturn

from ...utils import default_repr
from ..number import Float, Real, RealFloat
from .context import Context
from .format import Format

__all__ = [
    'REAL',
    'REAL_FORMAT'
]

#####################################################################
# Real format

@default_repr
class RealFormat(Format):
    """
    Number format for real numbers.

    Every finite value is representable under this format.
    It describes the set of representable values for `RealContext`.
    """

    def __eq__(self, other: object) -> bool:
        return isinstance(other, RealFormat)

    def __hash__(self) -> int:
        return hash(self.__class__)

    def representable_in(self, x: Float | RealFloat) -> bool:
        if not isinstance(x, Float | RealFloat):
            raise TypeError(f'Expected \'RealFloat\' or \'Float\', got \'{type(x)}\' for x={x}')
        return True

    def canonical_under(self, x: Float) -> bool:
        if not isinstance(x, Float):
            raise TypeError(f'Expected a representable \'Float\', got \'{type(x)}\' for x={x}')
        return True

    def normal_under(self, x: Float) -> bool:
        if not isinstance(x, Float):
            raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
        return x.is_nonzero()

    def normalize(self, x: Float) -> Float:
        if not isinstance(x, Float):
            raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
        return x


#####################################################################
# Real rounding context

@default_repr
class RealContext(Context):
    """
    Rounding context for real numbers.

    The rounding function under this context is the identity function.
    Values are never rounded under this context.
    """

    _fmt: RealFormat

    def __init__(self) -> None:
        self._fmt = RealFormat()

    def __str__(self) -> str:
        return 'REAL'

    def __eq__(self, other: object) -> bool:
        return isinstance(other, RealContext)

    def __hash__(self) -> int:
        return hash(self.__class__)

    def with_params(self, **kwargs: Any) -> 'RealContext':
        if kwargs:
            raise TypeError(f'Unexpected parameters {kwargs} for RealContext')
        return self

    def format(self) -> RealFormat:
        return self._fmt

    def is_stochastic(self) -> bool:
        return False

    def rounding_mode(self) -> None:
        return None

    def round_params(self) -> tuple[int | None, int | None]:
        return (None, None)

    def round(self, x: Real | RealFloat | str, *, exact: bool = False) -> Float:
        xr = self._round_prepare(x)
        return Float(x=xr, ctx=self)

    def round_at(self, x: Real | RealFloat | str, n: int, *, exact: bool = False) -> NoReturn:
        raise RuntimeError('cannot round at a specific position in real context')


REAL = RealContext()
"""
Alias for exact computation.
Operations are never rounded under this context.
"""

REAL_FORMAT = REAL.format()
"""
Singleton instance of `RealFormat`.
"""
