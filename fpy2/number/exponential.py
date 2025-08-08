"""
This module implements exponential numbers, i.e.., numbers equal to
2^k where k is an integer.
"""

from typing import Optional, Self

from ..utils import DEFAULT, DefaultOr, bitmask, default_repr
from .context import EncodableContext, Context
from .mp_float import MPFloatContext
from .round import RoundingMode
from .number import Float, RealFloat

def _validate_params(nbits: int, eoffset: int, rm: RoundingMode) -> None:
    if not isinstance(nbits, int):
        raise TypeError(f'Expected \'int\', got \'{type(nbits)}\' for nbits={nbits}')
    if not isinstance(eoffset, int):
        raise TypeError(f'Expected \'int\', got \'{type(eoffset)}\' for eoffset={eoffset}')
    if not isinstance(rm, RoundingMode):
        raise TypeError(f'Expected \'RoundingMode\', got \'{type(rm)}\' for rm={rm}')
    if nbits <= 0:
        raise ValueError(f'nbits must be positive, got nbits={nbits}')

def _exponent_bounds(nbits: int, eoffset: int) -> tuple[int, int]:
    """
    Calculate the minimum and maximum exponent bounds for the given
    number of bits and exponent offset.
    """

    # IEEE 754 style parameters
    es = nbits
    emax_0 = bitmask(es - 1)
    emin_0 = 1 - emax_0

    # apply exponent offset to compute final exponent parameters
    emax = emax_0 + eoffset
    emin = emin_0 + eoffset

    # subnormal exponent is just below the minimum exponent
    emin -= 1

    return emin, emax

@default_repr
class ExpContext(EncodableContext):
    """
    Rounding context for exponential numbers, i.e., numbers
    of the form 2^k where k is an integer. The context is
    parameterized by the size of the representation `nbits`,
    an exponent offset `eoffset`, and the rounding mode `rm`.

    This context implements `EncodableContext`.
    """

    nbits: int
    """size of the representation in bits"""

    eoffset: int
    """exponent offset"""

    rm: RoundingMode
    """rounding mode"""

    _mp_ctx: MPFloatContext
    """this context without exponent bounds"""

    _emin: int
    """minimum exponent value"""

    _emax: int
    """maximum exponent value"""

    def __init__(self, nbits: int, eoffset: int = 0, rm: RoundingMode = RoundingMode.RNE):
        _validate_params(nbits, eoffset, rm)
        emin, emax = _exponent_bounds(nbits, eoffset)

        self.nbits = nbits
        self.eoffset = eoffset
        self.rm = rm
        self._mp_ctx = MPFloatContext(1, rm=rm) # 1 bit of mantissa
        self._emin = emin
        self._emax = emax

    @property
    def emin(self) -> int:
        """The minimum representable exponent value"""
        return self._emin

    @property
    def emax(self) -> int:
        """The maximum representable exponent value"""
        return self._emax

    def with_params(self, **kwargs) -> Self:
        raise NotImplementedError

    def is_stochastic(self) -> bool:
        raise NotImplementedError

    def is_equiv(self, other: Context) -> bool:
        raise NotImplementedError

    def representable_under(self, x: Float | RealFloat) -> bool:
        raise NotImplementedError

    def canonical_under(self, x: Float) -> bool:
        raise NotImplementedError

    def normal_under(self, x: Float) -> bool:
        raise NotImplementedError

    def normalize(self, x: Float) -> Float:
        raise NotImplementedError

    def round_params(self) -> tuple[Optional[int], Optional[int]]:
        raise NotImplementedError

    def round(self, x, *, exact: bool = False) -> Float:
        raise NotImplementedError

    def round_at(self, x, n: int, *, exact: bool = False) -> Float:
        raise NotImplementedError

    def to_ordinal(self, x: Float, infval: bool = False) -> int:
        raise NotImplementedError

    def from_ordinal(self, x: int, infval: bool = False) -> Float:
        raise NotImplementedError

    def minval(self, s: bool = False) -> Float:
        raise NotImplementedError

    def maxval(self, s: bool = False) -> Float:
        raise NotImplementedError

    def encode(self, x: Float) -> int:
        raise NotImplementedError

    def decode(self, x: int) -> Float:
        raise NotImplementedError

