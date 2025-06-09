"""
This module defines the usual fixed-width fixed-point numbers.
"""

from typing import Optional

from ..utils import bitmask, default_repr

from .context import EncodableContext
from .mpb_fixed import MPBFixedContext, FixedOverflowKind
from .number import RealFloat, Float
from .round import RoundingMode

@default_repr
class FixedContext(MPBFixedContext, EncodableContext):
    """
    Rounding context for fixed-width fixed-point numbers.

    This context is parameterized by whether it is signed, `signed`,
    the scale factor `scale`, the total number of bits `nbits`,
    the rounding mode `rm`, and the overflow behavior `overflow`.

    Optionally, specify the following keywords:

    - `nan_value`: if NaN is not enabled, what value should NaN round to? [default: `None`];
      if not set, then `round()` will raise a `ValueError` on NaN.
    - `inf_value`: if Inf is not enabled, what value should Inf round to? [default: `None`];
      if not set, then `round()` will raise a `ValueError` on infinity.

    Unlike `MPBFixedContext`, the `FixedContext` inherits from
    `EncodableContext`, since the representation has a well-defined encoding.
    """

    signed: bool
    """is the representation signed?"""

    scale: int
    """the implicit scale factor of the representation"""

    nbits: int
    """the total number of bits in the representation"""

    def __init__(
        self,
        signed: bool,
        scale: int,
        nbits: int,
        rm: RoundingMode,
        overflow: FixedOverflowKind,
        *,
        nan_value: Optional[Float] = None,
        inf_value: Optional[Float] = None
    ):
        nmin = scale - 1
        pos_maxval, neg_maxval = _fixed_to_mpb_fixed(signed, scale, nbits)

        super().__init__(
            nmin,
            pos_maxval,
            rm,
            overflow,
            neg_maxval=neg_maxval,
            enable_nan=False,
            enable_inf=False,
            nan_value=nan_value,
            inf_value=inf_value
        )

        self.signed = signed
        self.scale = scale
        self.nbits = nbits


    def with_rm(self, rm: RoundingMode) -> 'FixedContext':
        return FixedContext(
            self.signed,
            self.scale,
            self.nbits,
            rm,
            self.overflow,
            nan_value=self.nan_value,
            inf_value=self.inf_value
        )

    def normalize(self, x: Float):
        if not isinstance(x, Float):
            raise TypeError(f'Expected \'Float\', got x={x}')
        x = super().normalize(x)
        x.ctx = self
        return x

    def round(self, x) -> Float:
        x = super().round(x)
        x.ctx = self
        return x

    def round_at(self, x, n: int) -> Float:
        if not isinstance(n, int):
            raise TypeError(f'Expected \'int\' for n={n}, got {type(n)}')
        x = super().round_at(x, n)
        x.ctx = self
        return x

    def minval(self, s: bool = False) -> Float:
        if not isinstance(s, bool):
            raise TypeError(f'Expected \'bool\' for s={s}, got {type(s)}')
        x = super().minval(s)
        x.ctx = self
        return x

    def maxval(self, s: bool = False) -> Float:
        if not isinstance(s, bool):
            raise TypeError(f'Expected \'bool\' for s={s}, got {type(s)}')
        x = super().maxval(s)
        x.ctx = self
        return x

    def from_ordinal(self, x: int, infval: bool = False) -> Float:
        if not isinstance(x, int):
            raise TypeError(f'Expected \'int\', got x={x}')
        y = super().from_ordinal(x, infval)
        y.ctx = self
        return y

    def encode(self, x: Float) -> int:
        if not isinstance(x, Float):
            raise TypeError(f'Expected \'Float\', got x={x}')
        if not self.is_representable(x):
            raise ValueError(f'Expected representable value, got x={x} for self={self}')
        raise NotImplementedError

    def decode(self, x: int) -> Float:
        if not isinstance(x, int):
            raise TypeError(f'Expected \'int\', got x={x}')
        raise NotImplementedError


def _fixed_to_mpb_fixed(
    signed: bool,
    scale: int,
    nbits: int,
) -> tuple[RealFloat, RealFloat]:
    """
    Computes the maximum positive and negative values
    for a fixed-width fixed-point representation.
    """

    if signed:
        # signed fixed-point numbers
        pos_maxval = RealFloat(exp=scale, c=bitmask(nbits - 1))
        neg_maxval = RealFloat(s=True, exp=scale, c=1 << (nbits - 1))
    else:
        # unsigned fixed-point numbers
        pos_maxval = RealFloat(exp=scale, c=bitmask(nbits))
        neg_maxval = RealFloat.zero()

    return pos_maxval, neg_maxval

