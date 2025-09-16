"""
This module defines fixed-width, sign-magnitude, fixed-point numbers.
"""

from ..utils import bitmask, default_repr, DefaultOr, DEFAULT

from .context import EncodableContext
from .mpb_fixed import MPBFixedContext
from .number import RealFloat, Float
from .round import RoundingMode, OverflowMode


@default_repr
class SMFixedContext(MPBFixedContext, EncodableContext):
    """
    Rounding context for fixed-width, sign-magnitude, fixed-point numbers.

    This context is parameterized by the scale factor `scale`,
    the total number of bits `nbits`, the rounding mode `rm`, and
    the overflow behavior `overflow`.

    Optionally, specify the following keywords:

    - `nan_value`: if NaN is not enabled, what value should NaN round to? [default: `None`];
      if not set, then `round()` will raise a `ValueError` on NaN.
    - `inf_value`: if Inf is not enabled, what value should Inf round to? [default: `None`];
      if not set, then `round()` will raise a `ValueError` on infinity.
    """

    scale: int
    """the implicit scale factor of the representation"""

    nbits: int
    """the total number of bits in the representation"""

    def __init__(
        self,
        scale: int,
        nbits: int,
        rm: RoundingMode = RoundingMode.RNE,
        overflow: OverflowMode = OverflowMode.WRAP,
        num_randbits: int | None = 0,
        *,
        nan_value: Float | None = None,
        inf_value: Float | None = None
    ):
        if not isinstance(scale, int):
            raise TypeError(f'Expected \'int\' for scale={scale}, got {type(scale)}')
        if not isinstance(nbits, int):
            raise TypeError(f'Expected \'int\' for nbits={nbits}, got {type(nbits)}')

        nmin = scale - 1
        pos_maxval = RealFloat(exp=scale, c=bitmask(nbits - 1))

        super().__init__(
            nmin=nmin,
            maxval=pos_maxval,
            rm=rm,
            overflow=overflow,
            num_randbits=num_randbits,
            enable_nan=False,
            enable_inf=False,
            nan_value=nan_value,
            inf_value=inf_value
        )

        self.scale = scale
        self.nbits = nbits

    def with_params(
        self, *,
        scale: DefaultOr[int] = DEFAULT,
        nbits: DefaultOr[int] = DEFAULT,
        rm: DefaultOr[RoundingMode] = DEFAULT,
        overflow: DefaultOr[OverflowMode] = DEFAULT,
        num_randbits: DefaultOr[int | None] = DEFAULT,
        nan_value: DefaultOr[Float | None] = DEFAULT,
        inf_value: DefaultOr[Float | None] = DEFAULT,
        **kwargs
    ):
        if scale is DEFAULT:
            scale = self.scale
        if nbits is DEFAULT:
            nbits = self.nbits
        if rm is DEFAULT:
            rm = self.rm
        if overflow is DEFAULT:
            overflow = self.overflow
        if num_randbits is DEFAULT:
            num_randbits = self.num_randbits
        if nan_value is DEFAULT:
            nan_value = self.nan_value
        if inf_value is DEFAULT:
            inf_value = self.inf_value
        if kwargs:
            raise TypeError(f'Unexpected keyword arguments: {kwargs}')
        return SMFixedContext(
            scale,
            nbits,
            rm,
            overflow,
            num_randbits,
            nan_value=nan_value,
            inf_value=inf_value
        )

    def encode(self, x: Float) -> int:
        if not isinstance(x, Float):
            raise TypeError(f'Expected \'Float\', got x={x}')
        if not self.representable_under(x):
            raise ValueError(f'Expected representable value, got x={x} for self={self}')

        # sign bit
        sbit = 1 if x.s else 0

        # magnitude
        if x.c == 0:
            c = 0
        else:
            offset = x.exp - self.scale
            if offset >= 0:
                # padding the value with zeroes
                c = x.c << offset
            else:
                # dropping lower bits
                # should be safe since the value is representable
                c = x.c >> -offset

        return (sbit << (self.nbits - 1)) | c

    def decode(self, x: int) -> Float:
        if not isinstance(x, int):
            raise TypeError(f'Expected \'int\', got x={x}')
        if x < 0 or x >= (1 << self.nbits):
            raise ValueError(f'Expected value in range [0, {1 << self.nbits}), got x={x}')

        s = bool((x >> (self.nbits - 1)) & 1)
        c = x & bitmask(self.nbits - 1)
        return Float(s, self.scale, c, ctx=self)
