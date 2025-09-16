"""
This module defines the usual fixed-width, two's complement, fixed-point numbers.
"""

from ..utils import bitmask, default_repr, DefaultOr, DEFAULT

from .context import EncodableContext
from .mpb_fixed import MPBFixedContext
from .number import RealFloat, Float
from .round import RoundingMode, OverflowMode

@default_repr
class FixedContext(MPBFixedContext, EncodableContext):
    """
    Rounding context for two's fixed-width, two's complement, fixed-point numbers.

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
        rm: RoundingMode = RoundingMode.RNE,
        overflow: OverflowMode = OverflowMode.WRAP,
        num_randbits: int | None = 0,
        *,
        nan_value: Float | None = None,
        inf_value: Float | None = None
    ):
        if not isinstance(signed, bool):
            raise TypeError(f'Expected \'bool\' for signed={signed}, got {type(signed)}')
        if not isinstance(scale, int):
            raise TypeError(f'Expected \'int\' for scale={scale}, got {type(scale)}')
        if not isinstance(nbits, int):
            raise TypeError(f'Expected \'int\' for nbits={nbits}, got {type(nbits)}')

        if signed:
            if nbits < 2:
                raise ValueError(f'For signed representation, nbits={nbits} must be at least 2')
        elif nbits < 1:
            raise ValueError(f'For unsigned representation, nbits={nbits} must be at least 1')

        nmin = scale - 1
        pos_maxval, neg_maxval = _fixed_to_mpb_fixed(signed, scale, nbits)

        super().__init__(
            nmin,
            pos_maxval,
            rm,
            overflow,
            num_randbits,
            neg_maxval=neg_maxval,
            enable_nan=False,
            enable_inf=False,
            nan_value=nan_value,
            inf_value=inf_value
        )

        self.signed = signed
        self.scale = scale
        self.nbits = nbits

    def with_params(
        self, *,
        signed: DefaultOr[bool] = DEFAULT,
        scale: DefaultOr[int] = DEFAULT,
        nbits: DefaultOr[int] = DEFAULT,
        rm: DefaultOr[RoundingMode] = DEFAULT,
        overflow: DefaultOr[OverflowMode] = DEFAULT,
        num_randbits: DefaultOr[int | None] = DEFAULT,
        nan_value: DefaultOr[Float | None] = DEFAULT,
        inf_value: DefaultOr[Float | None] = DEFAULT,
        **kwargs
    ) -> 'FixedContext':
        if signed is DEFAULT:
            signed = self.signed
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
        return FixedContext(
            signed,
            scale,
            nbits,
            rm,
            overflow,
            num_randbits,
            nan_value=nan_value,
            inf_value=inf_value
        )

    def normalize(self, x: Float):
        if not isinstance(x, Float):
            raise TypeError(f'Expected \'Float\', got x={x}')
        return Float(x=super().normalize(x), ctx=self)

    def round(self, x, *, exact: bool = False) -> Float:
        return Float(x=super().round(x, exact=exact), ctx=self)

    def round_at(self, x, n: int, *, exact: bool = False) -> Float:
        if not isinstance(n, int):
            raise TypeError(f'Expected \'int\' for n={n}, got {type(n)}')
        return Float(x=super().round_at(x, n, exact=exact), ctx=self)

    def minval(self, s: bool = False) -> Float:
        if not isinstance(s, bool):
            raise TypeError(f'Expected \'bool\' for s={s}, got {type(s)}')
        return Float(x=super().minval(s), ctx=self)

    def maxval(self, s: bool = False) -> Float:
        if not isinstance(s, bool):
            raise TypeError(f'Expected \'bool\' for s={s}, got {type(s)}')
        return Float(x=super().maxval(s), ctx=self)

    def from_ordinal(self, x: int, infval: bool = False) -> Float:
        if not isinstance(x, int):
            raise TypeError(f'Expected \'int\', got x={x}')
        return Float(x=super().from_ordinal(x, infval), ctx=self)

    def encode(self, x: Float) -> int:
        if not isinstance(x, Float):
            raise TypeError(f'Expected \'Float\', got x={x}')
        if not self.representable_under(x):
            raise ValueError(f'Expected representable value, got x={x} for self={self}')

        # normalize the scale value within the representation
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

            if self.signed and x.s:
                # apply 2's complement
                c = (1 << (self.nbits - 1)) - c

        # ensure the value fits in the bitmask
        if c > bitmask(self.nbits):
            raise OverflowError(f'Value {x} does not fit in {self.nbits} bits')
        return c

    def decode(self, x: int) -> Float:
        if not isinstance(x, int):
            raise TypeError(f'Expected \'int\', got x={x}')
        if x < 0 or x >= (1 << self.nbits):
            raise ValueError(f'Expected value in range [0, {1 << self.nbits}), got x={x}')

        if self.signed and x >= (1 << (self.nbits - 1)):
            # negative value encoded in 2's complement
            c = x - (1 << (self.nbits - 1))
            s = True
        else:
            # positive value
            c = x
            s = False

        return Float(s=s, exp=self.scale, c=c, ctx=self)


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

