"""
This module defines the usual fixed-width, two's complement, fixed-point numbers.
"""

from ..number import RealFloat, Float, RNG
from ..round import RoundingMode, OverflowMode
from ...utils import bitmask, default_repr, DefaultOr, DEFAULT

from .context import EncodableContext
from .format import EncodableFormat
from .mpb_fixed import MPBFixedFormat, MPBFixedContext


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
        pos_maxval = RealFloat(exp=scale, c=bitmask(nbits - 1))
        neg_maxval = RealFloat(s=True, exp=scale, c=1 << (nbits - 1))
    else:
        pos_maxval = RealFloat(exp=scale, c=bitmask(nbits))
        neg_maxval = RealFloat.zero()

    return pos_maxval, neg_maxval


class FixedFormat(MPBFixedFormat, EncodableFormat):
    """
    Number format for two's complement, fixed-width, fixed-point numbers.

    This format is parameterized by whether it is signed `signed`,
    the scale factor `scale`, and the total number of bits `nbits`.
    It describes the set of representable values for `FixedContext`.
    """

    signed: bool
    """is the representation signed?"""

    scale: int
    """the implicit scale factor of the representation"""

    nbits: int
    """the total number of bits in the representation"""

    def __init__(self, signed: bool, scale: int, nbits: int):
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

        pos_maxval, neg_maxval = _fixed_to_mpb_fixed(signed, scale, nbits)
        nmin = scale - 1

        MPBFixedFormat.__init__(self, nmin, pos_maxval, neg_maxval)
        self.signed = signed
        self.scale = scale
        self.nbits = nbits

    def __eq__(self, other):
        return (
            isinstance(other, FixedFormat)
            and self.signed == other.signed
            and self.scale == other.scale
            and self.nbits == other.nbits
        )

    def __hash__(self):
        return hash((self.__class__, self.signed, self.scale, self.nbits))

    def __repr__(self):
        return self.__class__.__name__ + f'(signed={self.signed!r}, scale={self.scale!r}, nbits={self.nbits!r})'

    def total_bits(self) -> int:
        return self.nbits

    def encode(self, x: Float) -> int:
        if not isinstance(x, Float):
            raise TypeError(f'Expected \'Float\', got x={x}')
        if not self.representable_in(x):
            raise ValueError(f'Expected representable value, got x={x} for self={self}')

        if x.c == 0:
            c = 0
        else:
            offset = x.exp - self.scale
            if offset >= 0:
                c = x.c << offset
            else:
                c = x.c >> -offset

            if self.signed and x.s:
                c = (1 << self.nbits) - c

        if c > bitmask(self.nbits):
            raise OverflowError(f'Value {x} does not fit in {self.nbits} bits')
        return c

    def decode(self, x: int) -> Float:
        if not isinstance(x, int):
            raise TypeError(f'Expected \'int\', got x={x}')
        if x < 0 or x >= (1 << self.nbits):
            raise ValueError(f'Expected value in range [0, {1 << self.nbits}), got x={x}')

        if self.signed:
            smask = (1 << (self.nbits - 1))
            if x & smask == 0:
                c = x
                s = False
            else:
                c = (1 << self.nbits) - x
                s = True
        else:
            c = x
            s = False

        return Float(s=s, exp=self.scale, c=c)


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
        rng: RNG | None = None,
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
            nmin, pos_maxval, rm, overflow, num_randbits,
            neg_maxval=neg_maxval, rng=rng,
            enable_nan=False, enable_inf=False,
            nan_value=nan_value, inf_value=inf_value
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
        rng: DefaultOr[RNG | None] = DEFAULT,
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
        if rng is DEFAULT:
            rng = self.rng
        if nan_value is DEFAULT:
            nan_value = self.nan_value
        if inf_value is DEFAULT:
            inf_value = self.inf_value
        if kwargs:
            raise TypeError(f'Unexpected keyword arguments: {kwargs}')
        return FixedContext(
            signed, scale, nbits, rm, overflow, num_randbits,
            rng=rng, nan_value=nan_value, inf_value=inf_value
        )

    def format(self) -> FixedFormat:
        return FixedFormat(self.signed, self.scale, self.nbits)

    @classmethod
    def from_format(
        cls,
        fmt: MPBFixedFormat,
        *,
        rm: RoundingMode = RoundingMode.RNE,
        overflow: OverflowMode | None = None,
        num_randbits: int | None = 0,
        rng: 'RNG | None' = None,
        nan_value: Float | None = None,
        inf_value: Float | None = None,
    ) -> 'FixedContext':
        """Creates a context from a `FixedFormat` and rounding parameters."""
        if not isinstance(fmt, FixedFormat):
            raise TypeError(f'Expected \'FixedFormat\', got {type(fmt)}')
        if overflow is None:
            overflow = OverflowMode.WRAP
        return cls(
            fmt.signed, fmt.scale, fmt.nbits, rm, overflow, num_randbits,
            rng=rng, nan_value=nan_value, inf_value=inf_value,
        )
