"""
This module defines concrete number format types.

Each format type corresponds to a context type and encapsulates
only the format-specific parameters (not rounding parameters).
"""

from ..number import Float, RealFloat
from ..round import RoundingMode
from ...utils import default_repr

from .format import Format, OrdinalFormat, SizedFormat, EncodableFormat

__all__ = [
    'MPFloatFormat',
    'MPFixedFormat',
    'MPSFloatFormat',
    'MPBFixedFormat',
    'MPBFloatFormat',
    'EFloatFormat',
    'IEEEFormat',
    'FixedFormat',
    'SMFixedFormat',
    'ExpFormat',
]

# Default rounding mode used by format objects when delegating to a context.
_DEFAULT_RM = RoundingMode.RNE


def _mp_float_ctx(pmax: int):
    from .mp_float import MPFloatContext
    return MPFloatContext(pmax, _DEFAULT_RM)


def _mp_fixed_ctx(nmin: int, enable_nan: bool, enable_inf: bool):
    from .mp_fixed import MPFixedContext
    return MPFixedContext(nmin, _DEFAULT_RM, enable_nan=enable_nan, enable_inf=enable_inf)


def _mps_float_ctx(pmax: int, emin: int):
    from .mps_float import MPSFloatContext
    return MPSFloatContext(pmax, emin, _DEFAULT_RM)


def _mpb_fixed_ctx(nmin: int, pos_maxval: RealFloat, neg_maxval: RealFloat,
                   enable_nan: bool, enable_inf: bool):
    from .mpb_fixed import MPBFixedContext
    return MPBFixedContext(
        nmin, pos_maxval, _DEFAULT_RM,
        neg_maxval=neg_maxval,
        enable_nan=enable_nan,
        enable_inf=enable_inf,
    )


def _mpb_float_ctx(pmax: int, emin: int, pos_maxval: RealFloat, neg_maxval: RealFloat):
    from .mpb_float import MPBFloatContext
    return MPBFloatContext(pmax, emin, pos_maxval, _DEFAULT_RM, neg_maxval=neg_maxval)


def _efloat_ctx(es: int, nbits: int, enable_inf: bool, nan_kind, eoffset: int):
    from .efloat import EFloatContext
    return EFloatContext(es, nbits, enable_inf, nan_kind, eoffset, _DEFAULT_RM)


def _fixed_ctx(signed: bool, scale: int, nbits: int):
    from .fixed import FixedContext
    from ..round import OverflowMode
    return FixedContext(signed, scale, nbits, _DEFAULT_RM, OverflowMode.SATURATE)


def _sm_fixed_ctx(scale: int, nbits: int):
    from .sm_fixed import SMFixedContext
    from ..round import OverflowMode
    return SMFixedContext(scale, nbits, _DEFAULT_RM, OverflowMode.SATURATE)


def _exp_ctx(nbits: int, eoffset: int):
    from .exponential import ExpContext
    return ExpContext(nbits, eoffset, _DEFAULT_RM)


@default_repr
class MPFloatFormat(Format):
    """
    Number format for multi-precision floating-point numbers.

    This format is parameterized by a fixed precision `pmax`.
    It describes the set of representable values for `MPFloatContext`.
    """

    pmax: int
    """maximum precision"""

    def __init__(self, pmax: int):
        if not isinstance(pmax, int):
            raise TypeError(f'Expected \'int\' for pmax={pmax}, got {type(pmax)}')
        if pmax < 1:
            raise ValueError(f'Expected positive integer for pmax={pmax}')
        self.pmax = pmax

    def __eq__(self, other):
        return isinstance(other, MPFloatFormat) and self.pmax == other.pmax

    def __hash__(self):
        return hash((self.__class__, self.pmax))

    def _ctx(self):
        return _mp_float_ctx(self.pmax)

    def is_equiv(self, other: 'Format') -> bool:
        return isinstance(other, MPFloatFormat) and self.pmax == other.pmax

    def representable_under(self, x):
        return self._ctx().representable_under(x)

    def canonical_under(self, x):
        return self._ctx().canonical_under(x)

    def normal_under(self, x):
        return self._ctx().normal_under(x)

    def normalize(self, x):
        return self._ctx().normalize(x)


@default_repr
class MPFixedFormat(OrdinalFormat):
    """
    Number format for multi-precision fixed-point numbers.

    This format is parameterized by the least-significant digit position
    `nmin` and optional NaN/Inf support flags.
    It describes the set of representable values for `MPFixedContext`.
    """

    nmin: int
    """the first unrepresentable digit"""

    enable_nan: bool
    """is NaN representable?"""

    enable_inf: bool
    """is infinity representable?"""

    def __init__(self, nmin: int, enable_nan: bool = False, enable_inf: bool = False):
        if not isinstance(nmin, int):
            raise TypeError(f'Expected \'int\' for nmin={nmin}, got {type(nmin)}')
        if not isinstance(enable_nan, bool):
            raise TypeError(f'Expected \'bool\' for enable_nan={enable_nan}, got {type(enable_nan)}')
        if not isinstance(enable_inf, bool):
            raise TypeError(f'Expected \'bool\' for enable_inf={enable_inf}, got {type(enable_inf)}')
        self.nmin = nmin
        self.enable_nan = enable_nan
        self.enable_inf = enable_inf

    def __eq__(self, other):
        return (
            isinstance(other, MPFixedFormat)
            and self.nmin == other.nmin
            and self.enable_nan == other.enable_nan
            and self.enable_inf == other.enable_inf
        )

    def __hash__(self):
        return hash((self.__class__, self.nmin, self.enable_nan, self.enable_inf))

    def _ctx(self):
        return _mp_fixed_ctx(self.nmin, self.enable_nan, self.enable_inf)

    @property
    def expmin(self) -> int:
        """The minimum exponent for this format. Equal to `nmin + 1`."""
        return self.nmin + 1

    def is_equiv(self, other: 'Format') -> bool:
        return (
            isinstance(other, MPFixedFormat)
            and self.nmin == other.nmin
            and self.enable_nan == other.enable_nan
            and self.enable_inf == other.enable_inf
        )

    def representable_under(self, x):
        return self._ctx().representable_under(x)

    def canonical_under(self, x):
        return self._ctx().canonical_under(x)

    def normal_under(self, x):
        return self._ctx().normal_under(x)

    def normalize(self, x):
        return self._ctx().normalize(x)

    def to_ordinal(self, x, infval: bool = False):
        return self._ctx().to_ordinal(x, infval)

    def to_fractional_ordinal(self, x):
        return self._ctx().to_fractional_ordinal(x)

    def from_ordinal(self, x: int, infval: bool = False):
        return self._ctx().from_ordinal(x, infval)

    def minval(self, s: bool = False):
        return self._ctx().minval(s)


@default_repr
class MPSFloatFormat(OrdinalFormat):
    """
    Number format for multi-precision floating-point numbers with subnormalization.

    This format is parameterized by a fixed precision `pmax` and
    minimum normalized exponent `emin`.
    It describes the set of representable values for `MPSFloatContext`.
    """

    pmax: int
    """maximum precision"""

    emin: int
    """minimum (normalized) exponent"""

    def __init__(self, pmax: int, emin: int):
        if not isinstance(pmax, int):
            raise TypeError(f'Expected \'int\' for pmax={pmax}, got {type(pmax)}')
        if pmax < 1:
            raise ValueError(f'Expected positive integer for pmax={pmax}')
        if not isinstance(emin, int):
            raise TypeError(f'Expected \'int\' for emin={emin}, got {type(emin)}')
        self.pmax = pmax
        self.emin = emin

    def __eq__(self, other):
        return (
            isinstance(other, MPSFloatFormat)
            and self.pmax == other.pmax
            and self.emin == other.emin
        )

    def __hash__(self):
        return hash((self.__class__, self.pmax, self.emin))

    def _ctx(self):
        return _mps_float_ctx(self.pmax, self.emin)

    @property
    def expmin(self) -> int:
        """Minimum unnormalized exponent."""
        return self.emin - self.pmax + 1

    @property
    def nmin(self) -> int:
        """First unrepresentable digit for every value in the format."""
        return self.expmin - 1

    def is_equiv(self, other: 'Format') -> bool:
        return (
            isinstance(other, MPSFloatFormat)
            and self.pmax == other.pmax
            and self.emin == other.emin
        )

    def representable_under(self, x):
        return self._ctx().representable_under(x)

    def canonical_under(self, x):
        return self._ctx().canonical_under(x)

    def normal_under(self, x):
        return self._ctx().normal_under(x)

    def normalize(self, x):
        return self._ctx().normalize(x)

    def to_ordinal(self, x, infval: bool = False):
        return self._ctx().to_ordinal(x, infval)

    def to_fractional_ordinal(self, x):
        return self._ctx().to_fractional_ordinal(x)

    def from_ordinal(self, x: int, infval: bool = False):
        return self._ctx().from_ordinal(x, infval)

    def minval(self, s: bool = False):
        return self._ctx().minval(s)

    def zero(self, s: bool = False) -> Float:
        """Returns a signed 0 under this format."""
        return self._ctx().zero(s)

    def min_subnormal(self, s: bool = False) -> Float:
        """Returns the smallest subnormal value with sign `s` under this format."""
        return self._ctx().min_subnormal(s)

    def max_subnormal(self, s: bool = False) -> Float:
        """Returns the largest subnormal value with sign `s` under this format."""
        return self._ctx().max_subnormal(s)

    def min_normal(self, s: bool = False) -> Float:
        """Returns the smallest normal value with sign `s` under this format."""
        return self._ctx().min_normal(s)


@default_repr
class MPBFixedFormat(SizedFormat):
    """
    Number format for multi-precision, bounded, fixed-point numbers.

    This format is parameterized by the least-significant digit position
    `nmin`, positive maximum value `pos_maxval`, and optional negative
    maximum value `neg_maxval`.
    It describes the set of representable values for `MPBFixedContext`.
    """

    nmin: int
    """the first unrepresentable digit"""

    pos_maxval: RealFloat
    """positive maximum value"""

    neg_maxval: RealFloat
    """negative maximum value"""

    enable_nan: bool
    """is NaN representable?"""

    enable_inf: bool
    """is infinity representable?"""

    def __init__(
        self,
        nmin: int,
        pos_maxval: RealFloat,
        neg_maxval: RealFloat | None = None,
        enable_nan: bool = False,
        enable_inf: bool = False,
    ):
        if not isinstance(nmin, int):
            raise TypeError(f'Expected \'int\' for nmin={nmin}, got {type(nmin)}')
        if not isinstance(pos_maxval, RealFloat):
            raise TypeError(f'Expected \'RealFloat\' for pos_maxval={pos_maxval}, got {type(pos_maxval)}')
        if neg_maxval is None:
            neg_maxval = RealFloat(s=True, x=pos_maxval)
        elif not isinstance(neg_maxval, RealFloat):
            raise TypeError(f'Expected \'RealFloat\' for neg_maxval={neg_maxval}, got {type(neg_maxval)}')
        if not isinstance(enable_nan, bool):
            raise TypeError(f'Expected \'bool\' for enable_nan={enable_nan}, got {type(enable_nan)}')
        if not isinstance(enable_inf, bool):
            raise TypeError(f'Expected \'bool\' for enable_inf={enable_inf}, got {type(enable_inf)}')
        self.nmin = nmin
        self.pos_maxval = pos_maxval
        self.neg_maxval = neg_maxval
        self.enable_nan = enable_nan
        self.enable_inf = enable_inf

    def __eq__(self, other):
        return (
            isinstance(other, MPBFixedFormat)
            and self.nmin == other.nmin
            and self.pos_maxval == other.pos_maxval
            and self.neg_maxval == other.neg_maxval
            and self.enable_nan == other.enable_nan
            and self.enable_inf == other.enable_inf
        )

    def __hash__(self):
        return hash((
            self.__class__,
            self.nmin,
            self.pos_maxval,
            self.neg_maxval,
            self.enable_nan,
            self.enable_inf,
        ))

    def _ctx(self):
        return _mpb_fixed_ctx(
            self.nmin, self.pos_maxval, self.neg_maxval,
            self.enable_nan, self.enable_inf,
        )

    @property
    def expmin(self) -> int:
        """The minimum exponent for this format. Equal to `nmin + 1`."""
        return self.nmin + 1

    def is_equiv(self, other: 'Format') -> bool:
        return (
            isinstance(other, MPBFixedFormat)
            and self.nmin == other.nmin
            and self.pos_maxval == other.pos_maxval
            and self.neg_maxval == other.neg_maxval
            and self.enable_nan == other.enable_nan
            and self.enable_inf == other.enable_inf
        )

    def representable_under(self, x):
        return self._ctx().representable_under(x)

    def canonical_under(self, x):
        return self._ctx().canonical_under(x)

    def normal_under(self, x):
        return self._ctx().normal_under(x)

    def normalize(self, x):
        return self._ctx().normalize(x)

    def to_ordinal(self, x, infval: bool = False):
        return self._ctx().to_ordinal(x, infval)

    def to_fractional_ordinal(self, x):
        return self._ctx().to_fractional_ordinal(x)

    def from_ordinal(self, x: int, infval: bool = False):
        return self._ctx().from_ordinal(x, infval)

    def minval(self, s: bool = False):
        return self._ctx().minval(s)

    def maxval(self, s: bool = False):
        return self._ctx().maxval(s)

    def infval(self, s: bool = False):
        return self._ctx().infval(s)

    def largest(self):
        return self._ctx().largest()

    def smallest(self):
        return self._ctx().smallest()


@default_repr
class MPBFloatFormat(SizedFormat):
    """
    Number format for multi-precision, bounded, floating-point numbers.

    This format is parameterized by a fixed precision `pmax`, minimum
    normalized exponent `emin`, and positive maximum value `pos_maxval`.
    It describes the set of representable values for `MPBFloatContext`.
    """

    pmax: int
    """maximum precision"""

    emin: int
    """minimum (normalized) exponent"""

    pos_maxval: RealFloat
    """positive maximum value"""

    neg_maxval: RealFloat
    """negative maximum value"""

    def __init__(
        self,
        pmax: int,
        emin: int,
        pos_maxval: RealFloat,
        neg_maxval: RealFloat | None = None,
    ):
        if not isinstance(pmax, int):
            raise TypeError(f'Expected \'int\' for pmax={pmax}, got {type(pmax)}')
        if pmax < 1:
            raise ValueError(f'Expected positive integer for pmax={pmax}')
        if not isinstance(emin, int):
            raise TypeError(f'Expected \'int\' for emin={emin}, got {type(emin)}')
        if not isinstance(pos_maxval, RealFloat):
            raise TypeError(f'Expected \'RealFloat\' for pos_maxval={pos_maxval}, got {type(pos_maxval)}')
        if neg_maxval is None:
            neg_maxval = RealFloat(s=True, x=pos_maxval)
        elif not isinstance(neg_maxval, RealFloat):
            raise TypeError(f'Expected \'RealFloat\' for neg_maxval={neg_maxval}, got {type(neg_maxval)}')
        self.pmax = pmax
        self.emin = emin
        self.pos_maxval = pos_maxval
        self.neg_maxval = neg_maxval

    def __eq__(self, other):
        return (
            isinstance(other, MPBFloatFormat)
            and self.pmax == other.pmax
            and self.emin == other.emin
            and self.pos_maxval == other.pos_maxval
            and self.neg_maxval == other.neg_maxval
        )

    def __hash__(self):
        return hash((self.__class__, self.pmax, self.emin, self.pos_maxval, self.neg_maxval))

    def _ctx(self):
        return _mpb_float_ctx(self.pmax, self.emin, self.pos_maxval, self.neg_maxval)

    @property
    def expmin(self) -> int:
        """Minimum unnormalized exponent."""
        return self.emin - self.pmax + 1

    @property
    def nmin(self) -> int:
        """First unrepresentable digit for every value in the format."""
        return self.expmin - 1

    def is_equiv(self, other: 'Format') -> bool:
        return (
            isinstance(other, MPBFloatFormat)
            and self.pmax == other.pmax
            and self.emin == other.emin
            and self.pos_maxval == other.pos_maxval
            and self.neg_maxval == other.neg_maxval
        )

    def representable_under(self, x):
        return self._ctx().representable_under(x)

    def canonical_under(self, x):
        return self._ctx().canonical_under(x)

    def normal_under(self, x):
        return self._ctx().normal_under(x)

    def normalize(self, x):
        return self._ctx().normalize(x)

    def to_ordinal(self, x, infval: bool = False):
        return self._ctx().to_ordinal(x, infval)

    def to_fractional_ordinal(self, x):
        return self._ctx().to_fractional_ordinal(x)

    def from_ordinal(self, x: int, infval: bool = False):
        return self._ctx().from_ordinal(x, infval)

    def zero(self, s: bool = False) -> Float:
        """Returns a signed 0 under this format."""
        return self._ctx().zero(s)

    def minval(self, s: bool = False):
        return self._ctx().minval(s)

    def min_subnormal(self, s: bool = False) -> Float:
        """Returns the smallest subnormal value with sign `s` under this format."""
        return self._ctx().min_subnormal(s)

    def max_subnormal(self, s: bool = False) -> Float:
        """Returns the largest subnormal value with sign `s` under this format."""
        return self._ctx().max_subnormal(s)

    def min_normal(self, s: bool = False) -> Float:
        """Returns the smallest normal value with sign `s` under this format."""
        return self._ctx().min_normal(s)

    def max_normal(self, s: bool = False) -> Float:
        """Returns the largest normal value with sign `s` under this format."""
        return self._ctx().max_normal(s)

    def maxval(self, s: bool = False):
        return self._ctx().maxval(s)

    def infval(self, s: bool = False):
        return self._ctx().infval(s)

    def largest(self):
        return self._ctx().largest()

    def smallest(self):
        return self._ctx().smallest()


@default_repr
class EFloatFormat(EncodableFormat):
    """
    Number format for the "extended" floating-point format.

    This format is parameterized by the exponent field size `es`,
    total representation size `nbits`, whether infinities are enabled
    `enable_inf`, the NaN kind `nan_kind`, and the exponent offset `eoffset`.
    It describes the set of representable values for `EFloatContext`.
    """

    es: int
    """size of the exponent field"""

    nbits: int
    """size of the total representation"""

    enable_inf: bool
    """whether to enable infinities"""

    nan_kind: 'EFloatNanKind'
    """how NaNs are encoded"""

    eoffset: int
    """the exponent offset"""

    def __init__(
        self,
        es: int,
        nbits: int,
        enable_inf: bool,
        nan_kind: 'EFloatNanKind',
        eoffset: int,
    ):
        from .efloat import EFloatNanKind
        if not isinstance(es, int):
            raise TypeError(f'Expected \'int\' for es={es}, got {type(es)}')
        if not isinstance(nbits, int):
            raise TypeError(f'Expected \'int\' for nbits={nbits}, got {type(nbits)}')
        if not isinstance(enable_inf, bool):
            raise TypeError(f'Expected \'bool\' for enable_inf={enable_inf}, got {type(enable_inf)}')
        if not isinstance(nan_kind, EFloatNanKind):
            raise TypeError(f'Expected \'EFloatNanKind\' for nan_kind={nan_kind}, got {type(nan_kind)}')
        if not isinstance(eoffset, int):
            raise TypeError(f'Expected \'int\' for eoffset={eoffset}, got {type(eoffset)}')
        self.es = es
        self.nbits = nbits
        self.enable_inf = enable_inf
        self.nan_kind = nan_kind
        self.eoffset = eoffset

    def __eq__(self, other):
        return (
            isinstance(other, EFloatFormat)
            and self.es == other.es
            and self.nbits == other.nbits
            and self.enable_inf == other.enable_inf
            and self.nan_kind == other.nan_kind
            and self.eoffset == other.eoffset
        )

    def __hash__(self):
        return hash((self.__class__, self.es, self.nbits, self.enable_inf, self.nan_kind, self.eoffset))

    def _ctx(self):
        return _efloat_ctx(self.es, self.nbits, self.enable_inf, self.nan_kind, self.eoffset)

    def is_equiv(self, other: 'Format') -> bool:
        return (
            isinstance(other, EFloatFormat)
            and self.es == other.es
            and self.nbits == other.nbits
            and self.enable_inf == other.enable_inf
            and self.nan_kind == other.nan_kind
            and self.eoffset == other.eoffset
        )

    def representable_under(self, x):
        return self._ctx().representable_under(x)

    def canonical_under(self, x):
        return self._ctx().canonical_under(x)

    def normal_under(self, x):
        return self._ctx().normal_under(x)

    def normalize(self, x):
        return self._ctx().normalize(x)

    def to_ordinal(self, x, infval: bool = False):
        return self._ctx().to_ordinal(x, infval)

    def to_fractional_ordinal(self, x):
        return self._ctx().to_fractional_ordinal(x)

    def from_ordinal(self, x: int, infval: bool = False):
        return self._ctx().from_ordinal(x, infval)

    def minval(self, s: bool = False):
        return self._ctx().minval(s)

    def maxval(self, s: bool = False):
        return self._ctx().maxval(s)

    def infval(self, s: bool = False):
        return self._ctx().infval(s)

    def largest(self):
        return self._ctx().largest()

    def smallest(self):
        return self._ctx().smallest()

    def total_bits(self) -> int:
        return self._ctx().total_bits()

    def encode(self, x: Float) -> int:
        return self._ctx().encode(x)

    def decode(self, x: int) -> Float:
        return self._ctx().decode(x)

    def has_nonzero(self) -> bool:
        """Returns whether this format can encode non-zero values."""
        return self._ctx().has_nonzero()


class IEEEFormat(EFloatFormat):
    """
    Number format for IEEE 754 floating-point values.

    This format is parameterized by the exponent field size `es`
    and total representation size `nbits`.
    It describes the set of representable values for `IEEEContext`.
    """

    def __init__(self, es: int, nbits: int):
        from .efloat import EFloatNanKind
        super().__init__(es, nbits, True, EFloatNanKind.IEEE_754, 0)

    def __repr__(self):
        return self.__class__.__name__ + f'(es={self.es}, nbits={self.nbits})'

    def __eq__(self, other):
        return isinstance(other, IEEEFormat) and self.es == other.es and self.nbits == other.nbits

    def __hash__(self):
        return hash((self.__class__, self.es, self.nbits))

    def _ctx(self):
        from .ieee754 import IEEEContext
        return IEEEContext(self.es, self.nbits, _DEFAULT_RM)

    def is_equiv(self, other: 'Format') -> bool:
        return isinstance(other, IEEEFormat) and self.es == other.es and self.nbits == other.nbits


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

        # Compute pos_maxval and neg_maxval from signed/scale/nbits
        from ...utils import bitmask
        if signed:
            pos_maxval = RealFloat(exp=scale, c=bitmask(nbits - 1))
            neg_maxval = RealFloat(s=True, exp=scale, c=1 << (nbits - 1))
        else:
            pos_maxval = RealFloat(exp=scale, c=bitmask(nbits))
            neg_maxval = RealFloat.zero()

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

    def _ctx(self):
        return _fixed_ctx(self.signed, self.scale, self.nbits)

    def is_equiv(self, other: 'Format') -> bool:
        return (
            isinstance(other, FixedFormat)
            and self.signed == other.signed
            and self.scale == other.scale
            and self.nbits == other.nbits
        )

    def total_bits(self) -> int:
        return self.nbits

    def encode(self, x: Float) -> int:
        return self._ctx().encode(x)

    def decode(self, x: int) -> Float:
        return self._ctx().decode(x)


class SMFixedFormat(MPBFixedFormat, EncodableFormat):
    """
    Number format for sign-magnitude, fixed-width, fixed-point numbers.

    This format is parameterized by the scale factor `scale` and
    the total number of bits `nbits`.
    It describes the set of representable values for `SMFixedContext`.
    """

    scale: int
    """the implicit scale factor of the representation"""

    nbits: int
    """the total number of bits in the representation"""

    def __init__(self, scale: int, nbits: int):
        if not isinstance(scale, int):
            raise TypeError(f'Expected \'int\' for scale={scale}, got {type(scale)}')
        if not isinstance(nbits, int):
            raise TypeError(f'Expected \'int\' for nbits={nbits}, got {type(nbits)}')
        if nbits < 2:
            raise ValueError(f'nbits={nbits} must be at least 2 for SMFixedFormat')

        from ...utils import bitmask
        nmin = scale - 1
        pos_maxval = RealFloat(exp=scale, c=bitmask(nbits - 1))

        MPBFixedFormat.__init__(self, nmin, pos_maxval)
        self.scale = scale
        self.nbits = nbits

    def __eq__(self, other):
        return (
            isinstance(other, SMFixedFormat)
            and self.scale == other.scale
            and self.nbits == other.nbits
        )

    def __hash__(self):
        return hash((self.__class__, self.scale, self.nbits))

    def __repr__(self):
        return self.__class__.__name__ + f'(scale={self.scale!r}, nbits={self.nbits!r})'

    def _ctx(self):
        return _sm_fixed_ctx(self.scale, self.nbits)

    def is_equiv(self, other: 'Format') -> bool:
        return (
            isinstance(other, SMFixedFormat)
            and self.scale == other.scale
            and self.nbits == other.nbits
        )

    def total_bits(self) -> int:
        return self.nbits

    def encode(self, x: Float) -> int:
        return self._ctx().encode(x)

    def decode(self, x: int) -> Float:
        return self._ctx().decode(x)


@default_repr
class ExpFormat(EncodableFormat):
    """
    Number format for exponential numbers (powers of 2).

    This format is parameterized by the representation size `nbits`
    and exponent offset `eoffset`.
    It describes the set of representable values for `ExpContext`.
    """

    nbits: int
    """size of the representation in bits"""

    eoffset: int
    """exponent offset"""

    def __init__(self, nbits: int, eoffset: int = 0):
        if not isinstance(nbits, int):
            raise TypeError(f'Expected \'int\' for nbits={nbits}, got {type(nbits)}')
        if nbits <= 0:
            raise ValueError(f'nbits must be positive, got nbits={nbits}')
        if not isinstance(eoffset, int):
            raise TypeError(f'Expected \'int\' for eoffset={eoffset}, got {type(eoffset)}')
        self.nbits = nbits
        self.eoffset = eoffset

    def __eq__(self, other):
        return (
            isinstance(other, ExpFormat)
            and self.nbits == other.nbits
            and self.eoffset == other.eoffset
        )

    def __hash__(self):
        return hash((self.__class__, self.nbits, self.eoffset))

    def _ctx(self):
        return _exp_ctx(self.nbits, self.eoffset)

    def is_equiv(self, other: 'Format') -> bool:
        return (
            isinstance(other, ExpFormat)
            and self.nbits == other.nbits
            and self.eoffset == other.eoffset
        )

    def representable_under(self, x):
        return self._ctx().representable_under(x)

    def canonical_under(self, x):
        return self._ctx().canonical_under(x)

    def normal_under(self, x):
        return self._ctx().normal_under(x)

    def normalize(self, x):
        return self._ctx().normalize(x)

    def to_ordinal(self, x, infval: bool = False):
        return self._ctx().to_ordinal(x, infval)

    def to_fractional_ordinal(self, x):
        return self._ctx().to_fractional_ordinal(x)

    def from_ordinal(self, x: int, infval: bool = False):
        return self._ctx().from_ordinal(x, infval)

    def minval(self, s: bool = False):
        return self._ctx().minval(s)

    def maxval(self, s: bool = False):
        return self._ctx().maxval(s)

    def infval(self, s: bool = False):
        return self._ctx().infval(s)

    def largest(self):
        return self._ctx().largest()

    def smallest(self):
        return self._ctx().smallest()

    def total_bits(self) -> int:
        return self.nbits

    def encode(self, x: Float) -> int:
        return self._ctx().encode(x)

    def decode(self, x: int) -> Float:
        return self._ctx().decode(x)
