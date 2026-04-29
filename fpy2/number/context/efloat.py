from enum import IntEnum

from ..number import RealFloat, Float, RNG
from ..round import RoundingMode, OverflowMode
from ...utils import default_repr, enum_repr, bitmask, DefaultOr, DEFAULT

from .context import EncodableContext
from .format import EncodableFormat
from .mpb_float import MPBFloatFormat


@enum_repr
class EFloatNanKind(IntEnum):
    """
    Describes how NaN values are encoded for `ExtFloatContext` rounding contexts.
    """
    IEEE_754 = 0
    """IEEE 754 compliant: NaNs have the largest exponent"""
    MAX_VAL = 1
    """NaN has largest exponent and mantissa of all ones"""
    NEG_ZERO = 2
    """NaN replaces -0"""
    NONE = 3
    """No NaNs"""


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

    nan_kind: EFloatNanKind
    """how NaNs are encoded"""

    eoffset: int
    """the exponent offset"""

    _mpb_fmt: MPBFloatFormat
    """underlying multi-precision bounded format"""

    _has_nonzero: bool
    """can this format encode any non-zero values?"""

    def __init__(
        self,
        es: int,
        nbits: int,
        enable_inf: bool,
        nan_kind: EFloatNanKind,
        eoffset: int,
    ):
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

        if not _format_is_valid(es, nbits, enable_inf, nan_kind):
            raise ValueError(
                f'Invalid format: es={es}, nbits={nbits}, enable_inf={enable_inf}, '
                f'nan_kind={nan_kind}, eoffset={eoffset}'
            )

        self.es = es
        self.nbits = nbits
        self.enable_inf = enable_inf
        self.nan_kind = nan_kind
        self.eoffset = eoffset

        self._mpb_fmt = _ext_to_mpb_fmt(es, nbits, enable_inf, nan_kind, eoffset)
        self._has_nonzero = _has_nonzero(nbits, enable_inf, nan_kind)

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

    @property
    def pmax(self) -> int:
        return self._mpb_fmt.pmax

    @property
    def emax(self) -> int:
        return self._mpb_fmt.emax

    @property
    def emin(self) -> int:
        return self._mpb_fmt.emin

    @property
    def expmax(self) -> int:
        return self._mpb_fmt.expmax

    @property
    def expmin(self) -> int:
        return self._mpb_fmt.expmin

    @property
    def nmin(self) -> int:
        return self._mpb_fmt.nmin

    @property
    def m(self) -> int:
        """Size of the mantissa field."""
        return self.pmax - 1

    @property
    def ebias(self) -> int:
        """The exponent "bias" when encoding / decoding values."""
        return self.emax - self.eoffset

    def has_nonzero(self) -> bool:
        """Returns True if this format can represent any non-zero values."""
        return self._has_nonzero

    def is_equiv(self, other) -> bool:
        return (
            isinstance(other, EFloatFormat)
            and self.es == other.es
            and self.nbits == other.nbits
            and self.enable_inf == other.enable_inf
            and self.nan_kind == other.nan_kind
            and self.eoffset == other.eoffset
        )

    def representable_in(self, x: RealFloat | Float) -> bool:
        match x:
            case Float():
                if x.isinf and not self.enable_inf:
                    return False
                if x.isnan and self.nan_kind == EFloatNanKind.NONE:
                    return False
            case RealFloat():
                pass
            case _:
                raise TypeError(f'Expected \'RealFloat\' or \'Float\', got \'{type(x)}\' for x={x}')

        if not self._mpb_fmt.representable_in(x):
            return False
        elif x.is_zero():
            return not (x.s and self.nan_kind == EFloatNanKind.NEG_ZERO)
        return self.has_nonzero()

    def canonical_under(self, x: Float) -> bool:
        if not isinstance(x, Float) or not self.representable_in(x):
            raise TypeError(f'Expected a representable \'Float\', got \'{type(x)}\' for x={x}')
        return self._mpb_fmt.canonical_under(x)

    def normal_under(self, x: Float) -> bool:
        if not isinstance(x, Float) or not self.representable_in(x):
            raise TypeError(f'Expected a representable \'Float\', got \'{type(x)}\' for x={x}')
        return self._mpb_fmt.normal_under(x)

    def normalize(self, x: Float) -> Float:
        if not isinstance(x, Float) or not self.representable_in(x):
            raise TypeError(f'Expected a representable \'Float\', got \'{type(x)}\' for x={x}')
        return self._mpb_fmt.normalize(x)

    def to_ordinal(self, x: Float, infval: bool = False) -> int:
        if not isinstance(x, Float) or not self.representable_in(x):
            raise TypeError(f'Expected a representable \'Float\', got \'{type(x)}\' for x={x}')
        return self._mpb_fmt.to_ordinal(x, infval=infval)

    def to_fractional_ordinal(self, x: Float):
        if not isinstance(x, Float):
            raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
        return self._mpb_fmt.to_fractional_ordinal(x)

    def from_ordinal(self, x: int, infval: bool = False) -> Float:
        return self._mpb_fmt.from_ordinal(x, infval=infval)

    def zero(self, s: bool = False) -> Float:
        """Returns a signed 0 under this format."""
        if not isinstance(s, bool):
            raise TypeError(f'Expected \'bool\' for s={s}, got {type(s)}')
        x = self._mpb_fmt.zero(s)
        if not self.representable_in(x):
            raise ValueError(f'not representable in this format: x={x}')
        return x

    def minval(self, s: bool = False) -> Float:
        if not isinstance(s, bool):
            raise TypeError(f'Expected \'bool\' for s={s}, got {type(s)}')
        x = self._mpb_fmt.minval(s)
        if not self.representable_in(x):
            raise ValueError(f'not representable in this format: x={x}')
        return x

    def min_subnormal(self, s: bool = False) -> Float:
        if not isinstance(s, bool):
            raise TypeError(f'Expected \'bool\' for s={s}, got {type(s)}')
        x = self._mpb_fmt.min_subnormal(s)
        if not self.representable_in(x):
            raise ValueError(f'not representable in this format: x={x}')
        return x

    def max_subnormal(self, s: bool = False) -> Float:
        if not isinstance(s, bool):
            raise TypeError(f'Expected \'bool\' for s={s}, got {type(s)}')
        x = self._mpb_fmt.max_subnormal(s)
        if not self.representable_in(x):
            raise ValueError(f'not representable in this format: x={x}')
        return x

    def min_normal(self, s: bool = False) -> Float:
        if not isinstance(s, bool):
            raise TypeError(f'Expected \'bool\' for s={s}, got {type(s)}')
        x = self._mpb_fmt.min_normal(s)
        if not self.representable_in(x):
            raise ValueError(f'not representable in this format: x={x}')
        return x

    def max_normal(self, s: bool = False) -> Float:
        if not isinstance(s, bool):
            raise TypeError(f'Expected \'bool\' for s={s}, got {type(s)}')
        x = self._mpb_fmt.max_normal(s)
        if not self.representable_in(x):
            raise ValueError(f'not representable in this format: x={x}')
        return x

    def maxval(self, s: bool = False) -> Float:
        if not isinstance(s, bool):
            raise TypeError(f'Expected \'bool\' for s={s}, got {type(s)}')
        x = self._mpb_fmt.maxval(s)
        if not self.representable_in(x):
            raise ValueError(f'not representable in this format: x={x}')
        return x

    def largest(self) -> Float:
        return self.maxval(s=False)

    def smallest(self) -> Float:
        x = self._mpb_fmt.maxval(True)
        return self.zero() if x.is_zero() else x

    def infval(self, s: bool = False) -> Float:
        if not isinstance(s, bool):
            raise TypeError(f'Expected \'bool\' for s={s}, got {type(s)}')
        return self._mpb_fmt.infval(s)

    def total_bits(self) -> int:
        return self.nbits

    def encode(self, x: Float) -> int:
        if not isinstance(x, Float):
            raise TypeError(f'Expected a representable \'Float\', got \'{type(x)}\' for x={x!r}')
        if not self.representable_in(x):
            raise ValueError(f'not representable under this format: x={x!r}')

        sbit = 1 if x.s else 0

        if x.isnan:
            match self.nan_kind:
                case EFloatNanKind.IEEE_754:
                    if self.enable_inf:
                        ebits = bitmask(self.es)
                        mbits = 1 << (self.m - 1)
                    else:
                        ebits = bitmask(self.es)
                        mbits = 0
                case EFloatNanKind.MAX_VAL:
                    ebits = bitmask(self.es)
                    mbits = bitmask(self.m)
                case EFloatNanKind.NEG_ZERO:
                    ebits = 0
                    mbits = 0
                case _:
                    raise RuntimeError(f'unexpected NaN kind {self.nan_kind}')
        elif x.isinf:
            match self.nan_kind:
                case EFloatNanKind.IEEE_754:
                    ebits = bitmask(self.es)
                    mbits = 0
                case EFloatNanKind.MAX_VAL:
                    if self.pmax == 1:
                        ebits = bitmask(self.es) - 1
                        mbits = 1
                    else:
                        ebits = bitmask(self.es)
                        mbits = bitmask(self.m) - 1
                case EFloatNanKind.NEG_ZERO | EFloatNanKind.NONE:
                    ebits = bitmask(self.es)
                    mbits = bitmask(self.m)
                case _:
                    raise RuntimeError(f'unexpected NaN kind {self.nan_kind}')
        elif x.is_zero():
            ebits = 0
            mbits = 0
        elif x.e <= self.emin:
            offset = x.exp - self.expmin
            if offset > 0:
                c = x.c << offset
            elif offset < 0:
                c = x.c >> -offset
            else:
                c = x.c

            ebits = 0
            mbits = c
        else:
            offset = x.p - self.pmax
            if offset > 0:
                c = x.c >> offset
            elif offset < 0:
                c = x.c << -offset
            else:
                c = x.c

            ebits = x.e - self.emin + 1
            mbits = c & bitmask(self.pmax - 1)

        return (sbit << (self.nbits - 1)) | (ebits << self.m) | mbits

    def decode(self, x: int) -> Float:
        if not isinstance(x, int) or x < 0 or x >= (1 << self.nbits):
            raise TypeError(f'Expected integer x={x} on [0, 2 ** {self.nbits})')

        emask = bitmask(self.es)
        mmask = bitmask(self.m)

        sbit = x >> (self.nbits - 1)
        ebits = (x >> self.m) & emask
        mbits = x & mmask

        s = sbit != 0

        match self.nan_kind:
            case EFloatNanKind.IEEE_754:
                if ebits == 0:
                    return Float(s=s, c=mbits, exp=self.expmin)
                elif ebits == emask:
                    if self.enable_inf and mbits == 0:
                        return Float(s=s, isinf=True)
                    return Float(s=s, isnan=True)
                else:
                    c = (1 << self.m) | mbits
                    exp = self.expmin + (ebits - 1)
                    return Float(s=s, c=c, exp=exp)

            case EFloatNanKind.MAX_VAL:
                ord_bits = (ebits << self.m) | mbits
                nan_bits = bitmask(self.nbits - 1)
                inf_bits = nan_bits - 1

                if ord_bits == nan_bits:
                    return Float(s=s, isnan=True)
                elif self.enable_inf and ord_bits == inf_bits:
                    return Float(s=s, isinf=True)
                else:
                    if ebits == 0:
                        return Float(s=s, c=mbits, exp=self.expmin)
                    c = (1 << self.m) | mbits
                    exp = self.expmin + (ebits - 1)
                    return Float(s=s, c=c, exp=exp)

            case EFloatNanKind.NEG_ZERO | EFloatNanKind.NONE:
                ord_bits = (ebits << self.m) | mbits
                inf_bits = bitmask(self.nbits - 1)

                if self.enable_inf and ord_bits == inf_bits:
                    return Float(s=s, isinf=True)

                if ebits == 0:
                    if mbits == 0:
                        if s and self.nan_kind == EFloatNanKind.NEG_ZERO:
                            return Float(s=s, isnan=True)
                        return Float(s=s, c=0, exp=self.expmin)
                    return Float(s=s, c=mbits, exp=self.expmin)

                c = (1 << self.m) | mbits
                exp = self.expmin + (ebits - 1)
                return Float(s=s, c=c, exp=exp)

            case _:
                raise RuntimeError(f'unexpected NaN kind {self.nan_kind}')


@default_repr
class EFloatContext(EncodableContext):
    """
    Rounding context for the "extended" floating-point format
    as described in Brett Saiki's blog post. These formats extend
    the usual IEEE 754 format with three addition parameters:

    - are infinities enabled?
    - how are NaNs encoded?
    - should the exponent be shifted?

    See https://uwplse.org/2025/02/17/Small-Floats.html for details.
    """

    es: int
    """size of the exponent field"""

    nbits: int
    """size of the total representation"""

    enable_inf: bool
    """whether to enable infinities"""

    nan_kind: EFloatNanKind
    """how NaNs are encoded"""

    eoffset: int
    """the exponent offset"""

    rm: RoundingMode
    """rounding mode"""

    overflow: OverflowMode
    """overflow behavior"""

    num_randbits: int | None
    """number of random bits for stochastic rounding, if applicable"""

    rng: RNG | None
    """random number generator for stochastic rounding, if applicable"""

    nan_value: Float | None
    """
    if NaN is not representable, what value should NaN round to?
    """

    inf_value: Float | None
    """
    if Inf is not representable, what value should Inf round to?
    """

    _fmt: EFloatFormat

    def __init__(
        self,
        es: int,
        nbits: int,
        enable_inf: bool,
        nan_kind: EFloatNanKind,
        eoffset: int,
        rm: RoundingMode = RoundingMode.RNE,
        overflow: OverflowMode = OverflowMode.OVERFLOW,
        num_randbits: int | None = 0,
        *,
        rng: RNG | None = None,
        nan_value: Float | None = None,
        inf_value: Float | None = None
    ):
        if not isinstance(rm, RoundingMode):
            raise TypeError(f'Expected \'RoundingMode\', got \'{type(rm)}\' for rm={rm}')
        if not isinstance(overflow, OverflowMode):
            raise TypeError(f'Expected \'OverflowMode\', got \'{type(overflow)}\' for overflow={overflow}')
        if num_randbits is not None and not isinstance(num_randbits, int):
            raise TypeError(f'Expected \'int\', got \'{type(num_randbits)}\' for num_randbits={num_randbits}')
        if overflow == OverflowMode.WRAP:
            raise ValueError('OverflowMode.WRAP is not supported for ExtFloatContext')

        self._fmt = EFloatFormat(es, nbits, enable_inf, nan_kind, eoffset)

        if nan_value is not None:
            if not isinstance(nan_value, Float):
                raise TypeError(f'Expected \'Float\' for nan_value={nan_value}, got {type(nan_value)}')
            if nan_kind == EFloatNanKind.NONE:
                if nan_value.isinf and not enable_inf:
                    raise ValueError(f'Cannot set NaN value to infinity when infinities are disabled: {nan_value}')
                elif not self._fmt._mpb_fmt.representable_in(nan_value):
                    raise ValueError(f'Cannot set NaN value to {nan_value} when it is not representable in this context')

        if inf_value is not None:
            if not isinstance(inf_value, Float):
                raise TypeError(f'Expected \'Float\' for inf_value={inf_value}, got {type(inf_value)}')
            if not enable_inf:
                if nan_kind == EFloatNanKind.NONE:
                    raise ValueError(f'Cannot set Inf value to NaN when NaNs are disabled: {inf_value}')
                elif not self._fmt._mpb_fmt.representable_in(inf_value):
                    raise ValueError(f'Cannot set Inf value to {inf_value} when it is not representable in this context')

        self.es = es
        self.nbits = nbits
        self.enable_inf = enable_inf
        self.nan_kind = nan_kind
        self.eoffset = eoffset
        self.rm = rm
        self.overflow = overflow
        self.num_randbits = num_randbits
        self.rng = rng
        self.nan_value = nan_value
        self.inf_value = inf_value

    def __eq__(self, other):
        return (
            isinstance(other, EFloatContext)
            and self.es == other.es
            and self.nbits == other.nbits
            and self.enable_inf == other.enable_inf
            and self.nan_kind == other.nan_kind
            and self.eoffset == other.eoffset
            and self.rm == other.rm
            and self.overflow == other.overflow
            and self.num_randbits == other.num_randbits
            and self.nan_value == other.nan_value
            and self.inf_value == other.inf_value
        )

    def __hash__(self):
        return hash((
            self.es,
            self.nbits,
            self.enable_inf,
            self.nan_kind,
            self.eoffset,
            self.rm,
            self.overflow,
            self.num_randbits,
            self.nan_value,
            self.inf_value
        ))

    @property
    def pmax(self):
        return self._fmt.pmax

    @property
    def emax(self):
        return self._fmt.emax

    @property
    def emin(self):
        return self._fmt.emin

    @property
    def expmax(self):
        return self._fmt.expmax

    @property
    def expmin(self):
        return self._fmt.expmin

    @property
    def nmin(self):
        return self._fmt.nmin

    @property
    def m(self):
        return self._fmt.m

    @property
    def ebias(self):
        return self._fmt.ebias

    def with_params(
        self, *,
        es: DefaultOr[int] = DEFAULT,
        nbits: DefaultOr[int] = DEFAULT,
        enable_inf: DefaultOr[bool] = DEFAULT,
        nan_kind: DefaultOr[EFloatNanKind] = DEFAULT,
        eoffset: DefaultOr[int] = DEFAULT,
        rm: DefaultOr[RoundingMode] = DEFAULT,
        overflow: DefaultOr[OverflowMode] = DEFAULT,
        num_randbits: DefaultOr[int | None] = DEFAULT,
        rng: DefaultOr[RNG | None] = DEFAULT,
        nan_value: DefaultOr[Float | None] = DEFAULT,
        inf_value: DefaultOr[Float | None] = DEFAULT,
        **kwargs
    ) -> 'EFloatContext':
        if es is DEFAULT:
            es = self.es
        if nbits is DEFAULT:
            nbits = self.nbits
        if enable_inf is DEFAULT:
            enable_inf = self.enable_inf
        if nan_kind is DEFAULT:
            nan_kind = self.nan_kind
        if eoffset is DEFAULT:
            eoffset = self.eoffset
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
            raise TypeError(f'Unexpected parameters {kwargs} for ExtFloatContext')
        return EFloatContext(
            es, nbits, enable_inf, nan_kind, eoffset,
            rm, overflow, num_randbits,
            rng=rng, nan_value=nan_value, inf_value=inf_value,
        )

    def is_stochastic(self) -> bool:
        return self.num_randbits != 0

    def has_nonzero(self) -> bool:
        return self._fmt.has_nonzero()

    def format(self) -> EFloatFormat:
        return self._fmt

    @classmethod
    def from_format(
        cls,
        fmt: EFloatFormat,
        *,
        rm: RoundingMode = RoundingMode.RNE,
        overflow: OverflowMode | None = None,
        num_randbits: int | None = 0,
        rng: 'RNG | None' = None,
        nan_value: Float | None = None,
        inf_value: Float | None = None
    ) -> 'EFloatContext':
        """Creates a context from an `EFloatFormat` and rounding parameters."""
        if not isinstance(fmt, EFloatFormat):
            raise TypeError(f'Expected \'EFloatFormat\', got {type(fmt)}')
        if overflow is None:
            overflow = OverflowMode.OVERFLOW
        return cls(
            fmt.es, fmt.nbits, fmt.enable_inf, fmt.nan_kind, fmt.eoffset,
            rm, overflow, num_randbits, rng=rng,
            nan_value=nan_value, inf_value=inf_value,
        )

    def round_params(self):
        return self._mpb_ctx().round_params()

    def _mpb_ctx(self):
        from .mpb_float import MPBFloatContext
        return MPBFloatContext(
            self._fmt._mpb_fmt.pmax,
            self._fmt._mpb_fmt.emin,
            self._fmt._mpb_fmt.pos_maxval,
            self.rm, self.overflow, self.num_randbits,
            neg_maxval=self._fmt._mpb_fmt.neg_maxval, rng=self.rng,
        )

    def _fixup(self, x: Float):
        if x.isnan and self.nan_kind == EFloatNanKind.NONE:
            if self.nan_value is None:
                if self.enable_inf:
                    return Float.inf(s=x.s, ctx=self)._with_flags(x)
                return self.maxval(s=x.s)._with_flags(x)
            return Float(s=x.s, x=self.nan_value, ctx=self)._with_flags(x)
        elif x.isinf and not self.enable_inf:
            if self.inf_value is None:
                if self.nan_kind != EFloatNanKind.NONE:
                    return Float.nan(s=x.s, ctx=self)._with_flags(x)
                return self.maxval(s=x.s)._with_flags(x)
            return Float(s=x.s, x=self.inf_value, ctx=self)._with_flags(x)
        elif x.is_zero() and x.s and self.nan_kind == EFloatNanKind.NEG_ZERO:
            return Float(x=x, s=False, ctx=self)._with_flags(x)
        return x

    def round(self, x, *, exact: bool = False) -> Float:
        y = self._mpb_ctx().round(x, exact=exact)
        y._ctx = self
        return self._fixup(y)

    def round_at(self, x, n, *, exact: bool = False) -> Float:
        x = self._mpb_ctx().round_at(x, n, exact=exact)
        x._ctx = self
        return self._fixup(x)

    def zero(self, s: bool = False) -> Float:
        return Float(x=self._fmt.zero(s), ctx=self)

    def min_subnormal(self, s: bool = False) -> Float:
        return Float(x=self._fmt.min_subnormal(s), ctx=self)

    def max_subnormal(self, s: bool = False) -> Float:
        return Float(x=self._fmt.max_subnormal(s), ctx=self)

    def min_normal(self, s: bool = False) -> Float:
        return Float(x=self._fmt.min_normal(s), ctx=self)

    def max_normal(self, s: bool = False) -> Float:
        return Float(x=self._fmt.max_normal(s), ctx=self)


def _format_is_valid(
    es: int,
    nbits: int,
    enable_inf: bool,
    nan_kind: EFloatNanKind,
):
    """Returns True if the EFloat format is valid."""
    if nbits < 1:
        return False
    if es < 0 or es >= nbits:
        return False

    p = nbits - es
    match nan_kind:
        case EFloatNanKind.IEEE_754:
            if es == 0:
                return False
            if enable_inf and p == 1:
                return False

        case EFloatNanKind.MAX_VAL:
            if es == 0:
                if p == 1:
                    return False
                elif enable_inf and p == 2:
                    return False
            elif es == 1:
                if enable_inf and p == 1:
                    return False

        case EFloatNanKind.NEG_ZERO | EFloatNanKind.NONE:
            if es == 0 and p == 1 and enable_inf:
                return False

        case _:
            raise RuntimeError(f'unexpected NaN kind{nan_kind}')

    return True


def _has_nonzero(nbits: int, enable_inf: bool, nan_style: EFloatNanKind) -> bool:
    if nbits > 2:
        return True
    elif nbits == 1:
        return False
    return (
        not enable_inf
        and (nan_style == EFloatNanKind.NEG_ZERO or nan_style == EFloatNanKind.NONE)
    )


def _binade_max(p: int, emin: int, e: int) -> RealFloat:
    if e >= emin:
        exp = e - p + 1
        return RealFloat(c=bitmask(p), exp=exp)
    shift = emin - e
    c = bitmask(p) >> shift
    exp = emin - p + 1
    return RealFloat(c=c, exp=exp)


def _ext_to_mpb_fmt(
    es: int,
    nbits: int,
    enable_inf: bool,
    nan_kind: EFloatNanKind,
    eoffset: int,
) -> MPBFloatFormat:
    """Computes the underlying MPBFloatFormat for an EFloat configuration."""
    p = nbits - es
    ebias = 0 if es == 0 else bitmask(es - 1)
    emax_0 = -1 if es == 0 else ebias
    emin_0 = 1 - ebias

    emax = emax_0 + eoffset
    emin = emin_0 + eoffset

    nmin = emin - p

    match nan_kind:
        case EFloatNanKind.IEEE_754:
            maxval = _binade_max(p, emin, emax)

        case EFloatNanKind.MAX_VAL:
            if p == 1:
                if enable_inf:
                    maxval = _binade_max(p, emin, emax - 1)
                else:
                    maxval = _binade_max(p, emin, emax)
            elif p == 2 and enable_inf:
                maxval = _binade_max(p, emin, emax)
            else:
                if enable_inf:
                    maxval = _binade_max(p, emin, emax + 1)
                    maxval = maxval.next_towards_zero(p=p, n=nmin)
                    maxval = maxval.next_towards_zero(p=p, n=nmin)
                else:
                    maxval = _binade_max(p, emin, emax + 1)
                    maxval = maxval.next_towards_zero(p=p, n=nmin)

        case EFloatNanKind.NEG_ZERO | EFloatNanKind.NONE:
            if p == 1:
                if enable_inf:
                    maxval = _binade_max(p, emin, emax)
                else:
                    maxval = _binade_max(p, emin, emax + 1)
            else:
                if enable_inf:
                    maxval = _binade_max(p, emin, emax + 1)
                    maxval = maxval.next_towards_zero(p=p, n=nmin)
                else:
                    maxval = _binade_max(p, emin, emax + 1)

        case _:
            raise RuntimeError(f'unexpected NaN kind {nan_kind}')

    if maxval.is_zero():
        maxval = RealFloat(c=0, exp=emin)

    return MPBFloatFormat(p, emin, maxval)
