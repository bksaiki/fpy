from enum import IntEnum

from ..utils import default_repr, enum_repr, bitmask, DefaultOr, DEFAULT

from .context import Context, EncodableContext
from .number import RealFloat, Float
from .mpb_float import MPBFloatContext
from .round import RoundingMode, OverflowMode

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

    nan_value: Float | None
    """
    if NaN is not representable, what value should NaN round to?
    if not set, then `round()` will produce to Inf or MAX_VAL whichever
    is representable.
    """

    inf_value: Float | None
    """
    if Inf is not representable, what value should Inf round to?
    if not set, then `round()` will produce NAN or MAX_VAL whichever
    is representable.
    """

    _mpb_ctx: MPBFloatContext
    """this context as an `MPBFloatContext`"""

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
        nan_value: Float | None = None,
        inf_value: Float | None = None
    ):
        if not isinstance(es, int):
            raise TypeError(f'Expected \'int\', got \'{type(es)}\' for es={es}')
        if not isinstance(nbits, int):
            raise TypeError(f'Expected \'int\', got \'{type(nbits)}\' for nbits={nbits}')
        if not isinstance(rm, RoundingMode):
            raise TypeError(f'Expected \'RoundingMode\', got \'{type(rm)}\' for rm={rm}')
        if not isinstance(overflow, OverflowMode):
            raise TypeError(f'Expected \'OverflowMode\', got \'{type(overflow)}\' for overflow={overflow}')
        if not isinstance(enable_inf, bool):
            raise TypeError(f'Expected \'bool\', got \'{type(enable_inf)}\' for enable_inf={enable_inf}')
        if not isinstance(nan_kind, EFloatNanKind):
            raise TypeError(f'Expected \'NanEncoding\', got \'{type(nan_kind)}\' for nan_encoding={nan_kind}')
        if not isinstance(eoffset, int):
            raise TypeError(f'Expected \'int\', got \'{type(eoffset)}\' for eoffset={eoffset}')
        if num_randbits is not None and not isinstance(num_randbits, int):
            raise TypeError(f'Expected \'int\', got \'{type(num_randbits)}\' for num_randbits={num_randbits}')

        # check validity
        if not _format_is_valid(es, nbits, enable_inf, nan_kind):
            raise ValueError(
                f'Invalid format: es={es}, nbits={nbits}, enable_inf={enable_inf}, '
                f'nan_kind={nan_kind}, eoffset={eoffset}'
            )

        if overflow == OverflowMode.WRAP:
            raise ValueError('OverflowMode.WRAP is not supported for ExtFloatContext')
        mpb_ctx = _ext_to_mpb(es, nbits, enable_inf, nan_kind, eoffset, rm, overflow, num_randbits)

        if nan_value is not None:
            if not isinstance(nan_value, Float):
                raise TypeError(f'Expected \'Float\' for nan_value={nan_value}, got {type(nan_value)}')
            if nan_kind == EFloatNanKind.NONE:
                # this field matters
                if nan_value.isinf and not enable_inf:
                    raise ValueError(f'Cannot set NaN value to infinity when infinities are disabled: {nan_value}')
                elif not mpb_ctx.representable_under(nan_value):
                    raise ValueError(f'Cannot set NaN value to {nan_value} when it is not representable in this context')

        if inf_value is not None:
            if not isinstance(inf_value, Float):
                raise TypeError(f'Expected \'Float\' for inf_value={inf_value}, got {type(inf_value)}')
            if not enable_inf:
                # this field matters
                if nan_kind == EFloatNanKind.NONE:
                    raise ValueError(f'Cannot set Inf value to NaN when NaNs are disabled: {inf_value}')
                elif not mpb_ctx.representable_under(inf_value):
                    raise ValueError(f'Cannot set Inf value to {inf_value} when it is not representable in this context')

        self.es = es
        self.nbits = nbits
        self.enable_inf = enable_inf
        self.nan_kind = nan_kind
        self.eoffset = eoffset
        self.rm = rm
        self.overflow = overflow
        self.num_randbits = num_randbits
        self.nan_value = nan_value
        self.inf_value = inf_value
        self._mpb_ctx = mpb_ctx

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
        """Maximum allowable precision."""
        return self._mpb_ctx.pmax

    @property
    def emax(self):
        """Maximum normalized exponent."""
        return self._mpb_ctx.emax

    @property
    def emin(self):
        """Minimum normalized exponent."""
        return self._mpb_ctx.emin

    @property
    def expmax(self):
        """Maximum unnormalized exponent."""
        return self._mpb_ctx.expmax

    @property
    def expmin(self):
        """Minimum unnormalized exponent."""
        return self._mpb_ctx.expmin

    @property
    def nmin(self):
        """
        First unrepresentable digit for every value in the representation.
        """
        return self._mpb_ctx.nmin

    @property
    def m(self):
        """Size of the mantissa field."""
        return self.pmax - 1

    @property
    def ebias(self):
        """The exponent "bias" when encoding / decoding values."""
        return self.emax - self.eoffset

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
        if nan_value is DEFAULT:
            nan_value = self.nan_value
        if inf_value is DEFAULT:
            inf_value = self.inf_value
        if kwargs:
            raise TypeError(f'Unexpected parameters {kwargs} for ExtFloatContext')
        return EFloatContext(
            es,
            nbits,
            enable_inf,
            nan_kind,
            eoffset,
            rm,
            overflow,
            num_randbits,
            nan_value=nan_value,
            inf_value=inf_value
        )

    def is_stochastic(self) -> bool:
        return self.num_randbits != 0

    def is_equiv(self, other):
        if not isinstance(other, Context):
            raise TypeError(f'Expected \'Context\', got \'{type(other)}\' for other={other}')
        return (
            isinstance(other, EFloatContext)
            and self.es == other.es
            and self.nbits == other.nbits
            and self.enable_inf == other.enable_inf
            and self.nan_kind == other.nan_kind
            and self.eoffset == other.eoffset
        )

    def representable_under(self, x: RealFloat | Float) -> bool:
        match x:
            case Float():
                if x.ctx is not None and self.is_equiv(x.ctx):
                    # same context, so representable
                    return True
                elif x.isinf and not self.enable_inf:
                    # Inf is not representable in this context
                    return False
                elif x.isnan and self.nan_kind == EFloatNanKind.NONE:
                    # NaN is not representable in this context
                    return False
            case RealFloat():
                pass
            case _:
                raise TypeError(f'Expected \'RealFloat\' or \'Float\', got \'{type(x)}\' for x={x}')

        if not self._mpb_ctx.representable_under(x):
            return False
        elif x.is_zero() and x.s and self.nan_kind == EFloatNanKind.NEG_ZERO:
            # -0 is not representable in this context
            return False
        else:
            # otherwise, it is representable
            return True

    def canonical_under(self, x: Float) -> bool:
        if not isinstance(x, Float) or not self.representable_under(x):
            raise TypeError(f'Expected a representable \'Float\', got \'{type(x)}\' for x={x}')
        return self._mpb_ctx.canonical_under(x)

    def _normalize(self, x: Float) -> Float:
        return self._mpb_ctx._normalize(x)

    def normalize(self, x: Float) -> Float:
        if not isinstance(x, Float) or not self.representable_under(x):
            raise TypeError(f'Expected a representable \'Float\', got \'{type(x)}\' for x={x}')
        return Float(x=self._normalize(x), ctx=self)

    def normal_under(self, x: Float) -> bool:
        if not isinstance(x, Float) or not self.representable_under(x):
            raise TypeError(f'Expected a representable \'Float\', got \'{type(x)}\' for x={x}')
        return self._mpb_ctx.normal_under(x)

    def round_params(self):
        return self._mpb_ctx.round_params()

    def _fixup(self, x: Float):
        if x.isnan and self.nan_kind == EFloatNanKind.NONE:
            # NaN is not representable in this context
            if self.nan_value is None:
                # resolve by default rules
                if self.enable_inf:
                    # NaN rounds to Inf
                    return Float.inf(s=x.s, ctx=self)
                else:
                    # NaN rounds to MAX_VAL
                    return self.maxval(s=x.s)
            return Float(s=x.s, x=self.nan_value, ctx=self)
        elif x.isinf and not self.enable_inf:
            # Inf is not representable in this context
            if self.inf_value is None:
                # resolve by default rules
                if self.nan_kind != EFloatNanKind.NONE:
                    # Inf rounds to NaN
                    return Float.nan(s=x.s, ctx=self)
                else:
                    # Inf rounds to MAX_VAL
                    return self.maxval(s=x.s)
            return Float(s=x.s, x=self.inf_value, ctx=self)
        elif x.is_zero() and x.s and self.nan_kind == EFloatNanKind.NEG_ZERO:
            # -0 is not representable in this context
            return Float(x=x, s=False, ctx=self)
        else:
            return x

    def round(self, x, *, exact: bool = False) -> Float:
        x = self._mpb_ctx.round(x, exact=exact)
        x._ctx = self
        return self._fixup(x)

    def round_at(self, x, n, *, exact: bool = False) -> Float:
        x = self._mpb_ctx.round_at(x, n, exact=exact)
        x._ctx = self
        return self._fixup(x)

    def to_ordinal(self, x: Float, infval: bool = False) -> int:
        if not isinstance(x, Float) or not self.representable_under(x):
            raise TypeError(f'Expected a representable \'Float\', got \'{type(x)}\' for x={x}')
        return self._mpb_ctx.to_ordinal(x, infval=infval)

    def from_ordinal(self, x: int, infval: bool = False) -> Float:
        y = self._mpb_ctx.from_ordinal(x, infval=infval)
        y._ctx = self
        return y

    def zero(self, s: bool = False):
        """
        Returns a signed 0 under this context.

        Raises `ValueError` if the value is not representable.
        """
        if not isinstance(s, bool):
            raise TypeError(f'Expected \'bool\' for s={s}, got {type(s)}')
        x = Float(x=self._mpb_ctx.zero(s), ctx=self)
        if not self.representable_under(x):
            raise ValueError(f'not representable in this context: x={x}')
        return x

    def minval(self, s: bool = False) -> Float:
        """
        Returns the smallest non-zero value with sign `s` under this context.

        Raises `ValueError` if the value is not representable.
        """
        if not isinstance(s, bool):
            raise TypeError(f'Expected \'bool\' for s={s}, got {type(s)}')
        x = self._mpb_ctx.minval(s)
        if not self.representable_under(x):
            raise ValueError(f'not representable in this context: x={x}')
        return Float(x=x, ctx=self)

    def min_subnormal(self, s: bool = False) -> Float:
        """
        Returns the smallest subnormal value with sign `s` under this context.

        Raises `ValueError` if the value is not representable.
        """
        if not isinstance(s, bool):
            raise TypeError(f'Expected \'bool\' for s={s}, got {type(s)}')
        x = self._mpb_ctx.min_subnormal(s)
        if not self.representable_under(x):
            raise ValueError(f'not representable in this context: x={x}')
        return Float(x=x, ctx=self)

    def max_subnormal(self, s: bool = False) -> Float:
        """
        Returns the largest subnormal value with sign `s` under this context.

        Raises `ValueError` if the value is not representable.
        """
        if not isinstance(s, bool):
            raise TypeError(f'Expected \'bool\' for s={s}, got {type(s)}')
        x = self._mpb_ctx.max_subnormal(s)
        if not self.representable_under(x):
            raise ValueError(f'not representable in this context: x={x}')
        return Float(x=x, ctx=self)

    def min_normal(self, s: bool = False) -> Float:
        """
        Returns the smallest normal value with sign `s` under this context.

        Raises `ValueError` if the value is not representable.
        """
        if not isinstance(s, bool):
            raise TypeError(f'Expected \'bool\' for s={s}, got {type(s)}')
        x = self._mpb_ctx.min_normal(s)
        if not self.representable_under(x):
            raise ValueError(f'not representable in this context: x={x}')
        return Float(x=x, ctx=self)

    def max_normal(self, s: bool = False) -> Float:
        """
        Returns the largest normal value with sign `s` under this context.

        Raises `ValueError` if the value is not representable.
        """
        if not isinstance(s, bool):
            raise TypeError(f'Expected \'bool\' for s={s}, got {type(s)}')
        x = self._mpb_ctx.max_normal(s)
        if not self.representable_under(x):
            raise ValueError(f'not representable in this context: x={x}')
        return Float(x=x, ctx=self)

    def maxval(self, s: bool = False) -> Float:
        """
        Returns the largest value with sign `s` under this context.

        Raises `ValueError` if the value is not representable.
        """
        if not isinstance(s, bool):
            raise TypeError(f'Expected \'bool\' for s={s}, got {type(s)}')
        x = self._mpb_ctx.maxval(s)
        if not self.representable_under(x):
            raise ValueError(f'Not representable in this context: x={x}')
        return Float(x=x, ctx=self)

    def infval(self, s: bool = False):
        """
        Returns the first non-representable value larger
        than `maxval` with sign `s`.
        """
        if not isinstance(s, bool):
            raise TypeError(f'Expected \'bool\' for s={s}, got {type(s)}')
        return self._mpb_ctx.infval(s)

    def encode(self, x: Float) -> int:
        if not isinstance(x, Float):
            raise TypeError(f'Expected a representable \'Float\', got \'{type(x)}\' for x={x!r}')
        if not self.representable_under(x):
            raise ValueError(f'not representable under this context: x={x!r}, ctx={self!r}')

        # sign bit
        sbit = 1 if x.s else 0

        # case split by class
        if x.isnan:
            # NaN: placement of NaN depends on NaN / Inf encoding
            match self.nan_kind:
                case EFloatNanKind.IEEE_754:
                    if self.enable_inf:
                        # usual IEEE 754 encoding: NaN => +qNaN(0)
                        ebits = bitmask(self.es)
                        mbits = 1 << (self.m - 1)
                    else:
                        # no infinity, NaN => +NaN(0)
                        ebits = bitmask(self.es)
                        mbits = 0
                case EFloatNanKind.MAX_VAL:
                    # NaN is the maximum encoding
                    ebits = bitmask(self.es)
                    mbits = bitmask(self.m)
                case EFloatNanKind.NEG_ZERO:
                    # NaN replaces -0
                    ebits = 0
                    mbits = 0
                case _:
                    # ExtNanKind.NONE => no NaN so impossible
                    raise RuntimeError(f'unexpected NaN kind {self.nan_kind}')
        elif x.isinf:
            # Inf: placement of Inf depends on NaN encoding
            match self.nan_kind:
                case EFloatNanKind.IEEE_754:
                    # usual IEEE 754 encoding
                    ebits = bitmask(self.es)
                    mbits = 0
                case EFloatNanKind.MAX_VAL:
                    # Inf one before the maximum encoding
                    if self.pmax == 1:
                        # Inf is in the previous binade
                        ebits = bitmask(self.es) - 1
                        mbits = 1
                    else:
                        # Inf is in the last binade
                        ebits = bitmask(self.es)
                        mbits = bitmask(self.m) - 1
                case EFloatNanKind.NEG_ZERO | EFloatNanKind.NONE:
                    # Inf is the maximum encoding
                    ebits = bitmask(self.es)
                    mbits = bitmask(self.m)
                case _:
                    raise RuntimeError(f'unexpected NaN kind {self.nan_kind}')
        elif x.is_zero():
            # zero
            ebits = 0
            mbits = 0
        elif x.e <= self.emin:
            # subnormal: sgn(x) * [ 0 | m ]
            # need to ensure that exp=self.expmin
            offset = x.exp - self.expmin
            if offset > 0:
                # need to increase precision of `c`
                c = x.c << offset
            elif offset < 0:
                # need to decrease precision of `c`
                c = x.c >> -offset
            else:
                # no change
                c = x.c

            ebits = 0
            mbits = c
        else:
            # normal: sgn(x) * [ eord | m ]
            # normalize so that p=self.pmax
            offset = x.p - self.pmax
            if offset > 0:
                # too much precision
                c = x.c >> offset
            elif offset < 0:
                # too little precision
                c = x.c << -offset
            else:
                # no change
                c = x.c

            ebits = x.e - self.emin + 1
            mbits = c & bitmask(self.pmax - 1)

        # encode the value
        return (sbit << (self.nbits - 1)) | (ebits << self.m) | mbits


    def decode(self, x: int) -> Float:
        if not isinstance(x, int) or x < 0 or x >= (1 << self.nbits):
            raise TypeError(f'Expected integer x={x} on [0, 2 ** {self.nbits})')

        # bitmasks
        emask = bitmask(self.es)
        mmask = bitmask(self.m)

        # extract bits
        sbit = x >> (self.nbits - 1)
        ebits = (x >> self.m) & emask
        mbits = x & mmask

        # sign bit
        s = sbit != 0

        # case split on NaN encoding
        match self.nan_kind:
            case EFloatNanKind.IEEE_754:
                # IEEE 754 style decoding
                if ebits == 0: # all zeros
                    # subnormal / zero
                    c = mbits
                    return Float(s=s, c=c, exp=self.expmin, ctx=self)
                elif ebits == emask: # all ones
                    # infinite / NaN
                    if self.enable_inf and mbits == 0:
                        # infinite (when enabled)
                        return Float(s=s, isinf=True, ctx=self)
                    else:
                        # otherwise NaN
                        return Float(s=s, isnan=True, ctx=self)
                else:
                    # normal number
                    c = (1 << self.m) | mbits
                    exp = self.expmin + (ebits - 1)
                    return Float(s=s, c=c, exp=exp, ctx=self)

            case EFloatNanKind.MAX_VAL:
                # NaN is the maximum encoding
                # if it exists, then Inf is in the previous encoding

                # easier to consider ebits | mbits together
                ord_bits = (ebits << self.m) | mbits

                # special values
                nan_bits = bitmask(self.nbits - 1) # all ones [E | M]
                inf_bits  = nan_bits - 1

                # check if `ord_bits` is a special value
                if ord_bits == nan_bits:
                    # NaN
                    return Float(s=s, isnan=True, ctx=self)
                elif self.enable_inf and ord_bits == inf_bits:
                    # Inf
                    return Float(s=s, isinf=True, ctx=self)
                else:
                    # real number
                    if ebits == 0:
                        # subnormal / zero
                        c = mbits
                        return Float(s=s, c=c, exp=self.expmin, ctx=self)
                    else:
                        # normal number
                        c = (1 << self.m) | mbits
                        exp = self.expmin + (ebits - 1)
                        return Float(s=s, c=c, exp=exp, ctx=self)

            case EFloatNanKind.NEG_ZERO | EFloatNanKind.NONE:
                # NaN replaces -0 (or no NaN)
                # if it exists, then Inf is the maximum encoding

                # easier to consider ebits | mbits together
                ord_bits = (ebits << self.m) | mbits

                # special value
                inf_bits = bitmask(self.nbits - 1) # all ones [E | M]

                # check if `ord_bits` is a special value
                if self.enable_inf and ord_bits == inf_bits:
                    # Inf
                    return Float(s=s, isinf=True, ctx=self)
                else:
                    # real number
                    if ebits == 0:
                        # subnormal / zero (or NaN)
                        if mbits == 0:
                            # zero (or NaN)
                            if s and self.nan_kind == EFloatNanKind.NEG_ZERO:
                                # NaN
                                return Float(s=s, isnan=True, ctx=self)
                            else:
                                # zero
                                return Float(s=s, c=0, exp=self.expmin, ctx=self)
                        else:
                            # subnormal
                            c = mbits
                            return Float(s=s, c=c, exp=self.expmin, ctx=self)
                    else:
                        # normal number
                        c = (1 << self.m) | mbits
                        exp = self.expmin + (ebits - 1)
                        return Float(s=s, c=c, exp=exp, ctx=self)

            case _:
                # this should never happen
                raise RuntimeError(f'unexpected NaN kind {self.nan_kind}')


def _format_is_valid(
    es: int,
    nbits: int,
    enable_inf: bool,
    nan_kind: EFloatNanKind,
):
    """Returns True if the `ExtFloatContext` format is valid."""
    # condition (i): positive bitwidth
    if nbits < 1:
        return False
    # condition (ii): non-negative exponent size (room for sign bit)
    if es < 0 or es >= nbits:
        return False

    # condition (iii): must encode 0
    # condition (iv): if nan_kind is IEEE 754, then special values
    #                 are only in the last binade
    p = nbits - es
    match nan_kind:
        case EFloatNanKind.IEEE_754:
            if es == 0:
                # would only encode special values: violates (iii)
                return False
            if enable_inf and p == 1:
                # no room for Inf _and_ NaN: violates (iv)
                return False

        case EFloatNanKind.MAX_VAL:
            if es == 0:
                if p == 1:
                    # only value is NaN: violates (iii)
                    return False
                elif enable_inf and p == 2:
                    # only values are Inf and NaN: violates (iii)
                    return False
            elif es == 1:
                if enable_inf and p == 1:
                    # Inf pushed to lower binade (only other one)
                    # but then only values are Inf and NaN: violates (iii)
                    return False

        case EFloatNanKind.NEG_ZERO | EFloatNanKind.NONE:
            if es == 0 and p == 1 and enable_inf:
                # no room for Inf and NaN: violates (iii)
                return False

        case _:
            raise RuntimeError(f'unexpected NaN kind{nan_kind}')

    return True


def _ext_to_mpb(
    es: int,
    nbits: int,
    enable_inf: bool,
    nan_kind: EFloatNanKind,
    eoffset: int,
    rm: RoundingMode,
    overflow: OverflowMode,
    num_randbits: int | None = 0,
) -> MPBFloatContext:
    """Converts between `ExtFloatContext` and `MPBFloatContext` parameters."""
    # IEEE 754 derived parameters
    p = nbits - es
    emax_0 = 0 if es == 0 else bitmask(es - 1)
    emin_0 = 1 - emax_0

    # apply exponent offset to compute final exponent parameters
    emax = emax_0 + eoffset
    emin = emin_0 + eoffset
    expmax = emax - p + 1
    expmin = emin - p + 1

    # compute the maximum value
    # the maximum value is encoding-dependent since depending
    # since the number of available numerical values depends
    # on the presence of infinities and the number of NaNs
    match nan_kind:
        case EFloatNanKind.IEEE_754:
            # the maximum value is the usual IEEE 754 maximum value
            maxval = RealFloat(c=bitmask(p), exp=expmax)

        case EFloatNanKind.MAX_VAL:
            # there is only 1 NaN and a possible infinity
            # the maximum value is just the "previous" encoding
            if p == 1:
                if enable_inf:
                    # NaN is in the last binade, Inf is in the previous,
                    # and the maximum value before that (1 below the usual IEEE 754 exponent)
                    maxval = RealFloat(c=bitmask(p), exp=expmax - 1)
                else:
                    # NaN is in the last binade, and the maximum value is
                    # the usual IEEE 754 maximum value
                    maxval = RealFloat(c=bitmask(p), exp=expmax)
            elif p == 2 and enable_inf:
                # NaN and Inf occupy the last binade,
                # so the maximum value is the usual IEEE 754 maximum value
                maxval = RealFloat(c=bitmask(p), exp=expmax)
            else:
                # maximum value is in the last binade
                if enable_inf:
                    # Inf and NaN occupy two values in the last binade
                    c = bitmask(p) - 2
                    maxval = RealFloat(c=c, exp=expmax+1)
                else:
                    # only NaN occupies the last binade
                    c = bitmask(p) - 1
                    maxval = RealFloat(c=c, exp=expmax+1)

        case EFloatNanKind.NEG_ZERO | EFloatNanKind.NONE:
            # NaN replaces -0 (or no NaN), there may be a possible infinity
            # the maximum value is either the maximum encoding or
            # the "previous" encoding before infinity
            if p == 1 and enable_inf:
                # Inf occupies the last binade, so the maximum value is
                # the usual IEEE 754 maximum value
                maxval = RealFloat(c=bitmask(p), exp=expmax)
            elif enable_inf:
                # Inf is the maximum encoding, so the maximum value is
                # the "previous" encoding before infinity
                c = bitmask(p) - 1
                maxval = RealFloat(c=c, exp=expmax+1)
            else:
                # the maximum value is the maximum encoding
                maxval = RealFloat(c=bitmask(p), exp=expmax+1)

        case _:
            raise RuntimeError(f'unexpected NaN kind {nan_kind}')

    # for es == 0, the exponent may be too low
    # we should normalize and ensure the exponent is in range
    maxval = maxval.normalize(p, expmin - 1)

    # create the related MPB context
    return MPBFloatContext(p, emin, maxval, rm, overflow, num_randbits)
