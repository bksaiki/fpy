"""
This module implements exponential numbers, i.e.., numbers equal to
2^k where k is an integer.

TODO: stochastic rounding
"""

from fractions import Fraction

from ..number import Float, RealFloat
from ..round import OverflowMode, RoundingMode, RoundingDirection
from ...utils import DEFAULT, DefaultOr, bitmask, default_repr

from .context import EncodableContext
from .format import EncodableFormat
from .mp_float import MPFloatFormat


def _exponent_bounds(nbits: int, eoffset: int) -> tuple[int, int]:
    """
    Calculate the minimum and maximum exponent bounds for the given
    number of bits and exponent offset.
    """
    es = nbits
    emax_0 = bitmask(es - 1)
    emin_0 = 1 - emax_0

    emax = emax_0 + eoffset
    emin = emin_0 + eoffset - 1  # subnormal exponent is just below the min normal

    return emin, emax


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

    _emin: int
    _emax: int
    _mp_fmt: MPFloatFormat

    def __init__(self, nbits: int, eoffset: int = 0):
        if not isinstance(nbits, int):
            raise TypeError(f'Expected \'int\' for nbits={nbits}, got {type(nbits)}')
        if nbits <= 0:
            raise ValueError(f'nbits must be positive, got nbits={nbits}')
        if not isinstance(eoffset, int):
            raise TypeError(f'Expected \'int\' for eoffset={eoffset}, got {type(eoffset)}')
        self.nbits = nbits
        self.eoffset = eoffset
        self._emin, self._emax = _exponent_bounds(nbits, eoffset)
        self._mp_fmt = MPFloatFormat(1)

    def __eq__(self, other):
        return (
            isinstance(other, ExpFormat)
            and self.nbits == other.nbits
            and self.eoffset == other.eoffset
        )

    def __hash__(self):
        return hash((self.__class__, self.nbits, self.eoffset))

    @property
    def pmax(self) -> int:
        """Maximum allowable precision."""
        return 1

    @property
    def emin(self) -> int:
        return self._emin

    @property
    def emax(self) -> int:
        return self._emax

    @property
    def ebias(self) -> int:
        return bitmask(self.nbits - 1) - self.eoffset

    def representable_in(self, x: Float | RealFloat) -> bool:
        match x:
            case Float():
                if x.isnan:
                    return True
                if x.isinf:
                    return False
            case RealFloat():
                pass
            case _:
                raise TypeError(f'Expected \'Float\' or \'RealFloat\', got \'{type(x)}\' for x={x}')

        if not self._mp_fmt.representable_in(x):
            return False
        if not x.is_positive():
            return False
        if x.e < self.emin or x.e > self.emax:
            return False
        return True

    def canonical_under(self, x: Float) -> bool:
        if not isinstance(x, Float):
            raise TypeError(f'Expected a representable \'Float\', got \'{type(x)}\' for x={x}')
        if not self.representable_in(x):
            raise ValueError(f'not representable under this format: x={x}')
        return self._mp_fmt.canonical_under(x)

    def normal_under(self, x: Float) -> bool:
        if not isinstance(x, Float):
            raise TypeError(f'Expected a representable \'Float\', got \'{type(x)}\' for x={x}')
        if not self.representable_in(x):
            raise ValueError(f'not representable under this format: x={x}')
        return self._mp_fmt.normal_under(x)

    def normalize(self, x: Float) -> Float:
        if not isinstance(x, Float):
            raise TypeError(f'Expected a representable \'Float\', got \'{type(x)}\' for x={x}')
        if not self.representable_in(x):
            raise ValueError(f'not representable under this format: x={x}')
        return self._mp_fmt.normalize(x)

    def _to_ordinal(self, x: RealFloat | Float) -> int:
        if isinstance(x, Float) and x.isnan:
            raise ValueError(f'can only convert a finite value x={x}')
        return x.e + self.ebias

    def to_ordinal(self, x: Float, infval: bool = False) -> int:
        if not isinstance(x, Float):
            raise TypeError(f'Expected a representable \'Float\', got \'{type(x)}\' for x={x}')
        if not self.representable_in(x):
            raise ValueError(f'not representable under this format: x={x}')
        if infval:
            raise ValueError(f'infval={infval} is not supported for ExpFormat')
        if x.isnan:
            raise ValueError(f'can only convert a finite value x={x}')
        return self._to_ordinal(x)

    def to_fractional_ordinal(self, x: Float) -> Fraction:
        if not isinstance(x, Float):
            raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
        if x.is_nar():
            raise ValueError(f'Expected a finite value for x={x}')

        if self.representable_in(x):
            return Fraction(self.to_ordinal(x))

        xr = x.as_real()
        above = xr.round(self.pmax, rm=RoundingMode.RTP)
        below = xr.round(self.pmax, rm=RoundingMode.RTN)
        if above == below:
            return Fraction(self._to_ordinal(above))

        delta_x: RealFloat = xr - below
        delta: RealFloat = above - below
        t = delta_x.as_rational() / delta.as_rational()

        below_ord = self._to_ordinal(below)
        return Fraction(below_ord) + t

    def from_ordinal(self, x: int, infval: bool = False) -> Float:
        if not isinstance(x, int):
            raise TypeError(f'Expected an integer, got \'{type(x)}\' for x={x}')
        if infval:
            raise ValueError(f'infval={infval} is not supported for ExpFormat')
        if x < 0 or x >= bitmask(self.nbits):
            raise ValueError(f'x={x} must be on [0, {bitmask(self.nbits)})')
        return self.decode(x)

    def minval(self, s: bool = False) -> Float:
        if s:
            raise ValueError('negative values are not representable')
        return Float(c=1, exp=self.emin)

    def maxval(self, s: bool = False) -> Float:
        if s:
            raise ValueError('negative values are not representable')
        return Float(c=1, exp=self.emax)

    def infval(self, s: bool = False) -> Float:
        if not isinstance(s, bool):
            raise TypeError(f'Expected \'bool\' for s={s}, got {type(s)}')
        if s:
            raise ValueError('negative values are not representable')
        maxval = self.maxval()._real
        infval = maxval.next_away_zero(p=self.pmax)
        return Float.from_real(infval)

    def largest(self) -> Float:
        return self.maxval()

    def smallest(self) -> Float:
        return self.minval()

    def total_bits(self) -> int:
        return self.nbits

    def encode(self, x: Float) -> int:
        if not isinstance(x, Float):
            raise TypeError(f'Expected a representable \'Float\', got \'{type(x)}\' for x={x}')
        if not self.representable_in(x):
            raise ValueError(f'not representable under this format: x={x}')

        if x.isnan:
            return bitmask(self.nbits)
        return x.e + self.ebias

    def decode(self, x: int) -> Float:
        if not isinstance(x, int):
            raise TypeError(f'Expected an integer, got \'{type(x)}\' for x={x}')
        if x < 0 or x >= 1 << self.nbits:
            raise ValueError(f'x={x} must be on [0, {1 << self.nbits})')

        if x == bitmask(self.nbits):
            return Float.nan()
        return Float(c=1, exp=x - self.ebias)


@default_repr
class ExpContext(EncodableContext):
    """
    Rounding context for exponential numbers, i.e., numbers
    of the form 2^k where k is an integer. The context is parameterized
    by the size of the representation `nbits`, an exponent offset `eoffset`,
    and the rounding mode `rm`.

    This context implements `EncodableContext`.
    The special value NaN is representable and is encoded as all ones.
    """

    nbits: int
    """size of the representation in bits"""

    eoffset: int
    """exponent offset"""

    rm: RoundingMode
    """rounding mode"""

    overflow: OverflowMode
    """overflow mode"""

    inf_value: Float | None
    """
    if Inf is not representable, what value should Inf round to?
    if not set, then `round()` will produce NaN.
    """

    _fmt: ExpFormat

    def __init__(
        self,
        nbits: int,
        eoffset: int = 0,
        rm: RoundingMode = RoundingMode.RNE,
        overflow: OverflowMode = OverflowMode.OVERFLOW,
        *,
        inf_value: Float | None = None
    ):
        if not isinstance(rm, RoundingMode):
            raise TypeError(f'Expected \'RoundingMode\', got \'{type(rm)}\' for rm={rm}')
        if not isinstance(overflow, OverflowMode):
            raise TypeError(f'Expected \'OverflowMode\', got \'{type(overflow)}\' for overflow={overflow}')
        if overflow not in (OverflowMode.OVERFLOW, OverflowMode.SATURATE):
            raise ValueError(f'overflow must be one of {OverflowMode.OVERFLOW}, {OverflowMode.SATURATE}, got {overflow}')
        if inf_value is not None and not isinstance(inf_value, Float):
            raise TypeError(f'Expected \'Float\' or None, got \'{type(inf_value)}\' for inf_value={inf_value}')

        self._fmt = ExpFormat(nbits, eoffset)

        if inf_value is not None:
            if not self._fmt._mp_fmt.representable_in(inf_value):
                raise ValueError(f'inf_value={inf_value} is not representable')
            if not inf_value.is_positive():
                raise ValueError(f'inf_value={inf_value} must be positive')
            if inf_value.e < self._fmt.emin or inf_value.e > self._fmt.emax:
                raise ValueError(f'inf_value={inf_value} must be within exponent bounds [{self._fmt.emin}, {self._fmt.emax}]')
            inf_value = self._fmt._mp_fmt.normalize(inf_value)

        self.nbits = nbits
        self.eoffset = eoffset
        self.rm = rm
        self.overflow = overflow
        self.inf_value = inf_value

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, ExpContext)
            and self.nbits == other.nbits
            and self.eoffset == other.eoffset
            and self.rm == other.rm
            and self.overflow == other.overflow
            and self.inf_value == other.inf_value
        )

    def __hash__(self):
        return hash((self.nbits, self.eoffset, self.rm, self.overflow, self.inf_value))

    @property
    def pmax(self) -> int:
        return 1

    @property
    def emin(self) -> int:
        return self._fmt.emin

    @property
    def emax(self) -> int:
        return self._fmt.emax

    @property
    def ebias(self) -> int:
        return self._fmt.ebias

    def with_params(
        self, *,
        nbits: DefaultOr[int] = DEFAULT,
        eoffset: DefaultOr[int] = DEFAULT,
        rm: DefaultOr[RoundingMode] = DEFAULT,
        overflow: DefaultOr[OverflowMode] = DEFAULT,
        inf_value: DefaultOr[Float | None] = DEFAULT,
        **kwargs
    ) -> 'ExpContext':
        if nbits is DEFAULT:
            nbits = self.nbits
        if eoffset is DEFAULT:
            eoffset = self.eoffset
        if rm is DEFAULT:
            rm = self.rm
        if overflow is DEFAULT:
            overflow = self.overflow
        if inf_value is DEFAULT:
            inf_value = self.inf_value
        if kwargs:
            raise TypeError(f'Unexpected parameters {kwargs} for ExpContext')

        return ExpContext(nbits, eoffset, rm, overflow, inf_value=inf_value)

    def is_stochastic(self) -> bool:
        return False

    def format(self) -> ExpFormat:
        return self._fmt

    @classmethod
    def from_format(
        cls,
        fmt: ExpFormat,
        *,
        rm: RoundingMode = RoundingMode.RNE,
        overflow: OverflowMode | None = None,
        inf_value: Float | None = None
    ) -> 'ExpContext':
        """Creates a context from an `ExpFormat` and rounding parameters."""
        if not isinstance(fmt, ExpFormat):
            raise TypeError(f'Expected \'ExpFormat\', got {type(fmt)}')
        if overflow is None:
            overflow = OverflowMode.OVERFLOW
        return cls(fmt.nbits, fmt.eoffset, rm, overflow, inf_value=inf_value)

    def round_params(self) -> tuple[int | None, int | None]:
        return 1, None

    def _overflow_to_infinity(self, s: bool):
        """Should overflows round to infinity (rather than MAX_VAL)?"""
        _, direction = self.rm.to_direction(s)
        match direction:
            case RoundingDirection.RTZ:
                return False
            case RoundingDirection.RAZ:
                return True
            case RoundingDirection.RTE:
                return True
            case RoundingDirection.RTO:
                return False
            case _:
                raise RuntimeError(f'unrechable {direction}')

    def _underflow_to_zero(self, s: bool):
        """
        Should underflows round to zero (rather than MIN_VAL)?

        For exponent numbers, the region between 0 and MIN_VAL is similar
        to the region between MAX_VAL and infinity.
        """
        _, direction = self.rm.to_direction(s)
        match direction:
            case RoundingDirection.RTZ:
                return True
            case RoundingDirection.RAZ:
                return False
            case RoundingDirection.RTE:
                return True
            case RoundingDirection.RTO:
                return False

    def _round_at(self, x: RealFloat | Float, n: int | None, exact: bool) -> Float:
        """
        Like `self.round()` but for only `RealFloat` and `Float` inputs.
        """
        # round with no exponent bound using a 1-bit MPFloat context
        from .mp_float import MPFloatContext
        mp_ctx = MPFloatContext(1, rm=self.rm)
        rounded = mp_ctx._round_at(x, n, exact)

        if rounded.isnan:
            return Float.nan(ctx=self)
        elif rounded.isinf:
            if self.inf_value is None:
                return Float.nan(ctx=self)
            return Float(x=self.inf_value, ctx=self)
        elif rounded.is_negative():
            return Float.nan(ctx=self)
        elif rounded.is_zero():
            return Float.nan(ctx=self)

        if rounded.e < self.emin:
            if exact:
                raise ValueError(f'Rounding {rounded} under self={self} with n={n} would overflow')
            match self.overflow:
                case OverflowMode.OVERFLOW:
                    if self._underflow_to_zero(rounded.s):
                        result = Float.nan(ctx=self)
                    else:
                        result = self.minval()
                case OverflowMode.SATURATE:
                    result = self.minval()
                case OverflowMode.ASSERT:
                    raise ValueError(f'Rounding {rounded} under self={self} with n={n} would underflow')
                case _:
                    raise RuntimeError(f'unreachable: {self.overflow}')

            result._real._flags._set_overflow(True)
            result._real._flags._set_inexact(True)
            return result

        elif rounded.e > self.emax:
            if exact:
                raise ValueError(f'Rounding {rounded} under self={self} with n={n} would overflow')
            match self.overflow:
                case OverflowMode.OVERFLOW:
                    if self._overflow_to_infinity(rounded.s):
                        return Float.nan(ctx=self)
                    return self.maxval()
                case OverflowMode.SATURATE:
                    return self.maxval()
                case OverflowMode.ASSERT:
                    raise ValueError(f'Rounding {rounded} under self={self} with n={n} would overflow')
                case _:
                    raise RuntimeError(f'unreachable: {self.overflow}')

        return Float(x=rounded, ctx=self)

    def round(self, x, *, exact: bool = False) -> Float:
        x = self._round_prepare(x)
        return self._round_at(x, None, exact)

    def round_at(self, x, n: int, *, exact: bool = False) -> Float:
        x = self._round_prepare(x)
        return self._round_at(x, n, exact)
