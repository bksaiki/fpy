"""
This module implements exponential numbers, i.e.., numbers equal to
2^k where k is an integer.

TODO: stochastic rounding
"""

from fractions import Fraction

from ..utils import DEFAULT, DefaultOr, bitmask, default_repr
from .context import EncodableContext, Context
from .gmp import mpfr_value
from .mp_float import MPFloatContext
from .round import OverflowMode, RoundingMode, RoundingDirection
from .number import Float, RealFloat

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

def _compute_params(
    nbits: int,
    eoffset: int,
    rm: RoundingMode,
    overflow: OverflowMode,
    inf_value: Float | None
):
    if not isinstance(nbits, int):
        raise TypeError(f'Expected \'int\', got \'{type(nbits)}\' for nbits={nbits}')
    if not isinstance(eoffset, int):
        raise TypeError(f'Expected \'int\', got \'{type(eoffset)}\' for eoffset={eoffset}')
    if not isinstance(rm, RoundingMode):
        raise TypeError(f'Expected \'RoundingMode\', got \'{type(rm)}\' for rm={rm}')
    if not isinstance(overflow, OverflowMode):
        raise TypeError(f'Expected \'OverflowMode\', got \'{type(overflow)}\' for overflow={overflow}')
    if inf_value is not None and not isinstance(inf_value, Float):
        raise TypeError(f'Expected \'Float\' or None, got \'{type(inf_value)}\' for inf_value={inf_value}')
    if nbits <= 0:
        raise ValueError(f'nbits must be positive, got nbits={nbits}')
    if overflow not in [OverflowMode.OVERFLOW, OverflowMode.SATURATE]:
        raise ValueError(f'overflow must be one of {OverflowMode.OVERFLOW}, {OverflowMode.SATURATE}, got {overflow}')

    emin, emax = _exponent_bounds(nbits, eoffset)
    mp_ctx = MPFloatContext(1, rm=rm) # 1 bit of mantissa

    if inf_value is not None:
        if not mp_ctx.representable_under(inf_value):
            raise ValueError(f'inf_value={inf_value} is not representable')
        if not inf_value.is_positive():
            raise ValueError(f'inf_value={inf_value} must be positive')
        if inf_value.e < emin or inf_value.e > emax:
            raise ValueError(f'inf_value={inf_value} must be within exponent bounds [{emin}, {emax}]')
        # ensure that p=1
        inf_value = mp_ctx.normalize(inf_value)

    return emin, emax, mp_ctx, inf_value


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

    _mp_ctx: MPFloatContext
    """this context without exponent bounds"""

    _emin: int
    """minimum exponent value"""

    _emax: int
    """maximum exponent value"""

    def __init__(
        self,
        nbits: int,
        eoffset: int = 0,
        rm: RoundingMode = RoundingMode.RNE,
        overflow: OverflowMode = OverflowMode.OVERFLOW,
        *,
        inf_value: Float | None = None
    ):
        emin, emax, mp_ctx, inf_value = _compute_params(nbits, eoffset, rm, overflow, inf_value)
        self.nbits = nbits
        self.eoffset = eoffset
        self.rm = rm
        self.overflow = overflow
        self.inf_value = inf_value
        self._mp_ctx = mp_ctx
        self._emin = emin
        self._emax = emax

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, ExpContext) and
            self.nbits == other.nbits and
            self.eoffset == other.eoffset and
            self.rm == other.rm and
            self.overflow == other.overflow and
            self.inf_value == other.inf_value
        )

    def __hash__(self):
        return hash((self.nbits, self.eoffset, self.rm, self.overflow, self.inf_value))

    @property
    def pmax(self) -> int:
        """maximum allowable precision."""
        return 1

    @property
    def emin(self) -> int:
        """The minimum representable exponent value"""
        return self._emin

    @property
    def emax(self) -> int:
        """The maximum representable exponent value"""
        return self._emax

    @property
    def ebias(self) -> int:
        """The exponent bias"""
        return bitmask(self.nbits - 1) - self.eoffset

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
        return ExpContext(
            nbits,
            eoffset,
            rm,
            overflow,
            inf_value=inf_value
        )

    def is_stochastic(self) -> bool:
        return False

    def is_equiv(self, other: Context) -> bool:
        if not isinstance(other, Context):
            raise TypeError(f'Expected \'Context\', got \'{type(other)}\' for other={other}')
        return (
            isinstance(other, ExpContext) and
            self.nbits == other.nbits and
            self.eoffset == other.eoffset
        )

    def representable_under(self, x: Float | RealFloat) -> bool:
        match x:
            case Float():
                if x.ctx is not None and self.is_equiv(x.ctx):
                    # same context, so representable
                    return True
                elif x.isnan:
                    # NaN is representable
                    return True
                elif x.isinf:
                    # Inf is not representable
                    return False
            case RealFloat():
                pass
            case _:
                raise TypeError(f'Expected \'Float\' or \'RealFloat\', got \'{type(x)}\' for x={x}')

        if not self._mp_ctx.representable_under(x):
            return False
        elif not x.is_positive():
            # must be positive
            return False
        elif x.e < self.emin or x.e > self.emax:
            # must be within exponent bounds
            return False
        else:
            # otherwise, it is representable
            return True

    def canonical_under(self, x: Float) -> bool:
        if not isinstance(x, Float):
            raise TypeError(f'Expected a representable \'Float\', got \'{type(x)}\' for x={x}')
        if not self.representable_under(x):
            raise ValueError(f'not representable under this context: x={x}, ctx={self}')
        return self._mp_ctx.canonical_under(x)

    def normal_under(self, x: Float) -> bool:
        if not isinstance(x, Float):
            raise TypeError(f'Expected a representable \'Float\', got \'{type(x)}\' for x={x}')
        if not self.representable_under(x):
            raise ValueError(f'not representable under this context: x={x}, ctx={self}')
        return self._mp_ctx.normal_under(x)

    def normalize(self, x: Float) -> Float:
        if not isinstance(x, Float):
            raise TypeError(f'Expected a representable \'Float\', got \'{type(x)}\' for x={x}')
        if not self.representable_under(x):
            raise ValueError(f'not representable under this context: x={x}, ctx={self}')
        return self._mp_ctx.normalize(x)

    def round_params(self) -> tuple[int | None, int | None]:
        return 1, None

    # TODO: copied from mpb_float.py, should be refactored
    def _overflow_to_infinity(self, s: bool):
        """Should overflows round to infinity (rather than MAX_VAL)?"""
        _, direction = self.rm.to_direction(s)
        match direction:
            case RoundingDirection.RTZ:
                # always round towards zero
                return False
            case RoundingDirection.RAZ:
                # always round towards infinity
                return True
            case RoundingDirection.RTE:
                # infinity is considered even for rounding
                return True
            case RoundingDirection.RTO:
                # infinity is considered even for rounding
                return False
            case _:
                raise RuntimeError(f'unrechable {direction}')
    
    def _underflow_to_zero(self, s: bool):
        """
        Should underflows round to zero (rather than MIN_VAL)?

        This is unlike IEEE 754 with _gradual_ underflow.
        For exponent numbers, the region between 0 and MIN_VAL is similar
        to the region between MAX_VAL and infinity:

        Let ZERO_VAL be the next representable value below MIN_VAL assuming
        no exponent bounds. If we round to ZERO_VAL, then we should round to 0,
        or rather NaN, since 0 is not representable.
        """
        _, direction = self.rm.to_direction(s)
        match direction:
            case RoundingDirection.RTZ:
                # always round towards zero
                return True
            case RoundingDirection.RAZ:
                # always round towards infinity
                return False
            case RoundingDirection.RTE:
                # zero is considered even for rounding
                return True
            case RoundingDirection.RTO:
                # zero is considered even for rounding
                return False

    def _round_at(self, x: RealFloat | Float, n: int | None, exact: bool) -> Float:
        """
        Like `self.round()` but for only `RealFloat` and `Float` inputs.

        Optionally specify `n` as the least absolute digit position.
        Only overrides rounding behavior when `n > self.nmin`.
        """
        # step 1. round with no exponent bound
        rounded = self._mp_ctx._round_at(x, n, exact)

        # step 2. check for special values
        if rounded.isnan:
            # always return +NaN
            return Float.nan(ctx=self)
        elif rounded.isinf:
            # Inf is not representable
            if self.inf_value is None:
                # default behavior is to return +NaN
                return Float.nan(ctx=self)
            else:
                # return the specified Inf value
                return Float(x=self.inf_value, ctx=self)
        elif rounded.is_negative():
            # negative values are not representable
            return Float.nan(ctx=self)
        elif rounded.is_zero():
            # zero is not representable
            return Float.nan(ctx=self)

        # step 3. check for overflow (or underflow)
        if rounded.e < self.emin:
            # check if underflow is allowed
            if exact:
                raise ValueError(f'Rounding {rounded} under self={self} with n={n} would overflow')
            # use the overflow behavior to determine underflow
            # since exponent numbers are symmetric about +1
            match self.overflow:
                case OverflowMode.OVERFLOW:
                    # check which way to round
                    if self._underflow_to_zero(rounded.s):
                        # round to zero => not representable, so return +NaN
                        return Float.nan(ctx=self)
                    else:
                        # round to +MIN_VAL
                        return self.minval()
                case OverflowMode.SATURATE:
                    # clamp to +MIN_VAL
                    return self.minval()
                case _:
                    raise RuntimeError(f'unreachable: {self.overflow}')
        elif rounded.e > self.emax:
            # check if underflow is allowed
            if exact:
                raise ValueError(f'Rounding {rounded} under self={self} with n={n} would overflow')

            # case split on overflow behavior
            match self.overflow:
                case OverflowMode.OVERFLOW:
                    # check which way to round
                    if self._overflow_to_infinity(rounded.s):
                        # round to +Inf => not representable, so return +NaN
                        return Float.nan(ctx=self)
                    else:
                        # round to +MAX_VAL
                        return self.maxval()
                case OverflowMode.SATURATE:
                    # clamp to +MAX_VAL
                    return self.maxval()
                case _:
                    raise RuntimeError(f'unreachable: {self.overflow}')

        # step 4. return rounded result
        return Float(x=rounded, ctx=self)

    def round(self, x, *, exact: bool = False) -> Float:
        x = self._round_prepare(x)
        return self._round_at(x, None, exact)

    def round_at(self, x, n: int, *, exact: bool = False) -> Float:
        x = self._round_prepare(x)
        return self._round_at(x, n, exact)

    def to_ordinal(self, x: Float, infval: bool = False) -> int:
        if not isinstance(x, Float):
            raise TypeError(f'Expected a representable \'Float\', got \'{type(x)}\' for x={x}')
        if not self.representable_under(x):
            raise ValueError(f'not representable under this context: x={x}, ctx={self}')
        if infval:
            raise ValueError(f'infval={infval} is not supported for ExpContext')
        if x.isnan:
            # NaN
            raise ValueError(f'can only convert a finite value x={x}')
        return self.encode(x)

    def from_ordinal(self, x: int, infval: bool = False) -> Float:
        if not isinstance(x, int):
            raise TypeError(f'Expected an integer, got \'{type(x)}\' for x={x}')
        if infval:
            raise ValueError(f'infval={infval} is not supported for ExpContext')
        if x < 0 or x >= bitmask(self.nbits):
            raise ValueError(f'x={x} must be on [0, {bitmask(self.nbits)})')
        return self.decode(x)

    def minval(self, s: bool = False) -> Float:
        if s:
            raise ValueError('negative values are not representable')
        return Float(c=1, exp=self.emin, ctx=self)

    def maxval(self, s: bool = False) -> Float:
        if s:
            raise ValueError('negative values are not representable')
        return Float(c=1, exp=self.emax, ctx=self)

    def encode(self, x: Float) -> int:
        if not isinstance(x, Float):
            raise TypeError(f'Expected a representable \'Float\', got \'{type(x)}\' for x={x}')
        if not self.representable_under(x):
            raise ValueError(f'not representable under this context: x={x}, ctx={self}')

        if x.isnan:
            # NaN: all ones
            return bitmask(self.nbits)
        else:
            # positive value
            return x.e + self.ebias

    def decode(self, x: int) -> Float:
        if not isinstance(x, int):
            raise TypeError(f'Expected an integer, got \'{type(x)}\' for x={x}')
        if x < 0 or x >= 1 << self.nbits:
            raise ValueError(f'x={x} must be on [0, {1 << self.nbits})')

        if x == bitmask(self.nbits):
            # NaN: all ones
            return Float.nan(ctx=self)
        else:
            # positive value
            exp = x - self.ebias
            return Float(c=1, exp=exp, ctx=self)
