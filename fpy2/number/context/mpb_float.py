"""
This module defines floating-point numbers as implemented by MPFR
but with subnormalization and a maximum value, that is multi-precision
and bounded. Hence, "MP-B."
"""

from ...utils import DEFAULT, DefaultOr, default_repr
from ..number import RNG, Float, RealFloat, same_value
from ..round import OverflowMode, RoundingDirection, RoundingMode
from .context import SizedContext, zero_sign
from .format import SizedFormat
from .mps_float import MPSFloatFormat


@default_repr
class MPBFloatFormat(SizedFormat):
    """
    Number format for multi-precision, bounded, floating-point numbers.

    This format is parameterized by a fixed precision `pmax`, minimum
    normalized exponent `emin`, and positive maximum value `pos_maxval`.
    It describes the set of representable values for `MPBFloatContext`.

    `enable_neg_zero` says whether `-0.0` is one of them.
    """

    pmax: int
    """maximum precision"""

    emin: int
    """minimum (normalized) exponent"""

    pos_maxval: RealFloat
    """positive maximum value"""

    neg_maxval: RealFloat
    """negative maximum value"""

    _mps_fmt: MPSFloatFormat
    """underlying unbounded format"""

    _pos_maxval_ord: int
    """precomputed ordinal of `self.pos_maxval`"""

    _neg_maxval_ord: int
    """precomputed ordinal of `self.neg_maxval`"""

    enable_nan: bool
    """whether NaN is representable"""

    enable_inf: bool
    """whether (signed) infinity is representable"""

    enable_neg_zero: bool
    """whether `-0.0` is representable"""

    def __init__(
        self,
        pmax: int,
        emin: int,
        pos_maxval: RealFloat,
        neg_maxval: RealFloat | None = None,
        enable_nan: bool = True,
        enable_inf: bool = True,
        enable_neg_zero: bool = True,
    ):
        if not isinstance(pmax, int):
            raise TypeError(f'Expected \'int\' for pmax={pmax}, got {type(pmax)}')
        if pmax < 1:
            raise ValueError(f'Expected positive integer for pmax={pmax}')
        if not isinstance(emin, int):
            raise TypeError(f'Expected \'int\' for emin={emin}, got {type(emin)}')
        if not isinstance(pos_maxval, RealFloat):
            raise TypeError(f'Expected \'RealFloat\' for pos_maxval={pos_maxval}, got {type(pos_maxval)}')
        if pos_maxval.s:
            raise ValueError(f'Expected positive pos_maxval={pos_maxval}')

        if neg_maxval is None:
            neg_maxval = RealFloat(s=True, x=pos_maxval)
        elif not isinstance(neg_maxval, RealFloat):
            raise TypeError(f'Expected \'RealFloat\' for neg_maxval={neg_maxval}, got {type(neg_maxval)}')
        elif not neg_maxval.s:
            raise ValueError(f'Expected negative neg_maxval={neg_maxval}')
        if not isinstance(enable_neg_zero, bool):
            raise TypeError(f'Expected \'bool\' for enable_neg_zero={enable_neg_zero}, got {type(enable_neg_zero)}')

        self.pmax = pmax
        self.emin = emin
        self.pos_maxval = pos_maxval
        self.neg_maxval = neg_maxval
        self.enable_nan = enable_nan
        self.enable_inf = enable_inf
        self.enable_neg_zero = enable_neg_zero

        # `representable_in` and `zero` both go through the unbounded format, so
        # carrying the flag there is what makes them honour it
        self._mps_fmt = MPSFloatFormat(
            pmax, emin, enable_nan, enable_inf, enable_neg_zero,
        )
        self._pos_maxval_ord = self._mps_fmt._to_ordinal(pos_maxval)
        self._neg_maxval_ord = self._mps_fmt._to_ordinal(neg_maxval)

    def __eq__(self, other):
        return (
            isinstance(other, MPBFloatFormat)
            and self.pmax == other.pmax
            and self.emin == other.emin
            and self.pos_maxval == other.pos_maxval
            and self.neg_maxval == other.neg_maxval
            and self.enable_nan == other.enable_nan
            and self.enable_inf == other.enable_inf
            and self.enable_neg_zero == other.enable_neg_zero
        )

    def __hash__(self):
        return hash((
            self.__class__, self.pmax, self.emin, self.pos_maxval, self.neg_maxval,
            self.enable_nan, self.enable_inf, self.enable_neg_zero,
        ))

    @property
    def expmin(self) -> int:
        """Minimum unnormalized exponent."""
        return self.emin - self.pmax + 1

    @property
    def nmin(self) -> int:
        """First unrepresentable digit for every value in the format."""
        return self.expmin - 1

    @property
    def emax(self) -> int:
        """Maximum normalized exponent."""
        pos_e = self.pos_maxval.e
        neg_e = self.neg_maxval.e
        return max(pos_e, neg_e)

    @property
    def expmax(self) -> int:
        """Maximum unnormalized exponent."""
        return self.emax - self.pmax + 1

    def representable_in(self, x: RealFloat | Float) -> bool:
        if isinstance(x, Float):
            if x.isnan:
                return self.enable_nan
            if x.isinf:
                return self.enable_inf
        if not self._mps_fmt.representable_in(x):
            return False
        if not x.is_nonzero():
            return True
        if x.s:
            return self.neg_maxval <= x
        return x <= self.pos_maxval

    def canonical_under(self, x: Float) -> bool:
        if not isinstance(x, Float) or not self.representable_in(x):
            raise TypeError(f'Expected a representable \'Float\', got \'{type(x)}\' for x={x}')
        return self._mps_fmt.canonical_under(x)

    def normal_under(self, x: Float) -> bool:
        if not isinstance(x, Float) or not self.representable_in(x):
            raise TypeError(f'Expected a representable \'Float\', got \'{type(x)}\' for x={x}')
        return self._mps_fmt.normal_under(x)

    def normalize(self, x: Float) -> Float:
        if not isinstance(x, Float) or not self.representable_in(x):
            raise TypeError(f'Expected a representable \'Float\', got \'{type(x)}\' for x={x}')
        return self._mps_fmt.normalize(x)

    def to_ordinal(self, x: Float, infval: bool = False) -> int:
        if not isinstance(x, Float):
            raise TypeError(f'Expected a \'Float\', got \'{type(x)}\' for x={x}')
        if not self.representable_in(x):
            raise ValueError(f'Expected a representable \'Float\', got x={x}')

        if x.isnan:
            raise TypeError(f'Expected a finite value for x={x}')
        elif x.isinf:
            if not infval:
                raise TypeError(f'Expected a finite value for x={x}')
            elif x.s:
                return self._neg_maxval_ord - 1
            else:
                return self._pos_maxval_ord + 1
        else:
            return self._mps_fmt.to_ordinal(x)

    def to_fractional_ordinal(self, x: Float):
        if not isinstance(x, Float):
            raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
        return self._mps_fmt.to_fractional_ordinal(x)

    def from_ordinal(self, x: int, infval: bool = False) -> Float:
        if not isinstance(x, int):
            raise TypeError(f'Expected an \'int\', got \'{type(x)}\' for x={x}')

        # sentinel ordinals map to infinity only when this format has it
        allow_inf = infval and self.enable_inf
        if x > self._pos_maxval_ord:
            if not allow_inf or x > self._pos_maxval_ord + 1:
                raise ValueError(f'Expected an \'int\' between {self._neg_maxval_ord} and {self._pos_maxval_ord}, got x={x}')
            return Float(isinf=True)
        elif x < self._neg_maxval_ord:
            if not allow_inf or x < self._neg_maxval_ord - 1:
                raise ValueError(f'Expected an \'int\' between {self._neg_maxval_ord} and {self._pos_maxval_ord}, got x={x}')
            return Float(s=True, isinf=True)
        else:
            return self._mps_fmt.from_ordinal(x)

    def zero(self, s: bool = False) -> Float:
        """Returns a signed 0 under this format, raising for `s=True` where it
        has no negative zero."""
        return self._mps_fmt.zero(s)

    def minval(self, s: bool = False) -> Float:
        """Returns the smallest non-zero value with sign `s` under this format."""
        return self._mps_fmt.minval(s)

    def min_subnormal(self, s: bool = False) -> Float:
        """Returns the smallest subnormal value with sign `s` under this format."""
        return self._mps_fmt.min_subnormal(s)

    def max_subnormal(self, s: bool = False) -> Float:
        """Returns the largest subnormal value with sign `s` under this format."""
        return self._mps_fmt.max_subnormal(s)

    def min_normal(self, s: bool = False) -> Float:
        """Returns the smallest normal value with sign `s` under this format."""
        return self._mps_fmt.min_normal(s)

    def max_normal(self, s: bool = False) -> Float:
        """Returns the largest normal value with sign `s` under this format."""
        if not isinstance(s, bool):
            raise TypeError(f'Expected \'bool\' for s={s}, got {type(s)}')
        return Float(x=self.neg_maxval) if s else Float(x=self.pos_maxval)

    def maxval(self, s: bool = False) -> Float:
        return self.max_normal(s)

    def infval(self, s: bool = False) -> Float:
        if not isinstance(s, bool):
            raise TypeError(f'Expected \'bool\' for s={s}, got {type(s)}')
        maxval = self.neg_maxval if s else self.pos_maxval
        infval = maxval.next_away_zero(p=self.pmax, n=self.nmin)
        return Float.from_real(infval)

    def largest(self) -> Float:
        return self.maxval(s=False)

    def smallest(self) -> Float:
        return self.maxval(s=True)


@default_repr
class MPBFloatContext(SizedContext):
    """
    Rounding context for multi-precision floating-point numbers with
    a minimum exponent (and subnormalization) and a maximum value.

    This context is parameterized by a fixed precision `pmax`,
    a minimum (normalized) exponent `emin`, a (positive) maximum value
    `maxval`, and a rounding mode `rm`. A separate negative maximum value
    may be specified as well, but by default it is set to the negative
    of `maxval`.

    Unlike `MPFloatContext`, the `MPBFloatContext` inherits from `SizedContext`
    since the set of representable values may be encoded in
    a finite amount of space.

    Optionally, specify the following keywords:

    - `enable_nan`: if `True`, then NaN is representable [default: `True`]
    - `enable_inf`: if `True`, then infinity is representable [default: `True`]
    - `enable_neg_zero`: if `True`, then `-0.0` is representable [default: `True`]
    - `nan_value`: if NaN is not enabled, what value should NaN round to? [default: `None`];
      if not set, then `round()` will raise a `ValueError` on NaN.
    - `inf_value`: if Inf is not enabled, what value should Inf round to? [default: `None`];
      if not set, then `round()` will raise a `ValueError` on infinity. An overflow
      that would round to infinity is substituted the same way.
    """

    pmax: int
    """maximum precision"""

    emin: int
    """minimum (normalized exponent)"""

    pos_maxval: RealFloat
    """positive maximum value"""

    neg_maxval: RealFloat
    """negative maximum value"""

    rm: RoundingMode
    """rounding mode"""

    overflow: OverflowMode
    """overflow behavior"""

    num_randbits: int | None
    """number of random bits for stochastic rounding, if applicable"""

    rng: RNG | None
    """random number generator for stochastic rounding, if applicable"""

    enable_nan: bool
    """is NaN representable?"""

    enable_inf: bool
    """is infinity representable?"""

    enable_neg_zero: bool
    """is `-0.0` representable?  When `False`, a negative value that rounds to
    zero gives `+0.0`, so a result is always in `format()`."""

    nan_value: Float | None
    """
    if NaN is not enabled, what value should NaN round to?
    if not set, then `round()` will raise a `ValueError`.
    """

    inf_value: Float | None
    """
    if Inf is not enabled, what value should Inf round to?
    if not set, then `round()` will raise a `ValueError`.
    """

    _fmt: MPBFloatFormat
    """precomputed format object"""

    def __init__(
        self,
        pmax: int,
        emin: int,
        maxval: RealFloat,
        rm: RoundingMode = RoundingMode.RNE,
        overflow: OverflowMode = OverflowMode.OVERFLOW,
        num_randbits: int | None = 0,
        *,
        neg_maxval: RealFloat | None = None,
        rng: RNG | None = None,
        enable_nan: bool = True,
        enable_inf: bool = True,
        enable_neg_zero: bool = True,
        nan_value: Float | None = None,
        inf_value: Float | None = None
    ):
        if not isinstance(rm, RoundingMode):
            raise TypeError(f'Expected \'RoundingMode\' for rm={rm}, got {type(rm)}')
        if not isinstance(overflow, OverflowMode):
            raise TypeError(f'Expected \'OverflowMode\' for overflow={overflow}, got {type(overflow)}')
        if num_randbits is not None and not isinstance(num_randbits, int):
            raise TypeError(f'Expected \'int\' for num_randbits={num_randbits}, got {type(num_randbits)}')
        if overflow == OverflowMode.WRAP:
            raise ValueError('OverflowMode.WRAP is not supported for MPBFloatContext')
        if not isinstance(enable_nan, bool):
            raise TypeError(f'Expected \'bool\' for enable_nan={enable_nan}, got {type(enable_nan)}')
        if not isinstance(enable_inf, bool):
            raise TypeError(f'Expected \'bool\' for enable_inf={enable_inf}, got {type(enable_inf)}')
        if not isinstance(enable_neg_zero, bool):
            raise TypeError(f'Expected \'bool\' for enable_neg_zero={enable_neg_zero}, got {type(enable_neg_zero)}')

        self._fmt = MPBFloatFormat(
            pmax, emin, maxval, neg_maxval, enable_nan, enable_inf,
            enable_neg_zero,
        )

        if nan_value is not None:
            if not isinstance(nan_value, Float):
                raise TypeError(f'Expected \'Float\' for nan_value={nan_value}, got {type(nan_value)}')
            if not enable_nan and not self._fmt.representable_in(nan_value):
                raise ValueError(f'Rounding NaN to unrepresentable value {nan_value}')

        if inf_value is not None:
            if not isinstance(inf_value, Float):
                raise TypeError(f'Expected \'Float\' for inf_value={inf_value}, got {type(inf_value)}')
            # the substitute takes the operand's sign, so both are reachable
            for signed in (Float(s=False, x=inf_value), Float(s=True, x=inf_value)):
                if not enable_inf and not self._fmt.representable_in(signed):
                    raise ValueError(f'Rounding Inf to unrepresentable value {signed}')

        self.pmax = pmax
        self.emin = emin
        self.pos_maxval = self._fmt.pos_maxval
        self.neg_maxval = self._fmt.neg_maxval
        self.rm = rm
        self.overflow = overflow
        self.num_randbits = num_randbits
        self.rng = rng
        self.enable_nan = enable_nan
        self.enable_inf = enable_inf
        self.enable_neg_zero = enable_neg_zero
        self.nan_value = nan_value
        self.inf_value = inf_value

    def __eq__(self, other):
        return (
            isinstance(other, MPBFloatContext)
            and self.pmax == other.pmax
            and self.emin == other.emin
            and self.pos_maxval == other.pos_maxval
            and self.neg_maxval == other.neg_maxval
            and self.rm == other.rm
            and self.overflow == other.overflow
            and self.num_randbits == other.num_randbits
            and self.enable_nan == other.enable_nan
            and self.enable_inf == other.enable_inf
            and self.enable_neg_zero == other.enable_neg_zero
            and same_value(self.nan_value, other.nan_value)
            and same_value(self.inf_value, other.inf_value)
        )

    def __hash__(self):
        return hash((
            self.pmax,
            self.emin,
            self.pos_maxval,
            self.neg_maxval,
            self.rm,
            self.overflow,
            self.num_randbits,
            self.enable_nan,
            self.enable_inf,
            self.nan_value,
            self.inf_value
        ))

    @property
    def emax(self):
        """Maximum normalized exponent."""
        return self._fmt.emax

    @property
    def expmax(self):
        """Maximum unnormalized exponent."""
        return self._fmt.expmax

    @property
    def expmin(self):
        """Minimum unnormalized exponent."""
        return self._fmt.expmin

    @property
    def nmin(self):
        """First unrepresentable digit for every value in the representation."""
        return self._fmt.nmin

    def with_params(
        self, *,
        pmax: DefaultOr[int] = DEFAULT,
        emin: DefaultOr[int] = DEFAULT,
        maxval: DefaultOr[RealFloat] = DEFAULT,
        rm: DefaultOr[RoundingMode] = DEFAULT,
        overflow: DefaultOr[OverflowMode] = DEFAULT,
        neg_maxval: DefaultOr[RealFloat] = DEFAULT,
        num_randbits: DefaultOr[int | None] = DEFAULT,
        rng: DefaultOr[RNG | None] = DEFAULT,
        enable_nan: DefaultOr[bool] = DEFAULT,
        enable_inf: DefaultOr[bool] = DEFAULT,
        enable_neg_zero: DefaultOr[bool] = DEFAULT,
        nan_value: DefaultOr[Float | None] = DEFAULT,
        inf_value: DefaultOr[Float | None] = DEFAULT,
        **kwargs
    ) -> 'MPBFloatContext':
        if pmax is DEFAULT:
            pmax = self.pmax
        if emin is DEFAULT:
            emin = self.emin
        if maxval is DEFAULT:
            maxval = self.pos_maxval
        if rm is DEFAULT:
            rm = self.rm
        if overflow is DEFAULT:
            overflow = self.overflow
        if neg_maxval is DEFAULT:
            neg_maxval = self.neg_maxval
        if num_randbits is DEFAULT:
            num_randbits = self.num_randbits
        if rng is DEFAULT:
            rng = self.rng
        if enable_nan is DEFAULT:
            enable_nan = self.enable_nan
        if enable_inf is DEFAULT:
            enable_inf = self.enable_inf
        if enable_neg_zero is DEFAULT:
            enable_neg_zero = self.enable_neg_zero
        if nan_value is DEFAULT:
            nan_value = self.nan_value
        if inf_value is DEFAULT:
            inf_value = self.inf_value
        if kwargs:
            raise TypeError(f'Unexpected keyword arguments: {kwargs}')
        return MPBFloatContext(
            pmax, emin, maxval, rm, overflow, num_randbits,
            neg_maxval=neg_maxval,
            rng=rng,
            enable_nan=enable_nan,
            enable_inf=enable_inf,
            enable_neg_zero=enable_neg_zero,
            nan_value=nan_value,
            inf_value=inf_value
        )

    def is_stochastic(self) -> bool:
        return self.num_randbits != 0

    def format(self) -> MPBFloatFormat:
        return self._fmt

    @classmethod
    def from_format(
        cls,
        fmt: MPBFloatFormat,
        *,
        rm: RoundingMode = RoundingMode.RNE,
        overflow: OverflowMode | None = None,
        num_randbits: int | None = 0,
        rng: 'RNG | None' = None,
        nan_value: Float | None = None,
        inf_value: Float | None = None
    ) -> 'MPBFloatContext':
        """Creates a context from a `MPBFloatFormat` and rounding parameters."""
        if not isinstance(fmt, MPBFloatFormat):
            raise TypeError(f'Expected \'MPBFloatFormat\', got {type(fmt)}')
        if overflow is None:
            overflow = OverflowMode.OVERFLOW
        return cls(
            fmt.pmax, fmt.emin, fmt.pos_maxval, rm, overflow, num_randbits,
            neg_maxval=fmt.neg_maxval,
            rng=rng,
            enable_nan=fmt.enable_nan,
            enable_inf=fmt.enable_inf,
            enable_neg_zero=fmt.enable_neg_zero,
            nan_value=nan_value,
            inf_value=inf_value
        )

    def rounding_mode(self):
        return self.rm

    def round_params(self):
        if self.num_randbits is None:
            return None, None
        return self.pmax + self.num_randbits, self.nmin - self.num_randbits

    def _is_overflowing(self, x: RealFloat) -> bool:
        """Checks if `x` is overflowing."""
        if x.s:
            return x < self.neg_maxval
        return x > self.pos_maxval

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
                return True
            case _:
                raise RuntimeError(f'unreachable {direction}')

    def _round_at(self, x: RealFloat | Float, n: int | None, exact: bool) -> Float:
        """
        Like `self.round()` but for only `RealFloat` and `Float` inputs.

        Optionally specify `n` as the least absolute digit position.
        Only overrides rounding behavior when `n > self.nmin`.
        """
        # step 1. handle special values
        if isinstance(x, Float):
            if x.isnan:
                if self.enable_nan:
                    return Float(isnan=True, ctx=self)
                elif self.nan_value is None:
                    raise ValueError('Cannot round NaN under this context')
                else:
                    return Float(x=self.nan_value, ctx=self)
            elif x.isinf:
                if self.enable_inf:
                    return Float(s=x.s, isinf=True, ctx=self)
                elif self.inf_value is None:
                    raise ValueError('Cannot round infinity under this context')
                else:
                    return Float(s=x.s, x=self.inf_value, ctx=self)
            else:
                x = x.as_real()

        # step 2. shortcut for exact zero values (preserve signed zero)
        if x.is_zero():
            return Float(s=zero_sign(x, self.enable_neg_zero), ctx=self)

        # step 3. select rounding parameter `n`
        if n is None or n < self.nmin:
            n = self.nmin

        # step 4. round value based on rounding parameters
        rounded = x.round(self.pmax, n, self.rm, self.num_randbits, rng=self.rng, exact=exact)

        # step 5. check for overflow
        if self._is_overflowing(rounded):
            if exact:
                raise ValueError(f'Rounding {x} under self={self} with n={n} would overflow')

            match self.overflow:
                case OverflowMode.OVERFLOW:
                    if self._overflow_to_infinity(rounded.s):
                        # the operand is a finite value whose magnitude ran out
                        # of format, so its sign is carried through -- as every
                        # other arm here does -- even where the infinity itself
                        # is substituted
                        if self.enable_inf:
                            result = Float(x=x, isinf=True, ctx=self)
                        elif self.inf_value is None:
                            raise ValueError('Cannot round to infinity under this context')
                        else:
                            result = Float(s=rounded.s, x=self.inf_value, ctx=self)
                    else:
                        result = self.maxval(rounded.s)
                case OverflowMode.SATURATE:
                    result = self.maxval(rounded.s)
                case OverflowMode.ASSERT:
                    raise OverflowError(f'Rounding {x} under self={self} with n={n} would overflow')
                case _:
                    raise RuntimeError(f'unreachable: {self.overflow}')

            result._real._flags._set_overflow(True)
            result._real._flags._set_inexact(True)
            return result

        # step 6. return rounded result
        return Float(x=rounded, s=zero_sign(rounded, self.enable_neg_zero), ctx=self)

    def round(self, x, *, exact: bool = False) -> Float:
        x = self._round_prepare(x)
        return self._round_at(x, None, exact)

    def round_at(self, x, n: int, *, exact: bool = False) -> Float:
        x = self._round_prepare(x)
        return self._round_at(x, n, exact)

    def zero(self, s: bool = False) -> Float:
        """Returns a signed 0 under this context."""
        return Float(x=self._fmt.zero(s), ctx=self)

    def min_subnormal(self, s: bool = False) -> Float:
        """Returns the smallest subnormal value with sign `s` under this context."""
        return Float(x=self._fmt.min_subnormal(s), ctx=self)

    def max_subnormal(self, s: bool = False) -> Float:
        """Returns the largest subnormal value with sign `s` under this context."""
        return Float(x=self._fmt.max_subnormal(s), ctx=self)

    def min_normal(self, s: bool = False) -> Float:
        """Returns the smallest normal value with sign `s` under this context."""
        return Float(x=self._fmt.min_normal(s), ctx=self)

    def max_normal(self, s: bool = False) -> Float:
        """Returns the largest normal value with sign `s` under this context."""
        return Float(x=self._fmt.max_normal(s), ctx=self)
