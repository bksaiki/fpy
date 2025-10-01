"""
This module defines floating-point numbers as implemented by MPFR
but with subnormalization and a maximum value, that is multi-precision
and bounded. Hence, "MP-B."
"""

from fractions import Fraction

from ..utils import default_repr, DefaultOr, DEFAULT

from .context import Context, SizedContext
from .number import RealFloat, Float
from .mps_float import MPSFloatContext
from .round import RoundingMode, RoundingDirection, OverflowMode
from .gmp import mpfr_value


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

    _mps_ctx: MPSFloatContext
    """this context without maximum values"""

    _pos_maxval_ord: int
    """precomputed ordinal of `self.pos_maxval`"""

    _neg_maxval_ord: int
    """precomputed ordinal of `self.neg_maxval`"""

    def __init__(
        self,
        pmax: int,
        emin: int,
        maxval: RealFloat, 
        rm: RoundingMode = RoundingMode.RNE,
        overflow: OverflowMode = OverflowMode.OVERFLOW,
        num_randbits: int | None = 0,
        *,
        neg_maxval: RealFloat | None = None
    ):
        if not isinstance(pmax, int):
            raise TypeError(f'Expected \'int\' for pmax={pmax}, got {type(pmax)}')
        if pmax < 1:
            raise TypeError(f'Expected integer p < 1 for p={pmax}')
        if not isinstance(emin, int):
            raise TypeError(f'Expected \'int\' for emin={emin}, got {type(emin)}')
        if not isinstance(maxval, RealFloat):
            raise TypeError(f'Expected \'RealFloat\' for maxval={maxval}, got {type(maxval)}')
        if not isinstance(rm, RoundingMode):
            raise TypeError(f'Expected \'RoundingMode\' for rm={rm}, got {type(rm)}')
        if not isinstance(overflow, OverflowMode):
            raise TypeError(f'Expected \'OverflowMode\' for overflow={overflow}, got {type(overflow)}')
        if num_randbits is not None and not isinstance(num_randbits, int):
            raise TypeError(f'Expected \'int\' for num_randbits={num_randbits}, got {type(num_randbits)}')

        if overflow == OverflowMode.WRAP:
            raise ValueError('OverflowMode.WRAP is not supported for MPBFloatContext')

        if maxval.s:
            raise ValueError(f'Expected positive maxval={maxval}, got {maxval}')
        elif maxval.p > pmax:
            raise ValueError(f'Expected maxval={maxval} to be representable in pmax={pmax} (p={maxval.p})')

        if neg_maxval is None:
            neg_maxval = RealFloat(s=True, x=maxval)
        elif not isinstance(neg_maxval, RealFloat):
            raise TypeError(f'Expected \'RealFloat\' for neg_maxval={neg_maxval}, got {type(neg_maxval)}')
        elif not neg_maxval.s:
            raise ValueError(f'Expected negative neg_maxval={neg_maxval}, got {neg_maxval}')
        elif neg_maxval.p > pmax:
            raise ValueError(f'Expected neg_maxval={neg_maxval} to be representable in pmax={pmax}')

        self.pmax = pmax
        self.emin = emin
        self.pos_maxval = maxval
        self.neg_maxval = neg_maxval
        self.rm = rm
        self.overflow = overflow
        self.num_randbits = num_randbits

        self._mps_ctx = MPSFloatContext(pmax, emin, rm, num_randbits)
        pos_maxval_mps = Float(x=self.pos_maxval, ctx=self._mps_ctx)
        neg_maxval_mps = Float(x=self.neg_maxval, ctx=self._mps_ctx)
        self._pos_maxval_ord = self._mps_ctx.to_ordinal(pos_maxval_mps)
        self._neg_maxval_ord = self._mps_ctx.to_ordinal(neg_maxval_mps)

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
        )

    def __hash__(self):
        return hash((self.pmax, self.emin, self.pos_maxval, self.neg_maxval, self.rm, self.overflow, self.num_randbits))

    @property
    def emax(self):
        """Maximum normalized exponent."""
        pos_e = self.pos_maxval.e
        neg_e = self.neg_maxval.e
        return max(pos_e, neg_e)

    @property
    def expmax(self):
        """Maximum unnormalized exponent."""
        return self.emax - self.pmax + 1

    @property
    def expmin(self):
        """Minimum unnormalized exponent."""
        return self._mps_ctx.expmin

    @property
    def nmin(self):
        """
        First unrepresentable digit for every value in the representation.
        """
        return self._mps_ctx.nmin

    def with_params(
        self, *,
        pmax: DefaultOr[int] = DEFAULT,
        emin: DefaultOr[int] = DEFAULT,
        maxval: DefaultOr[RealFloat] = DEFAULT,
        rm: DefaultOr[RoundingMode] = DEFAULT,
        overflow: DefaultOr[OverflowMode] = DEFAULT,
        neg_maxval: DefaultOr[RealFloat] = DEFAULT,
        num_randbits: DefaultOr[int | None] = DEFAULT,
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
        if kwargs:
            raise TypeError(f'Unexpected keyword arguments: {kwargs}')
        return MPBFloatContext(pmax, emin, maxval, rm, overflow, neg_maxval=neg_maxval)

    def is_stochastic(self) -> bool:
        return self.num_randbits != 0

    def is_equiv(self, other):
        if not isinstance(other, Context):
            raise TypeError(f'Expected \'Context\', got \'{type(other)}\' for other={other}')
        return (
            isinstance(other, MPBFloatContext)
            and self.nmin == other.nmin
            and self.pos_maxval == other.pos_maxval
            and self.neg_maxval == other.neg_maxval
        )

    def representable_under(self, x: RealFloat | Float) -> bool:
        match x:
            case Float():
                if x.ctx is not None and self.is_equiv(x.ctx):
                    # same context, so representable
                    return True
            case RealFloat():
                pass
            case _:
                raise TypeError(f'Expected \'RealFloat\' or \'Float\', got \'{type(x)}\' for x={x}')

        if not self._mps_ctx.representable_under(x):
            # not representable even without a maximum value
            return False
        elif not x.is_nonzero():
            # NaN, Inf, 0
            return True
        elif x.s:
            # check bounded (negative values)
            return self.neg_maxval <= x
        else:
            # check bounded (non-negative values)
            return x <= self.pos_maxval

    def canonical_under(self, x: Float):
        if not isinstance(x, Float) or not self.representable_under(x):
            raise TypeError(f'Expected a representable \'Float\', got \'{type(x)}\' for x={x}')
        return self._mps_ctx.canonical_under(x)

    def normal_under(self, x: Float):
        if not isinstance(x, Float) or not self.representable_under(x):
            raise TypeError(f'Expected a representable \'Float\', got \'{type(x)}\' for x={x}')
        return self._mps_ctx.normal_under(x)

    def _normalize(self, x: Float) -> Float:
        return self._mps_ctx._normalize(x)

    def normalize(self, x: Float):
        if not isinstance(x, Float) or not self.representable_under(x):
            raise TypeError(f'Expected a representable \'Float\', got \'{type(x)}\' for x={x}')
        return Float(x=self._normalize(x), ctx=self)

    def round_params(self):
        return self._mps_ctx.round_params()

    def _is_overflowing(self, x: RealFloat) -> bool:
        """Checks if `x` is overflowing."""
        if x.s:
            return x < self.neg_maxval
        else:
            return x > self.pos_maxval

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

    def _round_at(self, x: RealFloat | Float, n: int | None, exact: bool) -> Float:
        """
        Like `self.round()` but for only `RealFloat` and `Float` inputs.

        Optionally specify `n` as the least absolute digit position.
        Only overrides rounding behavior when `n > self.nmin`.
        """
        # step 1. handle special values
        if isinstance(x, Float):
            if x.isnan:
                return Float(isnan=True, ctx=self)
            elif x.isinf:
                return Float(s=x.s, isinf=True, ctx=self)
            else:
                x = x.as_real()

        # step 2. shortcut for exact zero values
        if x.is_zero():
            # exactly zero
            return Float(ctx=self)

        # step 3. select rounding parameter `n`
        if n is None or n < self.nmin:
            # no rounding parameter
            n = self.nmin

        # step 4. round value based on rounding parameters
        rounded = x.round(self.pmax, n, self.rm, self.num_randbits, exact=exact)

        # step 5. check for overflow
        if self._is_overflowing(rounded):
            # check if overflow is allowed
            if exact:
                raise ValueError(f'Rounding {x} under self={self} with n={n} would overflow')

            # case split on overflow behavior
            match self.overflow:
                case OverflowMode.OVERFLOW:
                    # overflowing => check which way to round
                    if self._overflow_to_infinity(rounded.s):
                        # overflow to infinity
                        return Float(x=x, isinf=True, ctx=self)
                    else:
                        # overflow to MAX_VAL
                        max_val = self.maxval(rounded.s)
                        return Float(x=max_val, ctx=self)
                case OverflowMode.SATURATE:
                    # always produce the maximum value
                    max_val = self.maxval(rounded.s)
                    return Float(x=max_val, ctx=self)
                case _:
                    raise RuntimeError(f'unreachable: {self.overflow}')

        # step 6. return rounded result
        return Float(x=rounded, ctx=self)

    def round(self, x, *, exact: bool = False) -> Float:
        x = self._round_prepare(x)
        return self._round_at(x, None, exact)

    def round_at(self, x, n: int, *, exact: bool = False) -> Float:
        x = self._round_prepare(x)
        return self._round_at(x, n, exact)

    def to_ordinal(self, x: Float, infval = False):
        if not isinstance(x, Float):
            raise TypeError(f'Expected a \'Float\', got \'{type(x)}\' for x={x}')
        if not self.representable_under(x):
            raise ValueError(f'Expected a representable \'Float\', got x={x} (x.ctx={x.ctx})')

        # case split by class
        if x.isnan:
            # NaN
            raise TypeError(f'Expected a finite value for x={x}')
        elif x.isinf:
            # Inf
            if not infval:
                raise TypeError(f'Expected a finite value for x={x}')
            elif x.s:
                # -Inf is mapped to 1 less than -MAX
                return self._neg_maxval_ord - 1
            else:
                # +Inf is mapped to 1 greater than +MAX
                return self._pos_maxval_ord + 1
        else:
            # finite, real
            return self._mps_ctx.to_ordinal(x)


    def from_ordinal(self, x, infval = False):
        if not isinstance(x, int):
            raise TypeError(f'Expected an \'int\', got \'{type(x)}\' for x={x}')
 
        if x > self._pos_maxval_ord:
            # ordinal too large to be a finite number
            if not infval or x > self._pos_maxval_ord + 1:
                # infinity ordinal is disabled or ordinal is too large to even be infinity
                raise ValueError(f'Expected an \'int\' between {self._neg_maxval_ord} and {self._pos_maxval_ord}, got x={x}')
            else:
                # +Inf
                return Float(isinf=True, ctx=self)
        elif x < self._neg_maxval_ord:
            # ordinal is too large to be a finite number
            if not infval or x < self._neg_maxval_ord - 1:
                # infinity ordinal is disabled or ordinal is too large to even be infinity
                raise ValueError(f'Expected an \'int\' between {self._neg_maxval_ord} and {self._pos_maxval_ord}, got x={x}')
            else:
                # -Inf
                return Float(s=True, isinf=True, ctx=self)
        else:
            # must be a finite number
            return Float(x=self._mps_ctx.from_ordinal(x), ctx=self)

    def zero(self, s: bool = False) -> Float:
        """Returns a signed 0 under this context."""
        if not isinstance(s, bool):
            raise TypeError(f'Expected \'bool\' for s={s}, got {type(s)}')
        return Float(x=self._mps_ctx.zero(s=s), ctx=self)

    def minval(self, s = False) -> Float:
        """Returns the smallest non-zero value with sign `s` under this context."""
        if not isinstance(s, bool):
            raise TypeError(f'Expected \'bool\' for s={s}, got {type(s)}')
        return Float(x=self._mps_ctx.minval(s=s), ctx=self)

    def min_subnormal(self, s = False) -> Float:
        """Returns the smallest subnormal value with sign `s` under this context."""
        if not isinstance(s, bool):
            raise TypeError(f'Expected \'bool\' for s={s}, got {type(s)}')
        return Float(x=self._mps_ctx.min_subnormal(s=s), ctx=self)

    def max_subnormal(self, s = False) -> Float:
        """Returns the largest subnormal value with sign `s` under this context."""
        if not isinstance(s, bool):
            raise TypeError(f'Expected \'bool\' for s={s}, got {type(s)}')
        return Float(x=self._mps_ctx.max_subnormal(s=s), ctx=self)

    def min_normal(self, s = False) -> Float:
        """Returns the smallest normal value with sign `s` under this context."""
        if not isinstance(s, bool):
            raise TypeError(f'Expected \'bool\' for s={s}, got {type(s)}')
        return Float(x=self._mps_ctx.min_normal(s=s), ctx=self)

    def max_normal(self, s: bool = False) -> Float:
        """Returns the largest normal value with sign `s` under this context."""
        if not isinstance(s, bool):
            raise TypeError(f'Expected \'bool\' for s={s}, got {type(s)}')
        if s:
            return Float(x=self.neg_maxval, ctx=self)
        else:
            return Float(x=self.pos_maxval, ctx=self)

    def maxval(self, s: bool = False) -> Float:
        """Returns the largest value with sign `s` under this context."""
        if not isinstance(s, bool):
            raise TypeError(f'Expected \'bool\' for s={s}, got {type(s)}')
        return self.max_normal(s=s)

    def infval(self, s: bool = False) -> RealFloat:
        """
        Returns the first non-representable value larger
        than `maxval` with sign `s`.
        """
        if not isinstance(s, bool):
            raise TypeError(f'Expected \'bool\' for s={s}, got {type(s)}')
        if s:
            return self.neg_maxval.next_away()
        else:
            return self.pos_maxval.next_away()
