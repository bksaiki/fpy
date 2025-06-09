"""
This module defines floating-point numbers as implemented by MPFR
but with subnormalization and a maximum value, that is multi-precision
and bounded. Hence, "MP-B."
"""

from fractions import Fraction
from typing import Optional

from ..utils import default_repr

from .context import SizedContext
from .number import RealFloat, Float
from .mps_float import MPSFloatContext
from .round import RoundingMode, RoundingDirection
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
        rm: RoundingMode, *,
        neg_maxval: Optional[RealFloat] = None
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

        self._mps_ctx = MPSFloatContext(pmax, emin, rm)
        pos_maxval_mps = Float(x=self.pos_maxval, ctx=self._mps_ctx)
        neg_maxval_mps = Float(x=self.neg_maxval, ctx=self._mps_ctx)
        self._pos_maxval_ord = self._mps_ctx.to_ordinal(pos_maxval_mps)
        self._neg_maxval_ord = self._mps_ctx.to_ordinal(neg_maxval_mps)


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

    def with_rm(self, rm: RoundingMode):
        return MPBFloatContext(self.pmax, self.emin, self.pos_maxval, rm, neg_maxval=self.neg_maxval)

    def is_representable(self, x: RealFloat | Float) -> bool:
        if not isinstance(x, RealFloat | Float):
            raise TypeError(f'Expected \'RealFloat\' or \'Float\', got \'{type(x)}\' for x={x}')

        if not self._mps_ctx.is_representable(x):
            # not representable even without a maximum value
            return False
        elif not x.is_nonzero():
            # NaN, Inf, 0
            return True
        elif x.s:
            # check bounded (negative values)
            # TODO: don't call maxval just use self.XXX_maxval directly
            return self.maxval(True) <= x
        else:
            # check bounded (non-negative values)
            return x <= self.maxval(False)

    def is_canonical(self, x: Float):
        if not isinstance(x, Float) or not self.is_representable(x):
            raise TypeError(f'Expected a representable \'Float\', got \'{type(x)}\' for x={x}')
        return self._mps_ctx.is_canonical(x)

    def is_normal(self, x: Float):
        if not isinstance(x, Float) or not self.is_representable(x):
            raise TypeError(f'Expected a representable \'Float\', got \'{type(x)}\' for x={x}')
        return self._mps_ctx.is_normal(x)

    def normalize(self, x: Float):
        if not isinstance(x, Float) or not self.is_representable(x):
            raise TypeError(f'Expected a representable \'Float\', got \'{type(x)}\' for x={x}')
        x = self._mps_ctx.normalize(x)
        x.ctx = self
        return x

    def round_params(self):
        return self._mps_ctx.round_params()

    def _is_overflowing(self, x: RealFloat) -> bool:
        """Checks if `x` is overflowing."""
        if x.s:
            return x < self.neg_maxval
        else:
            return x > self.pos_maxval

    def _overflow_to_infinity(self, x: RealFloat):
        """Should overflows round to infinity (rather than MAX_VAL)?"""
        _, direction = self.rm.to_direction(x.s)
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

    def _round_float_at(self, x: RealFloat | Float, n: Optional[int]) -> Float:
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
        rounded = x.round(self.pmax, n, self.rm)

        # step 5. check for overflow
        if self._is_overflowing(rounded):
            # overflowing => check which way to round
            if self._overflow_to_infinity(rounded):
                # overflow to infinity
                return Float(x=x, isinf=True, ctx=self)
            else:
                # overflow to MAX_VAL
                max_val = self.maxval(rounded.s)
                return Float(x=max_val, ctx=self)

        # step 6. return rounded result
        return Float(x=rounded, ctx=self)

    def _round_at(self, x, n: Optional[int]) -> Float:
        match x:
            case Float() | RealFloat():
                xr = x
            case int():
                xr = RealFloat.from_int(x)
            case float() | str():
                xr = mpfr_value(x, prec=self.pmax)
            case Fraction():
                if x.denominator == 1:
                    xr = RealFloat.from_int(int(x))
                else:
                    xr = mpfr_value(x, prec=self.pmax)
            case _:
                raise TypeError(f'not valid argument x={x}')

        return self._round_float_at(xr, n)

    def round(self, x) -> Float:
        return self._round_at(x, None)

    def round_at(self, x, n: int) -> Float:
        return self._round_at(x, n)

    def to_ordinal(self, x: Float, infval = False):
        if not isinstance(x, Float) or not self.is_representable(x):
            raise TypeError(f'Expected a representable \'Float\', got \'{type(x)}\' for x={x}')

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
            v = self._mps_ctx.from_ordinal(x)
            v.ctx = self
            return v

    def zero(self, s: bool = False) -> Float:
        """Returns a signed 0 under this context."""
        if not isinstance(s, bool):
            raise TypeError(f'Expected \'bool\' for s={s}, got {type(s)}')
        x = self._mps_ctx.zero(s=s)
        x.ctx = self
        return x

    def minval(self, s = False) -> Float:
        """Returns the smallest non-zero value with sign `s` under this context."""
        if not isinstance(s, bool):
            raise TypeError(f'Expected \'bool\' for s={s}, got {type(s)}')
        x = self._mps_ctx.minval(s=s)
        x.ctx = self
        return x

    def min_subnormal(self, s = False) -> Float:
        """Returns the smallest subnormal value with sign `s` under this context."""
        if not isinstance(s, bool):
            raise TypeError(f'Expected \'bool\' for s={s}, got {type(s)}')
        x = self._mps_ctx.min_subnormal(s=s)
        x.ctx = self
        return x

    def max_subnormal(self, s = False) -> Float:
        """Returns the largest subnormal value with sign `s` under this context."""
        if not isinstance(s, bool):
            raise TypeError(f'Expected \'bool\' for s={s}, got {type(s)}')
        x = self._mps_ctx.max_subnormal(s=s)
        x.ctx = self
        return x

    def min_normal(self, s = False) -> Float:
        """Returns the smallest normal value with sign `s` under this context."""
        if not isinstance(s, bool):
            raise TypeError(f'Expected \'bool\' for s={s}, got {type(s)}')
        x = self._mps_ctx.min_normal(s=s)
        x.ctx = self
        return x

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
