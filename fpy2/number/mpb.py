"""
This module defines floating-point numbers as implemented by MPFR
but with subnormalization and a maximum value, that is multi-precision
and bounded. Hence, "MP-B."
"""

from fractions import Fraction
from typing import Optional

from ..utils import default_repr, bitmask

from .context import SizedContext
from .float import Float
from .real import RealFloat
from .round import RoundingMode, RoundingDirection
from .utils import from_mpfr


@default_repr
class MPBContext(SizedContext):
    """
    Rounding context for multi-precision floating-point numbers with
    a minimum exponent (and subnormalization) and a maximum value.

    This context is parameterized by a fixed precision `pmax`,
    a minimum (normalized) exponent `emin`, a (positive) maximum value
    `maxval`, and a rounding mode `rm`. A separate negative maximum value
    may be specified as well, but by default it is set to the negative
    of `maxval`.

    Unlike `MPContext`, the `MPBContext` is inherits from `SizedContext`
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
            raise ValueError(f'Expected maxval={maxval} to be representable in pmax={pmax}')

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
        return self.emin - self.pmax + 1

    @property
    def nmin(self):
        """
        First unrepresentable digit for every value in the representation.
        """
        return self.expmin - 1

    def is_representable(self, x: Float) -> bool:
        if not isinstance(x, Float):
            raise TypeError(f'Expected \'RealFloat\', got \'{type(x)}\' for x={x}')

        if x.is_nar():
            # special values are valid
            return True
        elif x.exp < self.expmin or x.e > self.emax:
            # rough check on out of range values (even for zero)
            return False
        elif x.is_zero():
            # shortcut for exact zero
            return True
        elif x.p > self.pmax:
            # check on precision
            return False
        elif x.s:
            # tight check (negative values)
            return self.maxval(True) <= x <= self.minval(True)
        else:
            # tight check (non-negative values)
            return self.minval(False) <= x <= self.maxval(False)

    def is_canonical(self, x):
        if not isinstance(x, Float) or not self.is_representable(x):
            raise TypeError(f'Expected a representable \'Float\', got \'{type(x)}\' for x={x}')

        # case split by class
        if x.is_nar():
            # NaN or Inf
            return True
        elif x.c == 0:
            # zero
            return x.exp == self.expmin
        elif x.e < self.emin:
            # subnormal
            return x.exp == self.expmin
        else:
            # normal
            return x.p == self.pmax

    def normalize(self, x):
        if not isinstance(x, Float) or not self.is_representable(x):
            raise TypeError(f'Expected a representable \'Float\', got \'{type(x)}\' for x={x}')

        # case split by class
        if x.isnan:
            # NaN
            return Float(isnan=True, s=x.s, ctx=self)
        elif x.isinf:
            # Inf
            return Float(isinf=True, s=x.s, ctx=self)
        elif x.c == 0:
            # zero
            return Float(c=0, exp=self.expmin, s=x.s, ctx=self)
        else:
            # non-zero
            xr = x.as_real().normalize(self.pmax, self.nmin)
            return Float(x=x, exp=xr.exp, c=xr.c, ctx=self)

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

    def _round_float(self, x: RealFloat | Float):
        """Like `self.round()` but for only `RealFloat` and `Float` inputs"""
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

        # step 3. round value based on rounding parameters
        rounded = x.round(self.pmax, self.nmin, self.rm)

        # step 4. check for overflow
        if self._is_overflowing(rounded):
            # overflowing => check which way to round
            if self._overflow_to_infinity(rounded):
                # overflow to infinity
                return Float(x=x, isinf=True, ctx=self)
            else:
                # overflow to MAX_VAL
                max_val = self.maxval(rounded.s)
                return Float(x=max_val, ctx=self)

        # step 5. return rounded result
        return Float(x=rounded, ctx=self)


    def round(self, x):
        match x:
            case Float() | RealFloat():
                xr = x
            case int():
                xr = RealFloat(c=x)
            case float() | str():
                xr = from_mpfr(x, self.pmax)
            case Fraction():
                if x.is_integer():
                    xr = RealFloat(c=int(x))
                else:
                    xr = from_mpfr(x, self.pmax)
            case _:
                raise TypeError(f'not valid argument x={x}')

        return self._round_float(xr)

    def to_ordinal(self, x, infval = False):
        if not isinstance(x, Float) or not self.is_representable(x):
            raise TypeError(f'Expected a representable \'Float\', got \'{type(x)}\' for x={x}')
        if infval:
            raise ValueError('infval=True is invalid for contexts without a maximum value')

        # case split by class
        if x.is_nar():
            # NaN or Inf
            raise TypeError(f'Expected a finite value for x={x}')
        elif x.is_zero():
            # zero
            return 0
        else:
            # finite

            # canonicalize number if necessary
            if not x.is_canonical():
                x = x.normalize()

            # case split by class
            if x.e <= self.emin:
                # subnormal number
                eord = 0
                mord = x.c
            else:
                # normal number
                eord = x.e - self.emin + 1
                mord = x.c & bitmask(self.pmax - 1)

        uord = eord * self.pmax + mord
        return (-1 if x.s else 1) * uord

    def from_ordinal(self, x, infval = False):
        if not isinstance(x, int):
            raise TypeError(f'Expected an \'int\', got \'{type(x)}\' for x={x}')
        if infval:
            raise ValueError('infval=True is invalid for contexts without a maximum value')

        s = x < 0
        uord = abs(x)

        if x == 0:
            # zero
            return Float(ctx=self)
        else:
            # finite values
            eord, mord = divmod(uord, self.pmax)
            if eord == 0:
                # subnormal
                return Float(s=s, c=mord, exp=self.expmin, ctx=self)
            else:
                # normal
                c = 1 << self.pmax | mord
                exp = self.expmin + (eord - 1)
                return Float(s=s, c=c, exp=exp, ctx=self)

    def minval(self, s = False):
        raise NotImplementedError

    def maxval(self, s = False):
        if s:
            return Float(x=self.neg_maxval, ctx=self)
        else:
            return Float(x=self.pos_maxval, ctx=self)

