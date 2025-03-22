"""
This module defines floating-point numbers as implemented by MPFR
but with a subnormalization, that is multi-precision floating-point
numbers with subnormals. Hence, "MP-S."
"""

from fractions import Fraction

from ..utils import default_repr, bitmask

from .context import OrdinalContext
from .float import Float
from .mp import MPContext
from .real import RealFloat
from .round import RoundingMode
from .utils import from_mpfr

@default_repr
class MPSContext(OrdinalContext):
    """
    Rounding context for multi-precision floating-point numbers with
    a minimum exponent (and subnormalization).

    This context is parameterized by a fixed precision `pmax`,
    a minimum (normalized) exponent `emin`, and a rounding mode `rm`.
    It emulates floating-point numbers as implemented by MPFR
    with subnormalization.

    Unlike `MPContext`, the `MPSContext` is inherits from `OrdinalContext`
    since each representable value can be mapped to the ordinals.
    """

    pmax: int
    """maximum precision"""

    emin: int
    """minimum (normalized exponent)"""

    _mp_ctx: MPContext
    """this context without subnormalization"""

    rm: RoundingMode
    """rounding mode"""

    def __init__(self, pmax: int, emin: int, rm: RoundingMode):
        if not isinstance(pmax, int):
            raise TypeError(f'Expected \'int\' for pmax={pmax}, got {type(pmax)}')
        if pmax < 1:
            raise TypeError(f'Expected integer p < 1 for p={pmax}')
        if not isinstance(emin, int):
            raise TypeError(f'Expected \'int\' for emin={emin}, got {type(emin)}')
        if not isinstance(rm, RoundingMode):
            raise TypeError(f'Expected \'RoundingMode\' for rm={rm}, got {type(rm)}')

        self.pmax = pmax
        self.emin = emin
        self.rm = rm
        self._mp_ctx = MPContext(pmax, rm)

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

    def is_representable(self, x: RealFloat | Float) -> bool:
        if not isinstance(x, RealFloat | Float):
            raise TypeError(f'Expected \'RealFloat\' or \'Float\', got \'{type(x)}\' for x={x}')

        if not self._mp_ctx.is_representable(x):
            # not representable even without subnormalization
            return False
        elif not x.is_nonzero():
            # NaN, Inf, 0
            return True
        elif x.s:
            # tight check (negative values)
            return x <= self.minval(True)
        else:
            # tight check (non-negative values)
            return self.minval(False) <= x

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
            xr = x.real.normalize(self.pmax, self.nmin)
            return Float(x=x, exp=xr.exp, c=xr.c, ctx=self)

    def _round_float(self, x: RealFloat | Float):
        """Like `self.round()` but for only `RealFloat` and `Float` inputs"""
        # step 1. handle special values
        if isinstance(x, Float):
            if x.isnan:
                return Float(isnan=True, ctx=self)
            elif x.isinf:
                return Float(s=x.s, isinf=True, ctx=self)
            else:
                x = x.real

        # step 2. shortcut for exact zero values
        if x.is_zero():
            # exactly zero
            return Float(ctx=self)

        # step 3. round value based on rounding parameters
        return x.round(self.pmax, self.nmin, self.rm)

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

    def to_ordinal(self, x: Float, infval = False) -> int:
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
        elif x.e <= self.emin:
            # subnormal
            eord = 0
            mord = x.c << (x.exp - self.expmin)  # normalize so that exp=self.expmin
        else:
            # normal
            c = x.c << (self.pmax - x.p) # normalize so that p=self.pmax
            eord = x.e - self.emin + 1
            mord = c & bitmask(self.pmax - 1)

        uord = (eord << (self.pmax - 1)) + mord
        return (-1 if x.s else 1) * uord

    def from_ordinal(self, x: int, infval = False):
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

    def minval(self, s = False) -> Float:
        return Float(s=s, c=1, exp=self.expmin, ctx=self)


