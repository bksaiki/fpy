"""
This module defines floating-point numbers as implemented by MPFR
but with a subnormalization, that is multi-precision floating-point
numbers with subnormals. Hence, "MP-S."
"""

from fractions import Fraction
from typing import Optional

from ..utils import default_repr, bitmask

from .context import OrdinalContext
from .number import RealFloat, Float
from .mp_float import MPFloatContext
from .round import RoundingMode
from .gmp import mpfr_value

@default_repr
class MPSFloatContext(OrdinalContext):
    """
    Rounding context for multi-precision floating-point numbers with
    a minimum exponent (and subnormalization).

    This context is parameterized by a fixed precision `pmax`,
    a minimum (normalized) exponent `emin`, and a rounding mode `rm`.
    It emulates floating-point numbers as implemented by MPFR
    with subnormalization.

    Unlike `MPFloatContext`, the `MPSFloatContext` inherits from `OrdinalContext`
    since each representable value can be mapped to the ordinals.
    """

    pmax: int
    """maximum precision"""

    emin: int
    """minimum (normalized exponent)"""

    _mp_ctx: MPFloatContext
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
        self._mp_ctx = MPFloatContext(pmax, rm)

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

    def with_rm(self, rm: RoundingMode):
        return MPSFloatContext(self.pmax, self.emin, rm)

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
            xr = x._real.normalize(self.pmax, self.nmin)
            return Float(x=x, exp=xr.exp, c=xr.c, ctx=self)

    def is_normal(self, x: Float):
        if not isinstance(x, Float) or not self.is_representable(x):
            raise TypeError(f'Expected a representable \'Float\', got \'{type(x)}\' for x={x}')
        return x.is_nonzero() and x.e >= self.emin

    def round_params(self):
        return (self.pmax, self.nmin)

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
                x = x._real

        # step 2. shortcut for exact zero values
        if x.is_zero():
            # exactly zero
            return Float(ctx=self)

        # step 3. select rounding parameter `n`
        if n is None or n < self.nmin:
            # no rounding parameter
            n = self.nmin

        # step 3. round value based on rounding parameters
        return x.round(self.pmax, n, self.rm)

    def _round_at(self, x, n: Optional[int]) -> Float:
        match x:
            case Float() | RealFloat():
                xr = x
            case int():
                xr = RealFloat(c=x)
            case float() | str():
                xr = mpfr_value(x, prec=self.pmax)
            case Fraction():
                if x.denominator == 1:
                    xr = RealFloat(c=int(x))
                else:
                    xr = mpfr_value(x, prec=self.pmax)
            case _:
                raise TypeError(f'not valid argument x={x}')

        return self._round_float_at(xr, n)

    def round(self, x) -> Float:
        return self._round_at(x, None)

    def round_at(self, x, n: int) -> Float:
        return self._round_at(x, n)

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

            # ordinal components
            eord = 0
            mord = c
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

            # ordinal components
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
            eord, mord = divmod(uord, 1 << (self.pmax - 1))
            if eord == 0:
                # subnormal
                return Float(s=s, c=mord, exp=self.expmin, ctx=self)
            else:
                # normal
                c = (1 << (self.pmax - 1)) | mord
                exp = self.expmin + (eord - 1)
                return Float(s=s, c=c, exp=exp, ctx=self)


    def zero(self, s: bool = False) -> Float:
        """Returns a signed 0 under this context."""
        if not isinstance(s, bool):
            raise TypeError(f'Expected \'bool\' for s={s}, got {type(s)}')
        return Float(s=s, c=0, exp=self.expmin, ctx=self)

    def minval(self, s: bool = False) -> Float:
        """Returns the smallest non-zero value with sign `s` under this context."""
        if not isinstance(s, bool):
            raise TypeError(f'Expected \'bool\' for s={s}, got {type(s)}')
        return Float(s=s, c=1, exp=self.expmin, ctx=self)

    def min_subnormal(self, s: bool = False) -> Float:
        """Returns the smallest subnormal value with sign `s` under this context."""
        if not isinstance(s, bool):
            raise TypeError(f'Expected \'bool\' for s={s}, got {type(s)}')
        return self.minval(s)

    def max_subnormal(self, s: bool = False) -> Float:
        """Returns the largest subnormal value with sign `s` under this context."""
        if not isinstance(s, bool):
            raise TypeError(f'Expected \'bool\' for s={s}, got {type(s)}')
        c = bitmask(self.pmax - 1)
        exp = self.expmin
        return Float(s=s, c=c, exp=exp, ctx=self)

    def min_normal(self, s: bool = False) -> Float:
        """Returns the smallest normal value with sign `s` under this context."""
        if not isinstance(s, bool):
            raise TypeError(f'Expected \'bool\' for s={s}, got {type(s)}')
        c = 1 << (self.pmax - 1)
        exp = self.emin
        return Float(s=s, c=c, exp=exp, ctx=self)

