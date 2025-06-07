"""
This module defines floating-point numbers as implemented by MPFR,
that is, multi-precision floating-point numbers. Hence, "MP."
"""

from fractions import Fraction
from typing import Optional

from ..utils import default_repr, bitmask

from .context import Context
from .number import RealFloat, Float
from .round import RoundingMode
from .gmp import mpfr_value

@default_repr
class MPFloatContext(Context):
    """
    Rounding context for multi-precision floating-point numbers.

    This context is parameterized by a fixed precision `pmax`
    and a rounding mode `rm`. It emulates floating-point numbers
    as implemented by MPFR.
    """

    pmax: int
    """maximum precision"""

    rm: RoundingMode
    """rounding mode"""

    def __init__(self, pmax: int, rm: RoundingMode):
        if not isinstance(pmax, int):
            raise TypeError(f'Expected \'int\' for pmax={pmax}, got {type(pmax)}')
        if pmax < 1:
            raise TypeError(f'Expected integer p < 1 for p={pmax}')
        if not isinstance(rm, RoundingMode):
            raise TypeError(f'Expected \'RoundingMode\' for rm={rm}, got {type(rm)}')

        self.pmax = pmax
        self.rm = rm

    def with_rm(self, rm: RoundingMode):
        return MPFloatContext(self.pmax, rm)

    def is_representable(self, x: RealFloat | Float) -> bool:
        if not isinstance(x, RealFloat | Float):
            raise TypeError(f'Expected \'RealFloat\' or \'Float\', got \'{type(x)}\' for x={x}')

        # case split on class
        if isinstance(x, Float) and x.is_nar():
            # special value
            return True
        elif x.is_zero():
            # special values and zeros are valid
            return True
        elif x.p <= self.pmax:
            # precision is within bounds:
            return True
        else:
            # precision is possibly out of bounds
            # check if the value can be normalized with fewer digits
            p_over = x.p - self.pmax
            c_lost = x.c & bitmask(p_over) # bits that would be lost via normalization
            return c_lost == 0

    def is_canonical(self, x: Float) -> bool:
        if not isinstance(x, Float) or not self.is_representable(x):
            raise TypeError(f'Expected a representable \'Float\', got \'{type(x)}\' for x={x}')

        # case split on class
        if x.is_nar():
            # NaN or Inf
            return True
        elif x.is_zero():
            # zero
            return x.exp == 0
        else:
            # non-zero value
            return x.p == self.pmax

    def normalize(self, x: Float) -> Float:
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
            return Float(c=0, exp=0, s=x.s, ctx=self)
        else:
            # non-zero
            xr = x._real.normalize(self.pmax, None)
            return Float(x=x, exp=xr.exp, c=xr.c, ctx=self)

    def is_normal(self, x: Float) -> bool:
        if not isinstance(x, Float):
            raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
        return x.is_nonzero()

    def _round_float_at(self, x: RealFloat | Float, n: Optional[int]) -> Float:
        """
        Like `self.round()` but for only `RealFloat` and `Float` inputs.

        Optionally specify `n` as the least absolute digit position.
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

        # step 3. round value based on rounding parameters
        return x.round(max_p=self.pmax, min_n=n, rm=self.rm)

    def round_params(self):
        return (self.pmax, None)

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
