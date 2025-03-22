"""
This module defines floating-point numbers as implemented by MPFR,
that is, multi-precision floating-point numbers. Hence, "MP."
"""

from fractions import Fraction

from ..utils import default_repr

from .context import Context
from .float import Float
from .real import RealFloat
from .round import RoundingMode
from .utils import from_mpfr

@default_repr
class MPContext(Context):
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

    def is_representable(self, x: Float) -> bool:
        if not isinstance(x, Float):
            raise TypeError(f'Expected a representable \'Float\', got \'{type(x)}\' for x={x}')

        # case split on class
        if x.is_nar() or x.is_zero():
            # special values and zeros are valid
            return True
        else:
            # non-zero value
            return x.p <= self.pmax

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
            xr = x.as_real().normalize(self.pmax, None)
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
                x = x.as_real()

        # step 2. shortcut for exact zero values
        if x.is_zero():
            # exactly zero
            return Float(ctx=self)

        # step 3. round value based on rounding parameters
        return x.round(max_p=self.pmax, rm=self.rm)

    def round(self, x) -> Float:
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

