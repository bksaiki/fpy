"""
This module defines fixed-point numbers with a fixed least-significant digit
but no most-significand digit, that is, a fixed-point number with arbitrary precision.
Hence, "MP-F".
"""

from fractions import Fraction
from typing import Optional

from ..utils import default_repr

from .context import Context
from .number import Float
from .real import RealFloat
from .round import RoundingMode
from .gmp import mpfr_value

@default_repr
class MPFContext(Context):
    """
    Rounding context for mulit-precision fixed-point numbers.

    This context is parameterized by the most significant digit
    that is not representable `nmin` and a rounding mode `rm`.
    It emulates fixed-point numbers with arbitrary precision.
    """

    nmin: int
    """the first unrepresentable digit"""

    rm: RoundingMode
    """rounding mode"""

    def __init__(self, nmin: int, rm: RoundingMode):
        if not isinstance(nmin, int):
            raise TypeError(f'Expected \'int\' for nmin={nmin}, got {type(nmin)}')
        if not isinstance(rm, RoundingMode):
            raise TypeError(f'Expected \'RoundingMode\' for rm={rm}, got {type(rm)}')
        self.nmin = nmin
        self.rm = rm

    def with_rm(self, rm: RoundingMode):
        return MPFContext(self.nmin, rm)

    def is_representable(self, x: RealFloat | Float) -> bool:
        if not isinstance(x, RealFloat | Float):
            raise TypeError(f'Expected \'RealFloat\' or \'Float\', got \'{type(x)}\' for x={x}')

        # check for Inf or NaN
        if isinstance(x, Float) and x.is_nar():
            return False

        # extract real part
        match x:
            case Float():
                xr = x.as_real()
            case RealFloat():
                xr = x
            case _:
                raise RuntimeError(f'unreachable {x}')

        return xr.is_more_significant(self.nmin)

    def is_canonical(self, x: Float):
        if not isinstance(x, Float) and self.is_representable(x):
            raise TypeError(f'Expected a representable \'Float\', got \'{type(x)}\' for x={x}')
        return x.exp == self.nmin + 1

    def normalize(self, x: Float):
        if not isinstance(x, Float) and self.is_representable(x):
            raise TypeError(f'Expected a representable \'Float\', got \'{type(x)}\' for x={x}')

        offset = x.exp - (self.nmin + 1)
        if offset > 0:
            # shift the significand to the right
            c = x.c >> offset
            exp = x.exp - offset
        elif offset < 0:
            # shift the significand to the left
            c = x.c << -offset
            exp = x.exp - offset
        else:
            c = x.c
            exp = x.exp

        return Float(exp=exp, c=c, x=x, ctx=self)

    def round_params(self):
        return None, self.nmin

    def _round_float_at(self, x: RealFloat | Float, n: Optional[int]) -> Float:
        """
        Like `self.round_at()` but only for `RealFloat` or `Float` instances.

        Optionally, specify `n` to override the least absolute digit position.
        If `n < self.nmin`, it will be set to `self.nmin`.
        """
        if n is None:
            n = self.nmin
        else:
            n = max(n, self.nmin)

        # step 1. handle special values
        if isinstance(x, Float):
            if x.isnan:
                xr = RealFloat.zero()
                # raise ValueError(f'Cannot round NaN under this context')
            elif x.isinf:
                xr = RealFloat.zero()
                # raise ValueError(f'Cannot round Inf under this context')
            else:
                xr = x._real

        # step 2. shortcut for exact zero values
        if xr.is_zero():
            # exactly zero
            return Float(ctx=self)

        # step 3. round value based on rounding parameters
        return xr.round(min_n=n, rm=self.rm)

    def _round_at(self, x, n: Optional[int]) -> Float:
        match x:
            case Float() | RealFloat():
                xr = x
            case int():
                xr = RealFloat(c=x)
            case float() | str():
                xr = mpfr_value(x, n=self.nmin)
            case Fraction():
                if x.denominator == 1:
                    xr = RealFloat(c=int(x))
                else:
                    xr = mpfr_value(x, n=self.nmin)
            case _:
                raise TypeError(f'not valid argument x={x}')

        return self._round_float_at(xr, n)

    def round(self, x):
        return self._round_at(x, None)

    def round_at(self, x, n: int):
        return self._round_at(x, n)
