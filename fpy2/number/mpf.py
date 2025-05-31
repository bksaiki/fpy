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

@default_repr
class MPFContext(Context):
    """
    Rounding context for mulit-precision fixed-point numbers.

    This context is parameterized by the most significant digit
    that is not representable `nmin` and a rounding mode `rm`.
    It emulates fixed-point numbers with arbitrary precision.
    """

    nmin: int
    """the first unreprsentable digit"""

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

        # case split on class
        if xr.is_zero():
            # special values and zeros are valid
            return True
        elif xr.exp > self.nmin:
            # all digits are above `nmin`
            return True
        elif xr.e <= self.nmin:
            # all digits are below `nmin`
            return False
        else:
            # need to check the digits at or below `nmin`
            _, lo = xr.split(self.nmin)
            return lo.is_zero()


    def is_canonical(self, x):
        raise NotImplementedError

    def normalize(self, x):
        raise NotImplementedError

    def round_params(self):
        raise NotImplementedError

    def round(self, x):
        raise NotImplementedError

    def round_at(self, x, n):
        raise NotImplementedError
