"""
This module defines the basic floating-point number type
with infinity and NaN values.
"""

from typing import Optional

from .proj import ProjFloat
from .real import RealFloat
from .round import RoundingMode
from ..utils import Ordering


class UnboundFloat(ProjFloat):
    """
    A basic floating-point number with infinity and NaN values.
    This type is a subtype of `ProjFloat`.

    This type encodes a base-2 number in unnormalized scientific notation:
    `(-1)^s * 2^exp * c` where:
     - `s` is the sign;
     - `exp` is the absolute position of the least-significant bit (LSB),
       also called the unnormalized exponent; and
     - `c` is the integer significand.

    There are no constraints on the values of `exp` and `c`.
    Type allows for infinite and NaN values.
    """

    is_inf: bool = False
    """is this value infinity?"""

    def __init__(
        self,
        *args,
        x: Optional[RealFloat] = None,
        is_inf: Optional[bool] = None,
        **kwargs
    ):
        if x is not None and not isinstance(x, RealFloat):
            raise TypeError(f'expected RealFloat, got {type(x)}')

        super().__init__(*args, x=x, **kwargs)

        # is_inf
        if is_inf is not None:
            self.is_inf = is_inf
        elif x is not None and isinstance(x, UnboundFloat):
            self.is_inf = x.is_inf
        else:
            self.is_inf = type(self).is_inf

    def is_nar(self) -> bool:
        return not self.is_inf and super().is_nar()

    def compare(self, other):
        if self.is_nan or isinstance(other, UnboundFloat) and other.is_nan:
            return None
        elif self.is_inf:
            if isinstance(other, UnboundFloat) and other.is_inf:
                if self.s == other.s:
                    return Ordering.EQUAL
                else:
                    if self.s:
                        return Ordering.LESS
                    else:
                        return Ordering.GREATER
            else:
                if self.s:
                    return Ordering.LESS
                else:
                    return Ordering.GREATER
        elif isinstance(other, UnboundFloat) and other.is_inf:
            if other.s:
                return Ordering.GREATER
            else:
                return Ordering.LESS
        else:
            # OPT: already handled NaN so use RealFloat.compare
            return RealFloat.compare(self, other)

    def round(
        self,
        max_p: Optional[int] = None,
        min_n: Optional[int] = None,
        rm = RoundingMode.RNE
    ):
        if max_p is None and min_n is None:
            raise ValueError(f'must specify {max_p} or {min_n}')

        if self.is_nar():
            return UnboundFloat(x=self)
        else:
            return super().round(max_p, min_n, rm)
