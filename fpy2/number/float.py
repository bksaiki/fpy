"""
This module defines the basic floating-point number type
with infinity and NaN values.
"""

from typing import Optional

from .real import RealFloat
from ..utils import Ordering


class UnboundFloat(RealFloat):
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
    Unlike `RealFloat`, this type allows for infinite and NaN values.
    """

    is_inf: bool = False
    """is this value infinity?"""
    is_nan: bool = False
    """is this value NaN?"""

    def __init__(
        self,
        *args,
        x: Optional[RealFloat] = None,
        is_inf: Optional[bool] = None,
        is_nan: Optional[bool] = None,
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

        # is_nan
        if is_nan is not None:
            self.is_nan = is_nan
        elif x is not None and isinstance(x, UnboundFloat):
            self.is_nan = x.is_nan
        else:
            self.is_nan = type(self).is_nan

    def compare(self, other: RealFloat) -> Optional[Ordering]:
        """
        Compare this and a `RealFloat` instance returning an `Optional[Ordering]`.

        The result is `None` if either number is NaN.
        Otherwise, the result is the expected ordering.
        """
        raise NotImplementedError
