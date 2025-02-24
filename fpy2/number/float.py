"""
This module defines the basic floating-point number `Float` class.
"""

from typing import Optional, Self

from .real import RealFloat


class UnboundFloat(RealFloat):
    """
    An unbounded floating-point number with infinites and NaN.
    This type is a subtype of `RealFloat`.

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

    def __repr__(self):
      return self.__class__.__name__ + \
          '(s=' + repr(self.s) + \
          ', exp=' + repr(self.exp) + \
          ', c=' + repr(self.c) + \
          ', is_inf=' + repr(self.is_inf) + \
          ', is_nan=' + repr(self.is_nan) + \
          ', interval_size=' + repr(self.interval_size) + \
          ', interval_down=' + repr(self.interval_down) + \
          ', interval_closed=' + repr(self.interval_closed) + \
      ')'
