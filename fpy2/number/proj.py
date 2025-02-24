"""
This module defines the basic floating-point number type
with a NaN value.
"""

from typing import Optional

from .real import RealFloat
from ..utils import Ordering

class ProjFloat(RealFloat):
    """
    A basic floating-point number with a NaN value,
    or a "projective" floating-point number.
    This type is a subtype of `RealFloat`.

    This type encodes a base-2 number in unnormalized scientific notation:
    `(-1)^s * 2^exp * c` where:
     - `s` is the sign;
     - `exp` is the absolute position of the least-significant bit (LSB),
       also called the unnormalized exponent; and
     - `c` is the integer significand.

    There are no constraints on the values of `exp` and `c`.
    This type allows for a NaN value.
    """

    is_nan: bool = False
    """is this value NaN?"""

    def __init__(
        self,
        *args,
        x: Optional[RealFloat] = None,
        is_nan: Optional[bool] = None,
        **kwargs
    ):
        """
        Creates a new `ProjFloat` value.

        See `RealFloat.__init__` for arguments passed to the superclass.

        You can optionally specify `is_nan` to construct a NaN value.
        """
        if x is not None and not isinstance(x, RealFloat):
            raise TypeError(f'expected RealFloat, got {type(x)}')

        super().__init__(*args, x=x, **kwargs)

        # is_nan
        if is_nan is not None:
            self.is_nan = is_nan
        elif x is not None and isinstance(x, ProjFloat):
            self.is_nan = x.is_nan
        else:
            self.is_nan = type(self).is_nan

    def is_zero(self):
        return not self.is_nan and super().is_zero()

    def is_finite_real(self):
        # OPT: the super class is only finite real
        return not self.is_nan

    def is_nar(self):
        # OPT: the super class is only finite real
        return self.is_nan

    def is_integer(self):
        return not self.is_nan and super().is_integer()

    def bit(self, n: int) -> bool:
        if self.is_nan:
            raise TypeError('cannot call bit() on a NaN value')
        return super().bit(n)

    def normalize(self, p: int):
        if self.is_nan:
            raise TypeError('cannot call normalize() on a NaN value')
        return super().normalize(p)

    def split(self, p: int):
        if self.is_nan:
            raise TypeError('cannot call split() on a NaN value')
        return super().split(p)

    def compare(self, other: RealFloat) -> Optional[Ordering]:
        if not isinstance(other, RealFloat):
            raise TypeError(f"expected RealFloat, got {type(other)}")

        if self.is_nan:
            return None
        elif isinstance(other, ProjFloat) and other.is_nan:
            return None
        else:
            return super().compare(other)
