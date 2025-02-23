"""
This module defines the basic floating-point type.
"""

from typing import Self

from ..utils import bitmask

class Real:
    """
    The basic floating-point number.

    This type encodes a base-2 number in unnormalized scientific notation:

    `(-1)^s * 2^exp * c` where:
     - `s` is the sign;
     - `exp` is the absolute position of the least-significant bit (LSB),
       also called the unnormalized exponent; and
     - `c` is the integer significand.

    There are no constraints on the values of `exp` and `c`.
    Unlike IEEE 754, this number cannot encode infinity or NaN.
    """

    s: bool
    """is the sign negative?"""
    exp: int
    """absolute position of the LSB"""
    c: int
    """integer significand"""

    def __init__(self, s: bool, exp: int, c: int):
        if not isinstance(s, bool):
            raise TypeError(f"expected bool, got {type(s)} for `s`")
        if not isinstance(exp, int):
            raise TypeError(f"expected int, got {type(exp)} for `exp`")
        if not isinstance(c, int):
            raise TypeError(f"expected int, got {type(c)} for `c`")

        self.s = s
        self.exp = exp
        self.c = c

    def __repr__(self):
        return 'Real(s=' + repr(self.s) + ', exp=' + repr(self.exp) + ', c=' + repr(self.c) + ')'

    @property
    def base(self):
        """
        Integer base of this number. Always 2.
        """
        return 2

    @property
    def p(self):
        """
        Minimum number of binary digits required to represent this number.
        """
        return self.c.bit_length()

    @property
    def e(self) -> int:
        """
        Unnormalized exponent of this number.

        When `self.c == 0` (i.e. the number is zero), this method returns
        `self.exp - 1`. In other words, `self.c != 0` iff `self.e >= self.exp`.

        The interval `[self.exp, self.e]` represents the absolute positions
        of digits in the significand.
        """
        return self.exp + self.p - 1

    @property
    def n(self) -> int:
        """
        Position of the first unrepresentable digit below the significant digits.
        This is exactly `self.exp - 1`.
        """
        return self.exp - 1

    @property
    def m(self) -> int:
        """
        Signed significand.
        This is exactly `(-1)^self.s * self.c`.
        """
        return -self.c if self.s else self.c

    def is_zero(self) -> bool:
        """
        Returns whether this value represents zero.
        """
        return self.c == 0

    def is_integer(self) -> bool:
        """
        Returns whether this value is an integer.
        """
        if self.is_zero():
            return True

        # all significant digits are integer digits
        if self.exp >= 0:
            return True

        # all non-significant digits are fractional digits
        if self.e < 0:
            return False

        # must check if fractional digits are zero
        fbits = self.c & bitmask(-self.exp)
        return fbits == 0

    def bit(self, n: int) -> bool:
        """
        Returns the value of the digit at the `n`th position as a boolean.
        """
        if not isinstance(n, int):
            raise ValueError('expected an integer', n)

        # special case: 0 has no digits set
        if self.is_zero():
            return False

        # below the region of significance
        if n < self.exp:
            return False

        # above the region of significane
        if n > self.e:
            return False

        idx = n - self.exp
        bit = self.c & (1 << idx)
        return bit != 0

    def normalize(self, p: int):
        """
        Returns a copy of `self` that has exactly `p` bits of precision.
        If `p < self.p`, a `ValueError` is thrown.
        """
        if not isinstance(p, int) or p < 0:
            raise ValueError('expected a non-negative integer', p)

        # special case: 0 has no precision ever
        if self.is_zero():
            return self

        # check that the requested precision is enough
        if p < self.p:
            raise ValueError('insufficient precision', self, p)

        shift = p - self.p
        exp = self.exp - shift
        c = self.c << shift
        return Real(self.s, exp, c)

    def split(self, n: int):
        """
        Splits `self` into two `Real` values where the first value represents
        the digits above `n` and the second value represents the digits below
        and including `n`.
        """
        if not isinstance(n, int):
            raise ValueError('expected an integer', n)

        # special case: 0 has no precision
        if self.is_zero():
            hi = Real(self.s, n + 1, 0)
            lo = Real(self.s, n, 0)
            return (hi, lo)

        assert self.e is not None

        # check if all digits are in the lower part
        if n >= self.e:
            hi = Real(self.s, n + 1, 0)
            lo = Real(self.s, self.exp, self.c)
            return (hi, lo)

        # check if all digits are in the upper part
        if n < self.exp:
            hi = Real(self.s, self.exp, self.c)
            lo = Real(self.s, n, 0)
            return (hi, lo)

        # splitting the digits
        p_lo = (n + 1) - self.exp
        mask_lo = bitmask(p_lo)

        exp_hi = self.exp + p_lo
        c_hi = self.c >> p_lo

        exp_lo = self.exp
        c_lo = self.c & mask_lo

        hi = Real(self.s, exp_hi, c_hi)
        lo = Real(self.s, exp_lo, c_lo)
        return (hi, lo)
