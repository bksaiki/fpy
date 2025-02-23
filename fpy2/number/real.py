"""
This module defines the basic floating-point type.
"""

from typing import Optional, Self

from .round import RoundingMode, RoundingDirection
from ..utils import bitmask, Ordering

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

    s: bool = False
    """is the sign negative?"""
    exp: int = 0
    """absolute position of the LSB"""
    c: int = 0
    """integer significand"""

    def __init__(
        self,
        s: Optional[bool] = None,
        exp: Optional[int] = None,
        c: Optional[int] = None,
        x: Optional[Self] = None,
        e: Optional[int] = None,
        m: Optional[int] = None,
    ):
        """
        Creates a new `Real` value.

        The sign may be optionally specified with `s`.
        The exponent may be specified with `exp` or `e`.
        The significand may be specified with `c` or `m` (unless `x` is given).
        If `x` is given, any field not specified is copied from `x`.
        """
        # c and negative
        if c is not None:
            if m is not None:
                raise ValueError(f'cannot specify both c={c} and m={m}')
            self.c = c
            if s is not None:
                self.s = s
            elif x is not None:
                self.s = x.s
            else:
                self.s = type(self).s
        elif m is not None:
            if s is not None:
                raise ValueError(f'cannot specify both m={m} and s={s}')
            self.c = abs(m)
            self.s = m < 0
        elif x is not None:
            self.c = x.c
            if s is not None:
                self.s = s
            else:
                self.s = x.s
        else:
            self.c = type(self).c
            if s is not None:
                self.s = s
            else:
                self.s = type(self).s

        # exp
        if exp is not None:
            if e is not None:
                raise ValueError(f'cannot specify both exp={exp} and e={e}')
            self.exp = exp
        elif e is not None:
            self.exp = e - self.c.bit_length() + 1
        elif x is not None:
            self.exp = x.exp
        else:
            self.exp = type(self).exp


    def __repr__(self):
        return 'Real(s=' + repr(self.s) + ', exp=' + repr(self.exp) + ', c=' + repr(self.c) + ')'

    def __hash__(self):
        if self.c == 0:
            return hash(self.s)
        else:
            return hash((self.s, self.exp, self.c))

    def __eq__(self, other) -> bool:
        return isinstance(other, Real) and self.compare(other) == Ordering.EQUAL

    def __lt__(self, other: Self) -> bool:
        if not isinstance(other, Real):
            return TypeError(f'\'<\' not supported between instances \'Real\' and \'{type(other)}\'')
        return self.compare(other) == Ordering.LESS

    def __le__(self, other: Self) -> bool:
        if not isinstance(other, Real):
            return TypeError(f'\'<=\' not supported between instances \'Real\' and \'{type(other)}\'')
        return self.compare(other) != Ordering.GREATER

    def __gt__(self, other: Self) -> bool:
        if not isinstance(other, Real):
            return TypeError(f'\'>\' not supported between instances \'Real\' and \'{type(other)}\'')
        return self.compare(other) == Ordering.GREATER

    def __ge__(self, other: Self) -> bool:
        if not isinstance(other, Real):
            return TypeError(f'\'>=\' not supported between instances \'Real\' and \'{type(other)}\'')
        return self.compare(other) != Ordering.LESS

    @property
    def base(self):
        """Integer base of this number. Always 2."""
        return 2

    @property
    def p(self):
        """Minimum number of binary digits required to represent this number."""
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
        """Returns whether this value represents zero."""
        return self.c == 0

    def is_integer(self) -> bool:
        """Returns whether this value is an integer."""
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
        """Returns the value of the digit at the `n`th position as a boolean."""
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

    def compare(self, other: Self) -> Ordering:
        """Compare two `Real` values returning an `Ordering`."""
        if not isinstance(other, Real):
            raise TypeError(f"expected Real, got {type(other)}")

        if self.c == 0:
            if other.c == 0:
                return Ordering.EQUAL
            elif other.s:
                return Ordering.GREATER
            else:
                return Ordering.LESS
        elif other.c == 0:
            if self.s:
                return Ordering.LESS
            else:
                return Ordering.GREATER
        elif self.s != other.s:
            # non-zero signs are different
            if self.s:
                return Ordering.LESS
            else:
                return Ordering.GREATER
        else:
            # non-zero, signs are same
            match Ordering.from_compare(self.e, other.e):
                case Ordering.GREATER:
                    # larger magnitude based on MSB
                    cmp = Ordering.GREATER
                case Ordering.LESS:
                    # smaller magnitude based on MSB
                    cmp = Ordering.LESS
                case Ordering.EQUAL:
                    # need to actual compare the significands
                    n = min(self.n, other.n)
                    c1 = self.c << (self.n - n)
                    c2 = other.c << (other.n - n)
                    cmp = Ordering.from_compare(c1, c2)

            # adjust for the sign
            if self.s:
                return cmp.reverse()
            else:
                return cmp

    def is_identical_to(self, other: Self) -> bool:
        """Is the value encoded identically to another `Real` value?"""
        if not isinstance(other, Real):
            return TypeError(f'expected Real, got {type(other)}')
        return self.s == other.s and self.exp == other.exp and self.c == other.c

    def _round_params(self, max_p: Optional[int] = None, min_n: Optional[int] = None):
        """
        Computes rounding parameters `p` and `n`.

        Given `max_p` and `min_n`, computes the actual allowable precision `p`
        and the position of the first unrepresentable digit `n`.
        """
        if max_p is None:
            p = None
            if min_n is None:
                raise ValueError(f'must specify {max_p} or {min_n}')
            else:
                # fixed-point rounding => limited by n
                n = min_n
        else:
            p = max_p
            if min_n is None:
                # floating-point rounding => limited by fixed precision
                n = self.e - max_p
            else:
                # IEEE 754 floating-point rounding
                n = max(min_n, self.e - max_p)

        return p, n

    def _round_at(self, n: int):
        """
        Splits `self` at absolute digit position `n`.

        Computes the digits of `self.c` above digit `n` and the digit
        at position `n` as the "half" bit and a boolean to indicate
        if any digits below position n are 1.
        """

        kept, lost = self.split(n)
        if lost.is_zero():
            # no bits are remaining at or below n
            half_bit = False
            lower_bits = False
        elif lost.e == n:
            # the MSB of lo is at position n
            half_bit = (lost.c & bitmask(lost.p - 1)) != 0
            lower_bits = (lost.c & bitmask(lost.p - 1)) != 0
        else:
            # the MSB of lo is below position n
            half_bit = False
            lower_bits = True

        return kept, half_bit, lower_bits

    def _round_requires_increment(
        self,
        kept: Self,
        half_bit: bool,
        lower_bits: bool,
        nearest: bool,
        direction: RoundingDirection,
    ):
        """
        Does the rounding operation require incrementing truncated digits?
        """



    def _round_finalize(
        self,
        kept: Self,
        half_bit: bool,
        lower_bits: bool,
        p: Optional[int],
        rm: RoundingMode
    ):
        """
        Completes the rounding operation using truncated digits
        and rounding information.
        """

        # convert rounding mode
        nearest, direction = rm.to_direction(kept.s)

        # check if we need to increment to round correctly
        requires_increment = self._round_requires_increment(kept, half_bit, lower_bits, nearest, direction)
        if requires_increment:
            # increment the significand
            kept.c += 1
            rounded = True
            if p is not None and kept.c.bit_length() > p:
                kept.c >>= 1
                kept.exp += 1
        else:
            rounded = False

        raise NotImplementedError


    def round(
        self,
        max_p: Optional[int] = None,
        min_n: Optional[int] = None,
        rm: RoundingMode = RoundingMode.RNE,
    ):
        """
        Rounds `self` to another value with at most `max_p` digits of precision
        or a least absolute digit position `min_n` whichever bound is encountered
        first, using the rounding mode specified by `rm`.

        At least one of `max_p` or `min_n` must be specified:
        `max_p >= 0` while `min_n` may be any integer.

        If only `min_n` is given, rounding is performed like fixed-point
        rounding and the resulting significand may have more than `max_p` bits
        (any values can be clamped after this operation).
        If only `min_p` is given, rounding is performed liked floating-point
        without an exponent bound; the integer significand has at most
        `max_p` digits.
        If both are specified, rounding is performed like IEEE 754 floating-point
        arithmetic; `min_n` takes precedence,so the value may have less than
        `max_p` precision.
        """

        if max_p is None and min_n is None:
            raise ValueError(f'must specify {max_p} or {min_n}')

        # compute rounding parameters
        p, n = self._round_params(max_p, min_n)

        # split the significand and keep rounding bits
        kept, half_bit, lower_bits = self._round_at(n)
        assert p is None or kept.c.bit_length() <= p

        # finalize rounding
        return self._round_finalize(kept, half_bit, lower_bits, p, rm)
