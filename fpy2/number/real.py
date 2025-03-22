"""
This module defines the basic floating-point number type `RealFloat`.
"""

from typing import Optional, Self

from .round import RoundingMode, RoundingDirection
from ..utils import bitmask, default_repr, partial_ord, Ordering

@default_repr
class RealFloat:
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

    This type can also encode uncertainty introduced by rounding.
    The uncertaintly is represented by an interval, also called
    a rounding envelope. The interval includes this value and
    extends either below or above it (`interval_down`).
    The interval always contains this value and may contain
    the other endpoint as well (`interval_closed`).
    The size of the interval is `2**(exp + interval_size)`.
    It must be the case that `interval_size <= 0`.
    """

    s: bool = False
    """is the sign negative?"""
    exp: int = 0
    """absolute position of the LSB"""
    c: int = 0
    """integer significand"""

    interval_size: Optional[int] = None
    """rounding envelope: size relative to `2**exp`"""
    interval_down: bool = False
    """rounding envelope: does the interval extend towards zero?"""
    interval_closed: bool = False
    """rounding envelope: is the interval closed at the other endpoint?"""

    def __init__(
        self,
        s: Optional[bool] = None,
        exp: Optional[int] = None,
        c: Optional[int] = None,
        *,
        x: Optional['RealFloat'] = None,
        e: Optional[int] = None,
        m: Optional[int] = None,
        interval_size: Optional[int] = None,
        interval_down: Optional[bool] = None,
        interval_closed: Optional[bool] = None,
    ):
        """
        Creates a new `RealFloat` value.

        The sign may be optionally specified with `s`.
        The exponent may be specified with `exp` or `e`.
        The significand may be specified with `c` or `m` (unless `x` is given).
        If `x` is given, any field not specified is copied from `x`.
        """
        if x is not None and not isinstance(x, RealFloat):
            raise TypeError(f'expected RealFloat, got {type(x)}')

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

        # rounding envelope size
        if interval_size is not None:
            if interval_size > 0:
                raise ValueError(f'cannot specify interval_size={interval_size}, must be <= 0')
            self.interval_size = interval_size
        elif x is not None:
            self.interval_size = x.interval_size
        else:
            self.interval_size = type(self).interval_size

        # rounding envelope direction
        if interval_down is not None:
            self.interval_down = interval_down
        elif x is not None:
            self.interval_down = x.interval_down
        else:
            self.interval_down = type(self).interval_down

        # rounding envelope endpoint
        if interval_closed is not None:
            self.interval_closed = interval_closed
        elif x is not None:
            self.interval_closed = x.interval_closed
        else:
            self.interval_closed = type(self).interval_closed

    def __eq__(self, other):
        if not isinstance(other, RealFloat):
            return False
        ord = self.compare(other)
        return ord is not None and ord == Ordering.EQUAL

    def __lt__(self, other):
        if not isinstance(RealFloat):
            raise TypeError(f'\'<\' not supported between instances of \'{type(self)}\' \'{type(other)}\'')
        ord = self.compare(other)
        return ord is not None and ord == Ordering.LESS

    def __le__(self, other):
        if not isinstance(RealFloat):
            raise TypeError(f'\'<=\' not supported between instances of \'{type(self)}\' \'{type(other)}\'')
        ord = self.compare(other)
        return ord is not None and ord != Ordering.GREATER

    def __gt__(self, other):
        if not isinstance(RealFloat):
            raise TypeError(f'\'>\' not supported between instances of \'{type(self)}\' \'{type(other)}\'')
        ord = self.compare(other)
        return ord is not None and ord == Ordering.GREATER

    def __ge__(self, other):
        if not isinstance(RealFloat):
            raise TypeError(f'\'>=\' not supported between instances of \'{type(self)}\' \'{type(other)}\'')
        ord = self.compare(other)
        return ord is not None and ord != Ordering.LESS

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
        Normalized exponent of this number.

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

    @property
    def inexact(self) -> bool:
        """Is this value inexact?"""
        return self.interval_size is not None

    def is_zero(self) -> bool:
        """Returns whether this value represents zero."""
        return self.c == 0

    def is_nonzero(self) -> bool:
        """Returns whether this value does not represent zero."""
        return self.c != 0

    def is_positive(self) -> bool:
        """Returns whether this value is positive."""
        return self.c != 0 and not self.s

    def is_negative(self) -> bool:
        """Returns whether this value is negative."""
        return self.c != 0 and self.s

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

    def normalize(self, p: int, n: Optional[int] = None):
        """
        Returns a copy of `self` that has exactly `p` bits of precision.
        Optionally, specify `n` to ensure that if `y = x.normalize(p, n)`,
        then `y.exp > n` or `y` is zero.

        For non-zero values, raises a `ValueError` if any significand digits
        are shifted off, i.e., `x != x.normalize(p, n)`.
        """
        if not isinstance(p, int) or p < 0:
            raise ValueError('expected a non-negative integer', p)

        # special case: 0 has no precision
        if self.is_zero():
            return RealFloat()

        # compute maximum shift and resulting exponent
        shift = p - self.p
        exp = self.exp - shift

        # test if exponent is below `n`
        if n is not None and exp <= n:
            # too small, so adjust accordingly
            adjust = n - exp + 1
            shift -= adjust
            exp += adjust

        # compute new significand `c`
        if shift >= 0:
            # shifting left by a non-negative amount
            c = self.c << shift
        else:
            # shift right by a positive amount
            shift = -shift
            c = self.c >> shift
            # check that we didn't lose significant digits
            if (self.c & bitmask(shift)) != 0:
                raise ValueError(f'shifting off digits: p={p}, n={n}, x={self}')

        # return result
        return RealFloat(self.s, exp, c)


    def split(self, n: int):
        """
        Splits `self` into two `RealFloat` values where the first value represents
        the digits above `n` and the second value represents the digits below
        and including `n`.
        """
        if not isinstance(n, int):
            raise ValueError('expected an integer', n)

        if self.is_zero():
            # special case: 0 has no precision
            hi = RealFloat(self.s, n + 1, 0)
            lo = RealFloat(self.s, n, 0)
            return (hi, lo)
        elif n >= self.e:
            # check if all digits are in the lower part
            hi = RealFloat(self.s, n + 1, 0)
            lo = RealFloat(self.s, self.exp, self.c)
            return (hi, lo)
        elif n < self.exp:
            # check if all digits are in the upper part
            hi = RealFloat(self.s, self.exp, self.c)
            lo = RealFloat(self.s, n, 0)
            return (hi, lo)
        else:
            # splitting the digits
            p_lo = (n + 1) - self.exp
            mask_lo = bitmask(p_lo)

            exp_hi = self.exp + p_lo
            c_hi = self.c >> p_lo

            exp_lo = self.exp
            c_lo = self.c & mask_lo

            hi = RealFloat(self.s, exp_hi, c_hi)
            lo = RealFloat(self.s, exp_lo, c_lo)
            return (hi, lo)

    def compare(self, other: 'RealFloat'):
        """
        Compare `self` and `other` values returning an `Optional[Ordering]`.

        For two `RealFloat` values, the result is never `None`.
        """
        if not isinstance(other, RealFloat):
            raise TypeError(f'comparison not supported between \'RealFloat\' and \'{type(other)}\'')

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
        """Is the value encoded identically to another `RealFloat` value?"""
        if not isinstance(other, RealFloat):
            return TypeError(f'expected RealFloat, got {type(other)}')

        return (
            self.s == other.s
            and self.exp == other.exp
            and self.c == other.c
            and self.interval_size == other.interval_size
            and self.interval_down == other.interval_down
            and self.interval_closed == other.interval_closed
        )


    def next_away(self):
        """
        Computes the next number (with the same precision),
        away from zero.
        """
        c = self.c + 1
        exp = self.exp
        if c.bit_length() > self.p:
            # adjust the exponent since we exceeded precision bounds
            # the value is guaranteed to be a power of two
            c >>= 1
            exp  += 1

        return RealFloat(s=self.s, c=c, exp=exp)

    def next_towards(self):
        """
        Computes the previous number (with the same precision),
        towards zero.
        """
        c = self.c - 1
        exp = self.exp
        if c.bit_length() < self.p:
            # previously at a power of two
            # need to add a lower bit
            c = (c << 1) | 1
            exp -= 1

        return RealFloat(s=self.s, c=c, exp=exp)

    def next_up(self):
        """
        Computes the next number (with the same precison),
        towards positive infinity.
        """
        if self.s:
            return self.next_towards()
        else:   
            return self.next_away()

    def next_down(self):
        """
        Computes the previous number (with the same precision),
        towards negative infinity.
        """
        if self.s:
            return self.next_away()
        else:
            return self.next_towards()


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

    def _round_direction(
        self,
        kept: Self,
        half_bit: bool,
        lower_bits: bool,
        rm: RoundingMode,
    ):
        """
        Determines the direction to round based on the rounding mode.
        Also computes the rounding envelope.
        """

        # convert the rounding mode to a direction
        nearest, direction = rm.to_direction(kept.s)

        # rounding envelope
        interval_size: Optional[int] = None
        interval_closed: bool = False
        increment: bool = False

        # case split on nearest mode
        if nearest:
            # nearest rounding mode
            # case split on halfway bit
            if half_bit:
                # at least halfway
                interval_size = -1
                if lower_bits:
                    # above halfway
                    increment = True
                else:
                    # exact halfway
                    interval_closed = True
                    match direction:
                        case RoundingDirection.RTZ:
                            increment = False
                        case RoundingDirection.RAZ:
                            increment = True
                        case RoundingDirection.RTE:
                            is_even = (kept.c & 1) == 0
                            increment = not is_even
                        case RoundingDirection.RTO:
                            is_even = (kept.c & 1) == 0
                            increment = is_even
            else:
                # below halfway
                increment = False
                interval_closed = False
                if lower_bits:
                    # inexact
                    interval_size = -1
                else:
                    # exact
                    interval_size = None
        else:
            # non-nearest rounding mode
            interval_closed = False
            if half_bit or lower_bits:
                # inexact
                interval_size = 0
                match direction:
                    case RoundingDirection.RTZ:
                        increment = False
                    case RoundingDirection.RAZ:
                        increment = True
                    case RoundingDirection.RTE:
                        is_even = (kept.c & 1) == 0
                        increment = not is_even
                    case RoundingDirection.RTO:
                        is_even = (kept.c & 1) == 0
                        increment = is_even
            else:
                # exact
                interval_size = None
                increment = False

        return interval_size, interval_closed, increment

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
        and additional rounding information.
        """

        # prepare the rounding operation
        interval_size, interval_closed, increment = self._round_direction(kept, half_bit, lower_bits, rm)

        # increment if necessary
        if increment:
            kept.c += 1
            if p is not None and kept.c.bit_length() > p:
                # adjust the exponent since we exceeded precision bounds
                # the value is guaranteed to be a power of two
                kept.c >>= 1
                kept.exp += 1
                interval_size -= 1

        # interval direction is opposite of if we incremented
        interval_down = not increment

        # return the rounded value
        return RealFloat(
            x=kept,
            interval_size=interval_size,
            interval_down=interval_down,
            interval_closed=interval_closed
        )

    def round_at(self, n: int):
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


    def round(
        self,
        max_p: Optional[int] = None,
        min_n: Optional[int] = None,
        rm: RoundingMode = RoundingMode.RNE,
    ):
        """
        Creates a copy of `self` rounded to at most `max_p` digits of precision
        or a least absolute digit position `min_n`, whichever bound is encountered first,
        using the rounding mode specified by `rm`.

        At least one of `max_p` or `min_n` must be specified:
        `max_p >= 0` while `min_n` may be any integer.

        If only `min_n` is given, rounding is performed like fixed-point
        rounding and the resulting significand may have more than `max_p` bits
        (any values can be clamped after this operation).
        If only `min_p` is given, rounding is performed liked floating-point
        without an exponent bound; the integer significand has at
        most `max_p` digits.
        If both are specified, rounding is performed like IEEE 754 floating-point
        arithmetic; `min_n` takes precedence, so the value may have
        less than `max_p` precision.
        """

        if max_p is None and min_n is None:
            raise ValueError(f'must specify {max_p} or {min_n}')

        # step 1. compute rounding parameters
        p, n = self._round_params(max_p, min_n)

        # step 2. split the number at the rounding position
        kept, half_bit, lower_bits = self.round_at(n)

        # step 3. finalize the rounding operation
        return self._round_finalize(kept, half_bit, lower_bits, p, rm)
