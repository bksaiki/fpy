"""
This module defines two floating-point number types.

`RealFloat` - a floating-point number without infinities and NaN.
`Float` - `RealFloat` extended with infinities and NaN.
"""

import math
import numbers

from typing import Optional, Self

from ..utils import (
    bitmask,
    default_repr,
    float_to_bits,
    rcomparable,
    Ordering,
    FP64_NBITS,
    FP64_ES,
    FP64_M,
    FP64_EXPMIN
)

from .context import Context
from .globals import get_current_float_converter, get_current_str_converter
from .round import RoundingMode, RoundingDirection

###########################################################
# RealFloat

@default_repr
class RealFloat(numbers.Rational):
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
            if c < 0:
                raise ValueError(f'c={c} must be non-negative')
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

    def __str__(self):
        fn = get_current_str_converter()
        return fn(self)

    def __eq__(self, other):
        if not isinstance(other, RealFloat):
            return False
        ord = self.compare(other)
        return ord is not None and ord == Ordering.EQUAL

    def __lt__(self, other):
        if not isinstance(other, RealFloat):
            raise TypeError(f'\'<\' not supported between instances of \'{type(self)}\' \'{type(other)}\'')
        ord = self.compare(other)
        return ord is not None and ord == Ordering.LESS

    def __le__(self, other):
        if not isinstance(other, RealFloat):
            raise TypeError(f'\'<=\' not supported between instances of \'{type(self)}\' \'{type(other)}\'')
        ord = self.compare(other)
        return ord is not None and ord != Ordering.GREATER

    def __gt__(self, other):
        if not isinstance(other, RealFloat):
            raise TypeError(f'\'>\' not supported between instances of \'{type(self)}\' \'{type(other)}\'')
        ord = self.compare(other)
        return ord is not None and ord == Ordering.GREATER

    def __ge__(self, other):
        if not isinstance(other, RealFloat):
            raise TypeError(f'\'>=\' not supported between instances of \'{type(self)}\' \'{type(other)}\'')
        ord = self.compare(other)
        return ord is not None and ord != Ordering.LESS

    def __add__(self, other: 'RealFloat'):
        """
        Adds `self` and `other` exactly.

        This operation never fails when `other` is a `RealFloat`.
        """
        if not isinstance(other, RealFloat):
            raise TypeError(f'unsupported operand type(s) for +: \'RealFloat\' and \'{type(other)}\'')

        if self.c == 0:
            # 0 + b = b
            return RealFloat(x=other)
        elif other.c == 0:
            # a + 0 = a
            return RealFloat(x=self)
        else:
            # adding non-zero values

            # compute the smallest exponent and normalize
            exp = min(self.exp, other.exp)

            # normalize significands relative to `exp`
            c1 = self.c << (self.exp - exp)
            c2 = other.c << (other.exp - exp)

            # apply signs
            m1 = -c1 if self.s else c1
            m2 = -c2 if other.s else c2

            # add/subtract
            m = m1 + m2

            # decompose into `s` and `c`
            s = m < 0
            c = -m if s else m

            # return the result
            return RealFloat(s=s, exp=exp, c=c)


    def __radd__(self, other):
        return self + other

    def __neg__(self):
        """
        Unary minus.

        Returns this `RealFloat` with opposite sign (`self.s`)
        even when `self.is_zero()`.
        """
        return RealFloat(s=not self.s, x=self)

    def __pos__(self):
        """
        Unary plus. 

        Returns a copy of `self`.
        """
        return RealFloat(x=self)

    def __mul__(self, other: 'RealFloat'):
        """
        Multiplies `self` and `other` exactly.

        This operation never fails when `other` is a `RealFloat`.
        """
        if not isinstance(other, RealFloat):
            raise TypeError(f'unsupported operand type(s) for *: \'RealFloat\' and \'{type(other)}\'')

        s = self.s != other.s
        if self.c == 0 or other.c == 0:
            # 0 * b = 0 or a * 0 = 0
            # respects signedness
            return RealFloat(s=s)
        else:
            # multiplying non-zero values
            exp = self.exp + other.exp
            c = self.c * other.c
            return RealFloat(s=s, exp=exp, c=c)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        raise NotImplementedError('division cannot be implemented exactly')

    def __rtruediv__(self, other):
        raise NotImplementedError('division cannot be implemented exactly')

    def __pow__(self, exponent):
        """
        Raising `self` by `exponent` exactly.

        This operation is only valid for `exponent` of type `int` with `exponent >= 0`.
        """
        if not isinstance(exponent, int):
            raise TypeError(f'unsupported operand type(s) for **: \'RealFloat\' and \'{type(exponent)}\'')
        if exponent < 0:
            raise ValueError('negative exponent unsupported; cannot be implemented exactly')

        if exponent == 0:
            # b ** 0 = 1
            return RealFloat(c=1)
        else:
            # exponent > 0
            s = self.s and (exponent % 2 == 1)
            exp = self.exp * exponent
            c = self.c ** exponent
            return RealFloat(s=s, exp=exp, c=c)

    def __rpow__(self, base):
        raise NotImplementedError

    def __abs__(self):
        """
        Absolute value.

        Returns this `RealFloat` with `self.s = False`.
        """
        return RealFloat(s=False, x=self)

    def __trunc__(self):
        return self.round(min_n=-1, rm=RoundingMode.RTZ)

    def __floor__(self):
        return self.round(min_n=-1, rm=RoundingMode.RTN)

    def __ceil__(self):
        return self.round(min_n=-1, rm=RoundingMode.RTP)

    def __round__(self, ndigits=None):
        if ndigits is not None:
            if not isinstance(ndigits, int):
                raise TypeError(f'Expected \'int\' for ndigits, got {type(ndigits)}')
            if ndigits != 0:
                raise ValueError('Non-zero ndigits not supported')
            return self.round(max_p=ndigits, rm=RoundingMode.RNE)

        return self.round(min_n=-1, rm=RoundingMode.RNE)

    def __floordiv__(self, other):
        raise NotImplementedError('division cannot be implemented exactly')

    def __rfloordiv__(self, other):
        raise NotImplementedError('division cannot be implemented exactly')

    def __mod__(self, other):
        raise NotImplementedError('modulus cannot be implemented exactly')

    def __rmod__(self, other):
        raise NotImplementedError('modulus cannot be implemented exactly')

    def __float__(self):
        """
        Casts this value exactly to a native Python float.

        If the value is not representable, a `ValueError` is raised.
        """
        fn = get_current_float_converter()
        return fn(self)

    def __int__(self):
        """
        Casts this value exactly to a native Python int.

        If the value is not representable, a `ValueError` is raised.
        """
        if not self.is_integer():
            raise ValueError(f'cannot convert to int: {self}')

        # special case: 0
        if self.c == 0:
            return 0

        if self.exp >= 0:
            # `self.c` consists of integer digits
            c = self.c << self.exp
        else:
            # `self.c` consists of fractional digits
            # but safe to just shift them off
            c = self.c >> -self.exp

        return (-1 if self.s else 1) * c

    @staticmethod
    def from_int(x: int):
        """
        Creates a new `RealFloat` value from a Python `int`.

        This conversion is exact.
        """
        if not isinstance(x, int):
            raise TypeError(f'expected int, got {type(x)}')

        s = x < 0
        c = abs(x)
        return RealFloat(s=s, exp=0, c=c)

    @staticmethod
    def from_float(x: float):
        """
        Creates a new `RealFloat` value from a Python `float`.

        This conversion is exact.
        """
        if not isinstance(x, float):
            raise TypeError(f'expected float, got {type(x)}')

        # convert to bits
        b = float_to_bits(x)
        sbits = b >> (FP64_NBITS - 1)
        ebits = (b >> FP64_M) & bitmask(FP64_ES)
        mbits = b & bitmask(FP64_M)

        # sign
        s = sbits != 0

        # case split on exponent
        if ebits == 0:
            # zero / subnormal
            return RealFloat(s=s, exp=FP64_EXPMIN, c=mbits)
        elif ebits == bitmask(FP64_ES):
            # infinity / NaN
            raise ValueError(f'expected finite float, got x={x}')
        else:
            # normal
            exp = FP64_EXPMIN + (ebits - 1)
            c = (1 << FP64_M) | mbits
            return RealFloat(s=s, exp=exp, c=c)

    @staticmethod
    def zero(s: bool = False):
        """
        Creates a new `RealFloat` value representing zero.

        The sign may be specified with `s`.
        """
        return RealFloat(s=s, exp=0, c=0)

    @staticmethod
    def one(s: bool = False):
        """
        Creates a new `RealFloat` value representing one.

        The sign may be specified with `s`.
        """
        return RealFloat(s=s, exp=0, c=1)

    @staticmethod
    def power_of_2(exp: int, s: bool = False):
        """
        Creates a new `RealFloat` value representing `2**exp`.

        The sign may be specified with `s`.
        """
        if not isinstance(exp, int):
            raise TypeError(f'expected integer exponent, got {type(exp)}')
        return RealFloat(s=s, exp=exp, c=1)


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

    @property
    def numerator(self):
        if self.c == 0:
            # case: value is zero
            return 0
        elif self.exp >= 0:
            # case: value is definitely an integer
            return self.c << self.exp
        else:
            # case: fractional digits

            # compute gcd
            numerator = self.c
            denominator = (1 << -self.exp)
            gcd = math.gcd(numerator, denominator)

            # divide numerator
            return numerator // gcd

    @property
    def denominator(self):
        if self.c == 0 or self.exp >= 0:
            # case: value is zero or definitely an integer
            return 1
        else:
            # case: fractional digits

            # compute gcd
            numerator = self.c
            denominator = (1 << -self.exp)
            gcd = math.gcd(numerator, denominator)

            # divide numerator
            return denominator // gcd

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

    def is_more_significant(self, n: int) -> bool:
        """
        Returns `True` iff this value only has significant digits above `n`,
        that is, every non-zero digit in the number is more significant than `n`.

        When `n = -1`, this method is equivalent to `is_integer()`.

        This method is equivalent to::

            _, lo = self.split(n)
            return lo.is_zero()
        """
        if self.is_zero():
            return True

        # All significant digits are above n
        if self.exp > n:
            return True

        # All significant digits are at or below n
        if self.e <= n:
            return False

        # Some digits may be at or below n; check if those are zero
        n_relative = n - self.exp
        return (self.c & bitmask(n_relative + 1)) == 0

    def is_integer(self) -> bool:
        """
        Returns whether this value is an integer.
        """
        return self.is_more_significant(-1)

    def bit(self, n: int) -> bool:
        """
        Returns the value of the digit at the `n`-th position as a boolean.
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
            expmin = n + 1
            adjust = expmin - exp
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
            half_bit = (lost.c >> (lost.p - 1)) != 0
            lower_bits = (lost.c & bitmask(lost.p - 1)) != 0
        else:
            # the MSB of lo is below position n
            half_bit = False
            lower_bits = True

        return kept, half_bit, lower_bits

    def round_at_exact(self, n: int):
        """
        Like `self.round_at()` except that the result must be exact.
        Raises a `ValueError` if the result is not exact.

        See `self.round_at()` for more information.
        """

        kept, lost = self.split(n)
        if not lost.is_zero():
            raise ValueError(f'rounding off digits: n={n}, x={self}')
        return kept


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

    def round_exact(
        self,
        max_p: Optional[int] = None,
        min_n: Optional[int] = None
    ):
        """
        Like `self.round()` except the result must be exact.
        Raises a `ValueError` if the result is not exact.

        See `self.round()` for more information.
        """
        if max_p is None and min_n is None:
            raise ValueError(f'must specify {max_p} or {min_n}')

        # step 1. compute rounding parameters
        _, n = self._round_params(max_p, min_n)

        # step 2. split the number at the rounding position
        return self.round_at_exact(n)


###########################################################
# Float

@rcomparable(RealFloat)
class Float:
    """
    The basic floating-point number extended with infinities and NaN.

    This type encodes a base-2 number in unnormalized scientific
    notation `(-1)^s * 2^exp * c` where:

    - `s` is the sign;
    - `exp` is the absolute position of the least-significant bit (LSB),
      also called the unnormalized exponent; and
    - `c` is the integer significand.

    There are no constraints on the values of `exp` and `c`.
    Unlike `RealFloat`, this number can encode infinity and NaN.

    This type can also encode uncertainty introduced by rounding.
    The uncertaintly is represented by an interval, also called
    a rounding envelope. The interval includes this value and
    extends either below or above it (`interval_down`).
    The interval always contains this value and may contain
    the other endpoint as well (`interval_closed`).
    The size of the interval is `2**(exp + interval_size)`.
    It must be the case that `interval_size <= 0`.

    Instances of `Float` are usually constructed under
    some rounding context, i.e., the result of an operation with rounding.
    The attribute `ctx` stores that rounding context if one exists.
    In general, `Float` objects should not be manually constructed,
    but rather through context-based constructors.
    """

    isinf: bool = False
    """is this number is infinite?"""

    isnan: bool = False
    """is this number is NaN?"""

    ctx: Optional[Context] = None
    """rounding context during construction"""

    _real: RealFloat
    """the real number (if it is real)"""

    def __init__(
        self,
        s: Optional[bool] = None,
        exp: Optional[int] = None,
        c: Optional[int] = None,
        *,
        x: Optional[RealFloat | Self] = None,
        e: Optional[int] = None,
        m: Optional[int] = None,
        isinf: Optional[bool] = None,
        isnan: Optional[bool] = None,
        interval_size: Optional[int] = None,
        interval_down: Optional[bool] = None,
        interval_closed: Optional[bool] = None,
        ctx: Optional[Context] = None
    ):
        if x is not None and not isinstance(x, RealFloat | Float):
            raise TypeError(f'expected Float, got {type(x)}')

        if isinf is not None:
            self.isinf = isinf
        elif isinstance(x, Float):
            self.isinf = x.isinf
        else:
            self.isinf = type(self).isinf

        if isnan is not None:
            self.isnan = isnan
        elif isinstance(x, Float):
            self.isnan = x.isnan
        else:
            self.isnan = type(self).isnan

        if self.isinf and self.isnan:
            raise ValueError('cannot be both infinite and NaN')

        if ctx is not None:
            self.ctx = ctx
        elif isinstance(x, Float):
            self.ctx = x.ctx
        else:
            self.ctx = type(self).ctx

        if isinstance(x, RealFloat):
            real = x
        elif isinstance(x, Float):
            real = x._real
        else:
            real = None

        self._real = RealFloat(
            s=s,
            exp=exp,
            c=c,
            x=real,
            e=e,
            m=m,
            interval_size=interval_size,
            interval_down=interval_down,
            interval_closed=interval_closed
        )

    def __repr__(self):
        return (f'{self.__class__.__name__}('
            + 's=' + repr(self._real.s)
            + ', exp=' + repr(self._real.exp)
            + ', c=' + repr(self._real.c)
            + ', isinf=' + repr(self.isinf)
            + ', isnan=' + repr(self.isnan)
            + ', interval_size=' + repr(self._real.interval_size)
            + ', interval_down=' + repr(self._real.interval_size)
            + ', interval_closed=' + repr(self._real.interval_closed)
            + ', ctx=' + repr(self.ctx)
            + ')'
        )

    def __str__(self):
        fn = get_current_str_converter()
        return fn(self)

    def __eq__(self, other):
        ord = self.compare(other)
        return ord is not None and ord == Ordering.EQUAL

    def __lt__(self, other):
        ord = self.compare(other)
        return ord is not None and ord == Ordering.LESS

    def __le__(self, other):
        ord = self.compare(other)
        return ord is not None and ord != Ordering.GREATER

    def __gt__(self, other):
        ord = self.compare(other)
        return ord is not None and ord == Ordering.GREATER

    def __ge__(self, other):
        ord = self.compare(other)
        return ord is not None and ord != Ordering.LESS

    def __float__(self):
        """
        Casts this value exactly to a native Python float.

        If the value is not representable, a `ValueError` is raised.
        """
        fn = get_current_float_converter()
        return fn(self)

    def __int__(self):
        """
        Casts this value exactly to a native Python integer.

        If the value is not representable, a `ValueError` is raised.
        """
        if not self.is_integer():
            raise ValueError(f'{self} is not an integer')
        return int(self._real)

    @staticmethod
    def from_real(x: RealFloat, ctx: Optional[Context] = None) -> 'Float':
        """
        Converts a `RealFloat` number to a `Float` number.

        Optionally specify a rounding context under which to
        construct this value. If a rounding context is specified,
        `x` must be representable under `ctx`.
        """
        if not isinstance(x, RealFloat):
            raise TypeError(f'expected RealFloat, got {type(x)}')

        f = Float(x=x, ctx=ctx)
        if ctx is None:
            return f
        else:
            if not f.is_representable():
                raise ValueError(f'{x} is not representable under {ctx}')
            return f.normalize()

    @staticmethod
    def from_int(x: int, ctx: Optional[Context] = None) -> 'Float':
        """
        Converts an integer to a `Float` number.

        Optionally specify a rounding context under which to
        construct this value. If a rounding context is specified,
        `x` must be representable under `ctx`.
        """
        if not isinstance(x, int):
            raise TypeError(f'expected int, got {type(x)}')

        return Float.from_real(RealFloat.from_int(x), ctx)

    @staticmethod
    def from_float(x: float, ctx: Optional[Context] = None) -> 'Float':
        """
        Converts a native Python float to a `Float` number.

        Optionally specify a rounding context under which to
        construct this value. If a rounding context is specified,
        `x` must be representable under `ctx`.
        """
        if not isinstance(x, float):
            raise TypeError(f'expected int, got {type(x)}')

        return Float.from_real(RealFloat.from_float(x), ctx)

    @property
    def base(self):
        """Integer base of this number. Always 2."""
        return 2

    @property
    def s(self) -> bool:
        """Is the sign negative?"""
        return self._real.s

    @property
    def exp(self) -> int:
        """Absolute position of the LSB."""
        return self._real.exp

    @property
    def c(self) -> int:
        """Integer significand."""
        return self._real.c

    @property
    def p(self):
        """Minimum number of binary digits required to represent this number."""
        if self.is_nar():
            raise ValueError('cannot compute precision of infinity or NaN')
        return self._real.p

    @property
    def e(self) -> int:
        """
        Normalized exponent of this number.

        When `self.c == 0` (i.e. the number is zero), this method returns
        `self.exp - 1`. In other words, `self.c != 0` iff `self.e >= self.exp`.

        The interval `[self.exp, self.e]` represents the absolute positions
        of digits in the significand.
        """
        if self.is_nar():
            raise ValueError('cannot compute exponent of infinity or NaN')
        return self._real.e

    @property
    def n(self) -> int:
        """
        Position of the first unrepresentable digit below the significant digits.
        This is exactly `self.exp - 1`.
        """
        if self.is_nar():
            raise ValueError('cannot compute exponent of infinity or NaN')
        return self._real.n

    @property
    def m(self) -> int:
        """Significand of this number."""
        if self.is_nar():
            raise ValueError('cannot compute significand of infinity or NaN')
        return self._real.m

    @property
    def interval_size(self) -> int | None:
        """Rounding envelope: size relative to `2**exp`."""
        return self._real.interval_size

    @property
    def interval_down(self) -> bool | None:
        """Rounding envelope: extends below the value."""
        return self._real.interval_down

    @property
    def inexact(self) -> bool:
        """Return whether this number is inexact."""
        return self._real.inexact

    def is_zero(self) -> bool:
        """Returns whether this value represents zero."""
        return not self.is_nar() and self._real.is_zero()

    def is_positive(self) -> bool:
        """Returns whether this value is positive."""
        return not self.is_nar() and self._real.is_positive()

    def is_negative(self) -> bool:
        """Returns whether this value is negative."""
        return not self.is_nar() and self._real.is_negative()

    def is_integer(self) -> bool:
        """Returns whether this value is an integer."""
        return not self.is_nar() and self._real.is_integer()

    def is_finite(self) -> bool:
        """Returns whether this value is finite."""
        return not self.is_nar()

    def is_nonzero(self) -> bool:
        """Returns whether this value is (finite) nonzero."""
        return self.is_finite() and not self.is_zero()

    def is_nar(self) -> bool:
        """Return whether this number is infinity or NaN."""
        return self.isinf or self.isnan

    def is_representable(self) -> bool:
        """
        Checks if this number is representable under
        the rounding context during its construction.
        Usually just a sanity check.
        """
        return self.ctx is None or self.ctx.is_representable(self)

    def is_canonical(self) -> bool:
        """
        Returns if `x` is canonical under this context.

        This function only considers relevant attributes to judge
        if a value is canonical. Thus, there may be more than
        one canonical value for a given number despite the function name.
        The result of `self.normalize()` is always canonical.

        Raises a `ValueError` when `self.ctx is None`.
        """
        if self.ctx is None:
            raise ValueError(f'Float values without a context cannot be normalized: self={self}')
        return self.ctx.is_canonical(self)

    def is_normal(self) -> bool:
        """
        Returns if this number is "normal".

        For IEEE-style contexts, this means that the number is finite, non-zero,
        and `x.normalize()` has full precision.
        """
        if self.ctx is None:
            raise ValueError(f'Float values without a context cannot be normalized: self={self}')
        return self.ctx.is_normal(self)

    def as_real(self) -> RealFloat:
        """Returns the real part of this number."""
        if self.is_nar():
            raise ValueError('cannot convert infinity or NaN to real')
        return RealFloat(x=self._real)

    def normalize(self) -> 'Float':
        """
        Returns the canonical reprsentation of this number.

        Raises a `ValueError` when `self.ctx is None`.
        """
        if self.ctx is None:
            raise ValueError(f'cannot normalize without a context: self={self}')
        return self.ctx.normalize(self)

    def round(self, ctx: Context):
        """
        Rounds this number under the given context.

        This method is equivalent to `ctx.round(self)`.
        """
        if not isinstance(ctx, Context):
            raise TypeError(f'expected Context, got {type(ctx)}')
        return ctx.round(self)

    def round_at(self, ctx: Context, n: int) -> 'Float':
        """
        Rounds this number at the given position.

        This method is equivalent to `self.ctx.round_at(self, n)`.
        """
        if not isinstance(ctx, Context):
            raise TypeError(f'expected Context, got {type(ctx)}')
        return ctx.round_at(self, n)

    def round_integer(self, ctx: Context) -> 'Float':
        """
        Rounds this number to the nearest integer.

        This method is equivalent to `self.ctx.round_integer(self)`.
        """
        if not isinstance(ctx, Context):
            raise TypeError(f'expected Context, got {type(ctx)}')
        return ctx.round_integer(self)

    def compare(self, other: Self | RealFloat) -> Optional[Ordering]:
        """
        Compare `self` and `other` values returning an `Optional[Ordering]`.
        """
        if isinstance(other, RealFloat):
            if self.isnan:
                return None
            elif self.isnan:
                if self.s:
                    return Ordering.LESS
                else:
                    return Ordering.GREATER
            else:
                return self._real.compare(other)
        elif isinstance(other, Float):
            if self.isnan or other.isnan:
                return None
            elif self.isinf:
                if other.isinf and self.s == other.s:
                    return Ordering.EQUAL
                elif self.s:
                    return Ordering.LESS
                else:
                    return Ordering.GREATER
            elif other.isinf:
                if other.s:
                    return Ordering.GREATER
                else:
                    return Ordering.LESS
            else:
                return self._real.compare(other._real)
        else:
            raise TypeError(f'expected Float or RealFloat, got {type(other)}')
