"""
This module defines the `RealFloat` number type,
an arbitrary-precision, floating-point number without infinities and NaN.
"""

import math
import numbers
import random

from fractions import Fraction
from typing import Self

from ..globals import get_current_float_converter, get_current_str_converter
from ..round import RoundingMode, RoundingDirection
from ...utils import (
    bitmask,
    float_to_bits,
    is_dyadic,
    Ordering,
    FP64_M,
    FP64_EXPMIN,
    FP64_SMASK,
    FP64_EMASK,
    FP64_MMASK,
    FP64_EONES,
    FP64_IMPLICIT1,
)


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

    __slots__ = ('_s', '_exp', '_c', '_interval_size', '_interval_down', '_interval_closed')

    _s: bool
    """is the sign negative?"""
    _exp: int
    """absolute position of the LSB"""
    _c: int
    """integer significand"""

    _interval_size: int | None
    """rounding envelope: size relative to `2**exp`"""
    _interval_down: bool
    """rounding envelope: does the interval extend towards zero?"""
    _interval_closed: bool
    """rounding envelope: is the interval closed at the other endpoint?"""

    def __init__(
        self,
        s: bool | None = None,
        exp: int | None = None,
        c: int | None = None,
        *,
        x: Self | None = None,
        e: int | None = None,
        m: int | None = None,
        interval_size: int | None = None,
        interval_down: bool | None = None,
        interval_closed: bool | None = None,
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
            self._c = c
            if s is not None:
                self._s = s
            elif x is not None:
                self._s = x._s
            else:
                self._s = False
        elif m is not None:
            if s is not None:
                raise ValueError(f'cannot specify both m={m} and s={s}')
            self._c = abs(m)
            self._s = m < 0
        elif x is not None:
            self._c = x._c
            if s is not None:
                self._s = s
            else:
                self._s = x._s
        else:
            self._c = 0
            if s is not None:
                self._s = s
            else:
                self._s = False

        # exp
        if exp is not None:
            if e is not None:
                raise ValueError(f'cannot specify both exp={exp} and e={e}')
            self._exp = exp
        elif e is not None:
            self._exp = e - self._c.bit_length() + 1
        elif x is not None:
            self._exp = x._exp
        else:
            self._exp = 0

        # rounding envelope size
        if interval_size is not None:
            if interval_size > 0:
                raise ValueError(f'cannot specify interval_size={interval_size}, must be <= 0')
            self._interval_size = interval_size
        elif x is not None:
            self._interval_size = x._interval_size
        else:
            self._interval_size = None

        # rounding envelope direction
        if interval_down is not None:
            self._interval_down = interval_down
        elif x is not None:
            self._interval_down = x._interval_down
        else:
            self._interval_down = False

        # rounding envelope endpoint
        if interval_closed is not None:
            self._interval_closed = interval_closed
        elif x is not None:
            self._interval_closed = x._interval_closed
        else:
            self._interval_closed = False

    def __repr__(self):
        return (f'{self.__class__.__name__}('
            + 's=' + repr(self._s)
            + ', exp=' + repr(self._exp)
            + ', c=' + repr(self._c)
            + ', interval_size=' + repr(self._interval_size)
            + ', interval_down=' + repr(self._interval_down)
            + ', interval_closed=' + repr(self._interval_closed)
            + ')'
        )

    def __str__(self):
        fn = get_current_str_converter()
        return fn(self)

    def __hash__(self): # type: ignore
        # Complex has __hash__ = None, so mypy thinks there's a type mismatch.
        return hash((self._s, self._exp, self._c))

    def __eq__(self, other):
        if not isinstance(other, RealFloat | int | float | Fraction):
            return False
        ord = self.compare(other)
        return ord == Ordering.EQUAL

    def __lt__(self, other):
        ord = self.compare(other)
        return ord == Ordering.LESS

    def __le__(self, other):
        ord = self.compare(other)
        return ord == Ordering.LESS or ord == Ordering.EQUAL

    def __gt__(self, other):
        ord = self.compare(other)
        return ord == Ordering.GREATER

    def __ge__(self, other):
        ord = self.compare(other)
        return ord == Ordering.GREATER or ord == Ordering.EQUAL

    def __neg__(self):
        """
        Unary minus.

        Returns this `RealFloat` with opposite sign (`self.s`)
        even when `self.is_zero()`.
        """
        return RealFloat(s=not self._s, x=self)

    def __pos__(self):
        """
        Unary plus. 

        Returns a copy of `self`.
        """
        return RealFloat(x=self)

    def __add__(self, other):
        """
        Adds `self` and `other` exactly.

        This operation never fails when `other` is a `RealFloat`.
        """
        match other:
            case RealFloat():
                pass
            case int():
                other = RealFloat.from_int(other)
            case float():
                other = RealFloat.from_float(other)
            case Fraction():
                other = RealFloat.from_rational(other)
            case _:
                raise TypeError(f'unsupported operand type(s) for +: \'RealFloat\' and \'{type(other)}\'')

        if self._c == 0:
            # 0 + b = b
            return RealFloat(x=other)
        elif other._c == 0:
            # a + 0 = a
            return RealFloat(x=self)
        else:
            # adding non-zero values

            # compute the smallest exponent and normalize
            exp = min(self._exp, other._exp)

            # normalize significands relative to `exp`
            c1 = self._c << (self._exp - exp)
            c2 = other._c << (other._exp - exp)

            # apply signs
            m1 = -c1 if self._s else c1
            m2 = -c2 if other._s else c2

            # add/subtract
            m = m1 + m2

            # decompose into `s` and `c`
            s = m < 0
            c = -m if s else m

            # return the result
            return RealFloat(s=s, exp=exp, c=c)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        """
        Multiplies `self` and `other` exactly.

        This operation never fails when `other` is a `RealFloat`.
        """
        match other:
            case RealFloat():
                pass
            case int():
                other = RealFloat.from_int(other)
            case float():
                other = RealFloat.from_float(other)
            case Fraction():
                other = RealFloat.from_rational(other)
            case _:
                raise TypeError(f'unsupported operand type(s) for +: \'RealFloat\' and \'{type(other)}\'')

        s = self._s != other._s
        if self._c == 0 or other._c == 0:
            # 0 * b = 0 or a * 0 = 0
            # respects signedness
            return RealFloat(s=s)
        else:
            # multiplying non-zero values
            exp = self._exp + other._exp
            c = self._c * other._c
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
            s = self._s and (exponent % 2 == 1)
            exp = self._exp * exponent
            c = self._c ** exponent
            return RealFloat(s=s, exp=exp, c=c)

    def __rpow__(self, base):
        raise TypeError(f'unsupported operand type(s) for **: \'{type(base)}\' and \'RealFloat\'')

    def __abs__(self):
        """
        Absolute value.

        Returns this `RealFloat` with `self.s = False`.
        """
        return RealFloat(s=False, x=self)

    def __trunc__(self) -> int:
        return int(self.round(min_n=-1, rm=RoundingMode.RTZ))

    def __floor__(self) -> int:
        return int(self.round(min_n=-1, rm=RoundingMode.RTN))

    def __ceil__(self) -> int:
        return int(self.round(min_n=-1, rm=RoundingMode.RTP))

    def __round__(self, ndigits=None) -> int:
        return int(self.round(min_n=-1, rm=RoundingMode.RNE))

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
        if self._c == 0:
            return 0

        if self._exp >= 0:
            # `self.c` consists of integer digits
            c = self._c << self._exp
        else:
            # `self.c` consists of fractional digits
            # but safe to just shift them off
            c = self._c >> -self._exp

        return (-1 if self._s else 1) * c

    @property
    def s(self) -> bool:
        """property: is the sign negative?"""
        return self._s

    @property
    def exp(self) -> int:
        """property: absolute position of the LSB"""
        return self._exp

    @property
    def c(self) -> int:
        """property: integer significand"""
        return self._c

    @property
    def interval_size(self) -> int | None:
        """property: rounding envelope size relative to `2**exp`"""
        return self._interval_size

    @property
    def interval_down(self) -> bool:
        """property: rounding envelope extends towards zero"""
        return self._interval_down

    @property
    def interval_closed(self) -> bool:
        """property: rounding envelope is closed at the other endpoint"""
        return self._interval_closed

    def as_rational(self) -> Fraction:
        if self.is_zero():
            return Fraction(0)
        else:
            return self.m * (Fraction(2) ** self._exp)

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
        sbits = b & FP64_SMASK
        ebits = (b & FP64_EMASK) >> FP64_M
        mbits = b & FP64_MMASK

        # sign
        s = sbits != 0

        # case split on exponent
        if ebits == 0:
            # zero / subnormal
            return RealFloat(s=s, exp=FP64_EXPMIN, c=mbits)
        elif ebits == FP64_EONES:
            # infinity or NaN
            raise ValueError(f'expected finite float, got x={x}')
        else:
            # normal
            exp = FP64_EXPMIN + (ebits - 1)
            c = FP64_IMPLICIT1 | mbits
            return RealFloat(s=s, exp=exp, c=c)

    @staticmethod
    def from_rational(x: numbers.Rational):
        """
        Creates a new `RealFloat` value from a `Fraction`.

        Raise a `ValueError` if `x` is not a dyadic rational.
        """
        if not isinstance(x, numbers.Rational):
            raise TypeError(f'expected Rational, got {type(x)}')
        if not is_dyadic(x):
            raise ValueError(f'expected a dyadic rational, got `{x}`')
        if x == 0:
            # case: 0
            return RealFloat.zero()
        else:
            n = int(x.numerator)
            d = int(x.denominator)
            if d == 1:
                # case: integer
                return RealFloat.from_int(n)
            else:
                # case: has fractional bits
                exp = d.bit_length() - 1
                m = n * (2 ** exp) // d
                return RealFloat(m=m, exp=-exp)

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
        return self._c.bit_length()

    @property
    def e(self) -> int:
        """
        Normalized exponent of this number.

        When `self.c == 0` (i.e. the number is zero), this method returns
        `self.exp - 1`. In other words, `self.c != 0` iff `self.e >= self.exp`.

        The interval `[self.exp, self.e]` represents the absolute positions
        of digits in the significand.
        """
        return self._exp + self.p - 1

    @property
    def n(self) -> int:
        """
        Position of the first unrepresentable digit below the significant digits.
        This is exactly `self.exp - 1`.
        """
        return self._exp - 1

    @property
    def m(self) -> int:
        """
        Signed significand.
        This is exactly `(-1)^self.s * self.c`.
        """
        return -self._c if self._s else self._c

    @property
    def inexact(self) -> bool:
        """Is this value inexact?"""
        return self._interval_size is not None

    @property
    def numerator(self):
        return self.as_rational().numerator

    @property
    def denominator(self):
        return self.as_rational().denominator

    def is_zero(self) -> bool:
        """Returns whether this value represents zero."""
        return self._c == 0

    def is_nonzero(self) -> bool:
        """Returns whether this value does not represent zero."""
        return self._c != 0

    def is_positive(self) -> bool:
        """Returns whether this value is positive."""
        return self._c != 0 and not self._s

    def is_negative(self) -> bool:
        """Returns whether this value is negative."""
        return self._c != 0 and self._s

    def is_more_significant(self, n: int) -> bool:
        """
        Returns `True` iff this value only has significant digits above `n`,
        that is, every non-zero digit in the number is more significant than `n`.

        When `n = -1`, this method is equivalent to `is_integer()`.

        This method is equivalent to::

            _, lo = self.split(n)
            return lo.is_zero()
        """
        if not isinstance(n, int):
            raise TypeError(f'expected \'int\' for n, got {n}')

        if self.is_zero():
            return True

        # All significant digits are above n
        if self._exp > n:
            return True

        # All significant digits are at or below n
        if self.e <= n:
            return False

        # Some digits may be at or below n; check if those are zero
        n_relative = n - self._exp
        return (self._c & bitmask(n_relative + 1)) == 0

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

        # compute digit offset from `self.exp`
        offset = n - self._exp

        # outside the region of significance
        if offset < 0 or offset >= self.p:
            return False

        # test the `offset`-th bit of `self.c`
        return (self._c & (1 << offset)) != 0


    def normalize(self, p: int | None = None, n: int | None = None):
        """
        Returns a value numerically equivalent to `self` based on
        precision `p` and position `n`:

        - `None, None`: a copy of `self`, i.e., `self.exp == self.normalize().exp`, etc.
        - `p, None`: a copy of `self` that has exactly `p` bits of precision.
        - `None, n`: a copy of `self` where `self.exp == n + 1`.
        - `p, n`: a copy of `self` such that `self.exp >= n + 1` and
            has maximal precision up to `p` bits.

        Raises a `ValueError` if no such value exists, i.e.,
        if the value cannot be represented with the given `p` and `n`.
        """

        match p, n:
            case None, None:
                # return a copy of self
                return RealFloat(self._s, self._exp, self._c)
            case int(), None:
                # normalize to precision p
                if p < 0:
                    raise ValueError(f'precision must be non-negative: p={p}')
                # compute maximum shift and resulting exponent
                shift = p - self.p
                exp = self._exp - shift
            case None, int():
                # normalize to absolute position `n`
                exp = n + 1
                shift = self._exp - exp
            case int(), int():
                # normalize to precision p and position n
                # prefer absolute precision constraint over position constraint
                if p < 0:
                    raise ValueError(f'precision must be non-negative: p={p}')
                # compute maximum shift and resulting exponent
                shift = p - self.p
                exp = self._exp - shift
                # check if exponent is too small, adjust accordingly
                if exp <= n:
                    expmin = n + 1
                    adjust = expmin - exp
                    shift -= adjust
                    exp += adjust
            case _:
                if p is not None and not isinstance(p, int):
                    raise TypeError(f'expected \'int\' or \'None\' for p, got {type(p)}')
                if n is not None and not isinstance(n, int):
                    raise TypeError(f'expected \'int\' or \'None\' for n, got {type(n)}')
                raise RuntimeError('unreachable')

        # compute new significand `c`
        if shift == 0:
            # no shifting
            c = self._c
        elif shift > 0:
            # shifting left by a non-negative amount
            c = self._c << shift
        else:
            # shift right by a positive amount
            shift = -shift
            c = self._c >> shift
            # check that we didn't lose significant digits
            if (self._c & bitmask(shift)) != 0:
                raise ValueError(f'shifting off digits: p={p}, n={n}, x={self}')

        return RealFloat(self._s, exp, c)


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
            hi = RealFloat(self._s, n + 1, 0)
            lo = RealFloat(self._s, n, 0)
            return (hi, lo)
        elif n >= self.e:
            # check if all digits are in the lower part
            hi = RealFloat(self._s, n + 1, 0)
            lo = RealFloat(self._s, self._exp, self._c)
            return (hi, lo)
        elif n < self._exp:
            # check if all digits are in the upper part
            hi = RealFloat(self._s, self._exp, self._c)
            lo = RealFloat(self._s, n, 0)
            return (hi, lo)
        else:
            # splitting the digits
            p_lo = (n + 1) - self._exp
            mask_lo = bitmask(p_lo)

            exp_hi = self._exp + p_lo
            c_hi = self._c >> p_lo

            exp_lo = self._exp
            c_lo = self._c & mask_lo

            hi = RealFloat(self._s, exp_hi, c_hi)
            lo = RealFloat(self._s, exp_lo, c_lo)
            return (hi, lo)

    def compare(self, other: Self | int | float | Fraction) -> Ordering | None:
        """
        Compare `self` and `other` values returning an `Optional[Ordering]`.

        For two `RealFloat` values, the result is never `None`.
        """
        match other:
            case RealFloat():
                if self._c == 0:
                    if other._c == 0:
                        return Ordering.EQUAL
                    elif other._s:
                        return Ordering.GREATER
                    else:
                        return Ordering.LESS
                elif other._c == 0:
                    if self._s:
                        return Ordering.LESS
                    else:
                        return Ordering.GREATER
                elif self._s != other._s:
                    # non-zero signs are different
                    if self._s:
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
                            exp = min(self._exp, other._exp)
                            c1 = self._c << (self._exp - exp)
                            c2 = other._c << (other._exp - exp)
                            cmp = Ordering.from_compare(c1, c2)

                    # adjust for the sign
                    if self._s:
                        return cmp.reverse()
                    else:
                        return cmp
            case int():
                return self.compare(RealFloat.from_int(other))
            case float():
                if math.isnan(other):
                    return None
                elif math.isinf(other):
                    if other > 0:
                        return Ordering.LESS
                    else:
                        return Ordering.GREATER
                else:
                    return self.compare(RealFloat.from_float(other))
            case Fraction():
                f = self.as_rational()
                if f < other:
                    return Ordering.LESS
                elif f > other:
                    return Ordering.GREATER
                else:
                    return Ordering.EQUAL
            case _:
                raise TypeError(f'comparison not supported between \'RealFloat\' and \'{type(other)}\'')

    def is_identical_to(self, other: Self) -> bool:
        """Is the value encoded identically to another `RealFloat` value?"""
        if not isinstance(other, RealFloat):
            return TypeError(f'expected RealFloat, got {type(other)}')

        return (
            self._s == other._s
            and self._exp == other._exp
            and self._c == other._c
            and self._interval_size == other._interval_size
            and self._interval_down == other._interval_down
            and self._interval_closed == other._interval_closed
        )


    def next_away(self):
        """
        Computes the next number (with the same precision),
        away from zero.
        """
        c = self._c + 1
        exp = self._exp
        if c.bit_length() > self.p:
            # adjust the exponent since we exceeded precision bounds
            # the value is guaranteed to be a power of two
            c >>= 1
            exp  += 1

        return RealFloat(s=self._s, c=c, exp=exp)

    def next_towards(self):
        """
        Computes the previous number (with the same precision),
        towards zero.
        """
        c = self._c - 1
        exp = self._exp
        if c.bit_length() < self.p:
            # previously at a power of two
            # need to add a lower bit
            c = (c << 1) | 1
            exp -= 1

        return RealFloat(s=self._s, c=c, exp=exp)

    def next_up(self):
        """
        Computes the next number (with the same precison),
        towards positive infinity.
        """
        if self._s:
            return self.next_towards()
        else:   
            return self.next_away()

    def next_down(self):
        """
        Computes the previous number (with the same precision),
        towards negative infinity.
        """
        if self._s:
            return self.next_away()
        else:
            return self.next_towards()


    def _round_params(self, max_p: int | None = None, min_n: int | None = None):
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
        nearest, direction = rm.to_direction(kept._s)

        # rounding envelope
        interval_size: int | None = None
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
                            is_even = (kept._c & 1) == 0
                            increment = not is_even
                        case RoundingDirection.RTO:
                            is_even = (kept._c & 1) == 0
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
                        is_even = (kept._c & 1) == 0
                        increment = not is_even
                    case RoundingDirection.RTO:
                        is_even = (kept._c & 1) == 0
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
        p: int | None,
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
            kept._c += 1
            if p is not None and kept._c.bit_length() > p:
                # adjust the exponent since we exceeded precision bounds
                # the value is guaranteed to be a power of two
                kept._c >>= 1
                kept._exp += 1

                assert interval_size is not None, 'interval_size is None when rounding is exact'
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

    def _round_at(
        self,
        p: int | None,
        n: int,
        rm: RoundingMode,
        exact: bool
    ):
        """
        Rounds `self` at absolute digit position `n` using the rounding mode
        specified by `rm`. Optionally, specify `p` to limit the precision
        of the result to at most `p` bits. If `exact` is `True`, the result
        must be exact.
        """

        # step 1. split the number at the rounding position
        kept, lost = self.split(n)

        # step 2. check if rounding was exact (if so, we're done)
        if lost.is_zero():
            return kept

        if exact:
            raise ValueError(f'rounding off digits: self={self}, n={n}')

        # step 3. recover the rounding bits
        if lost.e == n:
            # the MSB of lo is at position n
            half_bit = (lost._c >> (lost.p - 1)) != 0
            lower_bits = (lost._c & bitmask(lost.p - 1)) != 0
        else:
            # the MSB of lo is below position n
            half_bit = False
            lower_bits = True

        return self._round_finalize(kept, half_bit, lower_bits, p, rm)

    def _round_at_stochastic(
        self,
        p: int | None,
        n: int,
        rm: RoundingMode,
        num_randbits: int | None,
        randbits: int | None,
        exact: bool
    ):
        """
        Rounds `self` stochastically at absolute digit position `n` using
        `num_randbits` rounding digits. The rounding mode `rm` decides how to
        round the extended precision value (with rounding digits).
        Optionally, specify `p` to limit the precision of the result to
        at most `p` bits. If `exact` is `True`, the result must be exact.
        """
    
        # step 1. compute the actual number of rounding bits to use
        if num_randbits is None:
            # use all the bits (in theory, `num_randbits == float('inf')`)
            # but the actual number of bits is limited by the precision of `self`
            num_randbits = self.p

        # step 2. compute rounding parameters for extended-precision value
        n_rand = n - num_randbits
        if p is None:
            p_rand = None
        else:
            p_rand = p - num_randbits

        if randbits is None:
            randbits = random.getrandbits(num_randbits)

        # step 1. round the number to obtain the extended-precision value
        xr = self._round_at(p_rand, n_rand, rm, exact)

        # step 2. split the number at the rounding position to get the rounding bits
        _, lost = xr.split(n)

        # step 3. normalize `lost` so that `lost.n == n_rand`
        offset = lost._exp - (n_rand + 1)
        if offset > 0:
            lost_c = lost._c << offset
        elif offset < 0:
            lost_c = lost._c >> -offset
        else:
            lost_c = lost._c

        # step 4. round down if the random bits are larger than the rounding bits
        if randbits >= lost_c:
            # round down
            return self._round_at(p, n, RoundingMode.RTZ, exact)
        else:
            # round up
            return self._round_at(p, n, RoundingMode.RAZ, exact)


    def round_at(
        self,
        n: int,
        p: int | None = None,
        rm: RoundingMode = RoundingMode.RNE,
        num_randbits: int | None = 0,
        *,
        randbits: int | None = None,
        exact: bool = False):
        """
        Creates a copy of `self` rounded at absolute digit position `n`
        using the rounding mode specified by `rm`. If `exact` is `True`,
        the result must be exact.
        """
        if not isinstance(n, int):
            raise TypeError(f'Expected \'int\' for n={n}, got {type(n)}')
        if not isinstance(rm, RoundingMode):
            raise TypeError(f'Expected \'RoundingMode\' for rm={rm}, got {type(rm)}')
        if p is not None and not isinstance(p, int):
            raise TypeError(f'Expected \'int\' for p={p}, got {type(p)}')
        if num_randbits is not None and not isinstance(num_randbits, int):
            raise TypeError(f'Expected \'int\' for num_randbits={num_randbits}, got {type(num_randbits)}')
        if randbits is not None and not isinstance(randbits, int):
            raise TypeError(f'Expected \'int\' for randbits={randbits}, got {type(randbits)}')

        if (num_randbits is not None
            and randbits is not None
            and (randbits < 0 or randbits >= (1 << num_randbits))):
            raise ValueError(f'randbits must be in [0, {1 << num_randbits}), got {randbits}')

        if num_randbits == 0:
            # non-stochastic rounding
            return self._round_at(p, n, rm, exact)
        else:
            # stochastic rounding
            return self._round_at_stochastic(p, n, rm, num_randbits, randbits, exact)

    def round(self,
        max_p: int | None = None,
        min_n: int | None = None,
        rm: RoundingMode = RoundingMode.RNE,
        num_randbits: int | None = 0,
        *,
        randbits: int | None = None,
        exact: bool = False,
    ):
        """
        Creates a copy of `self` rounded to at most `max_p` digits of precision
        or a least absolute digit position `min_n`, whichever bound
        is encountered first.

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

        When `num_randbits=0`, the rounding is performed by rounding
        using the rounding mode specified by `rm`.

        When `randbits` is specified, the rounding is performed stochastically
        using `randbits` rounding bits to decide which way to round. If `randbits=None`,
        then all additional bits are considered rounding bits. The rounding mode specified
        by `rm` decides how the additional rounding bits are themselves rounded.
        If `randbits=None`, rounding is decided by Python's native `random` module,
        otherwise the value is used as the "randomly" sampled bits.
        """

        if max_p is not None and not isinstance(max_p, int):
            raise TypeError(f'Expected \'int\' for max_p={max_p}, got {type(max_p)}')
        if min_n is not None and not isinstance(min_n, int):
            raise TypeError(f'Expected \'int\' for min_n={min_n}, got {type(min_n)}')
        if not isinstance(rm, RoundingMode):
            raise TypeError(f'Expected \'RoundingMode\' for rm={rm}, got {type(rm)}')
        if num_randbits is not None and not isinstance(num_randbits, int):
            raise TypeError(f'Expected \'int\' for num_randbits={num_randbits}, got {type(num_randbits)}')
        if randbits is not None and not isinstance(randbits, int):
            raise TypeError(f'Expected \'int\' for randbits={randbits}, got {type(randbits)}')

        if max_p is None and min_n is None:
            raise ValueError(f'must specify {max_p} or {min_n}')

        if (num_randbits is not None
            and randbits is not None
            and (randbits < 0 or randbits >= (1 << num_randbits))):
            raise ValueError(f'randbits must be in [0, {1 << num_randbits}), got {randbits}')

        # step 1. compute rounding parameters
        p, n = self._round_params(max_p, min_n)

        # step 2. round at the specified position
        if num_randbits == 0:
            # non-stochastic rounding
            return self._round_at(p, n, rm, exact)
        else:
            # stochastic rounding
            return self._round_at_stochastic(p, n, rm, num_randbits, randbits, exact)
