"""
This module defines the `RealFloat` number type,
an arbitrary-precision, floating-point number without infinities and NaN.
"""

import math
import numbers
import numpy as np
import random

from fractions import Fraction
from typing import Self, TypeAlias, overload

from ..globals import get_current_float_converter, get_current_str_converter
from ..round import RoundingMode, RoundingDirection
from ...utils import (
    bitmask,
    float_to_bits,
    is_dyadic,
    is_power_of_two,
    Ordering,
    FP64_M,
    FP64_EXPMIN,
    FP64_SMASK,
    FP64_EMASK,
    FP64_MMASK,
    FP64_EONES,
    FP64_IMPLICIT1,
)

from .flags import Flags


RNG: TypeAlias = random.Random | np.random.Generator
"""Type alias for random number generators."""


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

    This type also contains status flags (`Flags`) to indicate
    exceptional events that occured during rounding.
    """

    __slots__ = ('_s', '_exp', '_c', '_flags')

    _s: bool
    """is the sign negative?"""
    _exp: int
    """absolute position of the LSB"""
    _c: int
    """integer significand"""
    _flags: Flags
    """status flags for exceptional events during rounding"""

    def __init__(
        self,
        s: bool | None = None,
        exp: int | None = None,
        c: int | None = None,
        *,
        x: Self | None = None,
        e: int | None = None,
        m: int | None = None,
        overflow: bool | None = None,
        tiny_pre: bool | None = None,
        tiny_post: bool | None = None,
        inexact: bool | None = None,
        carry: bool | None = None
    ):
        """
        Creates a new `RealFloat` value.

        The sign may be optionally specified with `s`.
        The exponent may be specified with `exp` or `e`.
        The significand may be specified with `c` or `m` (unless `x` is given).
        If `x` is given, any field not specified is copied from `x`.
        """
        # if x is not None and not isinstance(x, RealFloat):
        #     raise TypeError(f'expected RealFloat, got {type(x)}')

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

        # flags
        if x is None:
            self._flags = Flags(
                overflow=overflow,
                tiny_pre=tiny_pre,
                tiny_post=tiny_post,
                inexact=inexact,
                carry=carry
            )
        else:
            self._flags = Flags(
                x=x._flags,
                overflow=overflow,
                tiny_pre=tiny_pre,
                tiny_post=tiny_post,
                inexact=inexact,
                carry=carry
            )


    def __repr__(self):
        return (f'{self.__class__.__name__}('
            + 's=' + repr(self._s)
            + ', exp=' + repr(self._exp)
            + ', c=' + repr(self._c)
            + ', flags=' + repr(self._flags)
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

    @overload
    def __add__(self, other: 'RealFloat') -> 'RealFloat': ...
    @overload
    def __add__(self, other: int) -> 'RealFloat': ...
    @overload
    def __add__(self, other: float) -> Self | float: ...
    @overload
    def __add__(self, other: Fraction) -> 'RealFloat': ...

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
                if math.isnan(other) or math.isinf(other):
                    # Convert self to float and perform float arithmetic
                    return float(self) + other
                else:
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

    @overload
    def __radd__(self, other: 'RealFloat') -> 'RealFloat': ...
    @overload
    def __radd__(self, other: int) -> 'RealFloat': ...
    @overload
    def __radd__(self, other: float) -> Self | float: ...
    @overload
    def __radd__(self, other: Fraction) -> 'RealFloat': ...

    def __radd__(self, other):
        return self + other

    @overload
    def __sub__(self, other: 'RealFloat') -> 'RealFloat': ...
    @overload
    def __sub__(self, other: int) -> 'RealFloat': ...
    @overload
    def __sub__(self, other: float) -> Self | float: ...
    @overload
    def __sub__(self, other: Fraction) -> 'RealFloat': ...

    def __sub__(self, other):
        return self + (-other)
    
    @overload
    def __rsub__(self, other: 'RealFloat') -> 'RealFloat': ...
    @overload
    def __rsub__(self, other: int) -> 'RealFloat': ...
    @overload
    def __rsub__(self, other: float) -> Self | float: ...
    @overload
    def __rsub__(self, other: Fraction) -> 'RealFloat': ...

    def __rsub__(self, other):
        return (-self) + other

    @overload
    def __mul__(self, other: 'RealFloat') -> 'RealFloat': ...
    @overload
    def __mul__(self, other: int) -> 'RealFloat': ...
    @overload
    def __mul__(self, other: float) -> Self | float: ...
    @overload
    def __mul__(self, other: Fraction) -> 'RealFloat': ...

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
                if math.isnan(other) or math.isinf(other):
                    # Convert self to float and perform float arithmetic
                    other_sgn = math.copysign(1.0, other) # extract the sign bit
                    s = self._s != (other_sgn < 0)
                    res_sgn = -1.0 if s else 1.0
                    return other * res_sgn
                else:
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

    @overload
    def __rmul__(self, other: 'RealFloat') -> 'RealFloat': ...
    @overload
    def __rmul__(self, other: int) -> 'RealFloat': ...
    @overload
    def __rmul__(self, other: float) -> Self | float: ...
    @overload
    def __rmul__(self, other: Fraction) -> 'RealFloat': ...

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
    def overflow(self) -> bool:
        """Overflow flag: the result exceeded the representable range."""
        return self._flags.overflow

    @property
    def tiny_pre(self) -> bool:
        """Tiny before rounding flag: the result before rounding satisfies `|x| < 2^emin`."""
        return self._flags.tiny_pre

    @property
    def tiny_post(self) -> bool:
        """
        Tiny after rounding flag: the result after rounding
        (without subnormalization) satisfies `|x| < 2^emin`.
        """
        return self._flags.tiny_post

    @property
    def inexact(self) -> bool:
        """Inexact flag: the rounded result is not the same as the exact result."""
        return self._flags.inexact

    @property
    def carry(self) -> bool:
        """Carry flag: the rounded result has a different exponent than the exact result."""
        return self._flags.carry

    @property
    def underflow_pre(self) -> bool:
        """Underflow before rounding flag: `self.tiny_pre and self.inexact`."""
        return self._flags.underflow_pre

    @property
    def underflow_post(self) -> bool:
        """Underflow after rounding flag: `self.tiny_post and self.inexact`."""
        return self._flags.underflow_post

    def as_rational(self) -> Fraction:
        if self._c == 0: # case: zero
            return Fraction(0)
        elif self._exp >= 0: # case: definitely integer
            return Fraction(self.m * (2 ** self._exp))
        else: # case: likely fractional
            return Fraction(self.m, 2 ** (-self._exp))

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

    def is_power_of_two(self) -> bool:
        """Returns whether this value is a power of two, i.e., `(-1)^s * 2^e`."""
        return self._c != 0 and is_power_of_two(self._c)

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
                raise ValueError(f'invalid parameters: p={p}, n={n}')

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

    def is_identical_to(self, other: 'RealFloat') -> bool:
        """Is the value encoded identically to another `RealFloat` value?"""
        if not isinstance(other, RealFloat):
            return TypeError(f'expected RealFloat, got {type(other)}')
        return (
            self._s == other._s
            and self._exp == other._exp
            and self._c == other._c
        )

    def _extract_and_normalize(self, n: int, p: int | None):
        # extract the relevant parameters
        if self._exp != n + 1 or (p is not None and self.p > p):
            # need to normalize first
            x = self.normalize(p, n)
            c = x._c
            exp = x._exp
        else:
            c = self._c
            exp = self._exp

        return c, exp

    def _next_away(self, n: int, p: int | None = None):
        """
        Computes the next number with exponent above `n` and precision
        up to `p` bits, away from zero.
        """
        # extract the relevant parameters for the next value
        c, exp = self._extract_and_normalize(n, p)

        # increment the significand
        c += 1

        # adjust the exponent if we exceeded precision bounds
        if p is not None and c.bit_length() > p:
            # the value is guaranteed to be a power of two
            c >>= 1
            exp += 1

        return RealFloat(s=self._s, c=c, exp=exp)

    def _next_towards(self, n: int, p: int | None = None):
        """
        Computes the next number with exponent above `n` and precision
        up to `p` bits, towards zero.
        """
        # extract the relevant parameters for the next value
        c, exp = self._extract_and_normalize(n, p)

        # decrement the significand
        c -= 1

        # adjust the exponent if we lost a significant bit
        if p is not None and exp > n + 1 and c.bit_length() < p:
            # previously at a power of two, need to add a lower bit
            c = (c << 1) | 1
            exp -= 1

        return RealFloat(s=self._s, c=c, exp=exp)

    def next_away_zero(self, p: int | None = None, n: int | None = None):
        """
        Computes the next number away from zero.

        If `p` or `n` is specified, then `self` is normalized
        accordingly before computing the next value.
        Otherwise, the step size if `2 ** self.exp` even when `self.c == 0`.

        If `self == 0`, then a ValueError is raised.
        """
        if n is None:
            n = self.n
        return self._next_away(n, p)

    def next_towards_zero(self, p: int | None = None, n: int | None = None):
        """
        Computes the next number towards zero.

        If `p` or `n` is specified, then `self` is normalized
        accordingly before computing the next value.
        Otherwise, the step size if `2 ** self.exp` even when `self.c == 0`.

        If `self == 0`, then a ValueError is raised.
        """
        if self._c == 0:
            raise ValueError('zero does not have a next value')
        if n is None:
            n = self.n
        return self._next_towards(n, p)

    def next_up(self, p: int | None = None, n: int | None = None):
        """
        Computes the next number towards positive infinity.

        If `p` or `n` is specified, then `self` is normalized
        accordingly before computing the next value.
        Otherwise, the step size if `2 ** self.exp` even when `self.c == 0`.
        """
        if self._c == 0:
            # step away from zero towards positive infinity
            x = RealFloat(exp=self._exp) if self._s else self
            return x.next_away_zero(p, n)
        elif self._s:
            # x < 0, so need to step towards zero
            return self.next_towards_zero(p, n)
        else:
            # x >= 0, so need to step away from zero
            return self.next_away_zero(p, n)

    def next_down(self, p: int | None = None, n: int | None = None):
        """
        Computes the next number towards negative infinity.

        If `p` or `n` is specified, then `self` is normalized
        accordingly before computing the next value.
        Otherwise, the step size if `2 ** self.exp` even when `self.c == 0`.
        """
        if self._c == 0:
            # step away from zero towards positive infinity
            x = self if self._s else RealFloat(s=True, exp=self._exp)
            return x.next_away_zero(p, n)
        elif self._s:
            # x < 0, so need to step away from zero
            return self.next_away_zero(p, n)
        else:
            # x >= 0, so need to step towards zero
            return self.next_towards_zero(p, n)

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

    def _round_increment_direction(self, direction: RoundingDirection):
        """
        When in the halfway case for nearest rounding modes,
        or not representable in non-nearest rounding modes,
        determines whether to round up based on the rounding direction
        `direction` and the parity of the least significant bit of
        the significand.
        """
        match direction:
            case RoundingDirection.RTZ:
                return False
            case RoundingDirection.RAZ:
                return True
            case RoundingDirection.RTE:
                return (self._c & 1) != 0
            case RoundingDirection.RTO:
                return (self._c & 1) == 0
            case _:
                raise ValueError(f'invalid rounding direction: {direction}')

    def _round_increment(
        self,
        lost: 'RealFloat',
        n: int,
        rm: RoundingMode,
    ):
        """
        Determines whether we need to increment this value based on
        the rounding mode `rm` and the lost digits `lost` below
        at or below the rounding position `n`.

        Must be the case that `lost` is non-zero.
        """
        # convert the rounding mode to a direction
        nearest, direction = rm.to_direction(self._s)

        # case split on nearest mode
        if nearest:
            # nearest rounding mode
            # extract halfway bit and lower bits
            if lost.e == n:
                # the MSB of lo is at position n
                half_bit = (lost._c >> (lost.p - 1)) != 0
                lower_bits = (lost._c & bitmask(lost.p - 1)) != 0
            else:
                # the MSB of lo is below position n
                half_bit = False
                lower_bits = True

            # case split on halfway bit
            if half_bit:
                # at least halfway
                if lower_bits:
                    # above halfway
                    increment = True
                else:
                    # exact halfway
                    increment = self._round_increment_direction(direction)
            else:
                # below halfway
                increment = False
        else:
            # non-nearest rounding mode
            increment = self._round_increment_direction(direction)

        return increment

    def _tiny_pre(self, emin: int) -> bool:
        """
        Computes tininess before rounding.

        This is a property of floating-point rounding.
        A value `x` is tiny before rounding if `|x| < 2^emin`.
        """
        return self.is_zero() or self.e < emin

    def _tiny_post(self, kept: Self, emin: int, n: int, rm: RoundingMode):
        """
        Computes tininess after rounding.

        This is a property of floating-point rounding.
        A value `x` is tiny after rounding if the result of rounding
        (without subnormalization) satisfies `|x| < 2^emin`.

        Assumes that
        - `tiny_pre` is `True`: `|self| < 2^emin`
        - `kept` was the non-zero result after incrementing
        """
        # fast check for tininess
        if kept.e < emin - 1:
            # definitely tiny: |kept| < 2^(emin - 1)
            return True

        # harder check for tininess
        # are we at or below the "next" representable value below 2^emin
        p = emin - n
        cutoff = RealFloat(s=self._s, c=bitmask(p), exp=n)
        if (self >= cutoff if self._s else self <= cutoff):
            # definitely tiny: |kept| < 2^(emin - 1) * (2 - 2^{emin - p + 1})
            return True

        # this is the only interesting case: |kept| = 2^emin
        # re-round by splitting one digit lower, so we do subnormalize
        # we are tiny post rounding if we do not increment
        kept, lost = self.split(n - 1)
        increment = kept._round_increment(lost, n - 1, rm)
        return not increment


    def _round_at(
        self,
        p: int | None,
        n: int,
        emin: int | None,
        rm: RoundingMode,
        exact: bool
    ):
        """
        Rounds `self` at absolute digit position `n` using the rounding mode
        specified by `rm`. Optionally, specify `p` to limit the precision
        of the result to at most `p` bits. If `exact` is `True`, the result
        must be exact.
        """

        # compute tininess before rounding
        tiny_pre = emin is not None and self._tiny_pre(emin)

        # other flags set to their default values
        tiny_post = tiny_pre
        inexact = False
        carry = False

        if self._exp > n and (p is None or self.p <= p):
            # fast path: definitely representable values
            kept = RealFloat(s=self._s, exp=self._exp, c=self._c)
        else:
            # normal path: need to split the value
            # step 1. split the number at the rounding position
            kept, lost = self.split(n)

            # step 2. check if rounding was exact (if so, we're done)
            if not lost.is_zero():
                # check that we're allowed to round
                inexact = True
                if exact:
                    raise ValueError(f'rounding off digits: self={self}, n={n}')

                # step 3. check if we need to increment
                increment = kept._round_increment(lost, n, rm)

                # step 4. increment if necessary
                if increment:
                    kept._c += 1
                    if p is not None:
                        if kept._c.bit_length() > p:
                            # adjust the exponent since we exceeded precision bounds
                            # the value is guaranteed to be a power of two
                            kept._c >>= 1
                            kept._exp += 1
                            carry = True

                        # possibly need to recompute tiny_post
                        if tiny_pre:
                            assert emin is not None
                            tiny_post = self._tiny_post(kept, emin, n, rm)

        # set flags
        kept._flags = Flags(tiny_pre=tiny_pre, tiny_post=tiny_post, inexact=inexact, carry=carry)
        return kept


    def _generate_randbits(self, rng: RNG | None, k: int) -> int:
        """
        Generates a random k-bit integer. If `rng` is `None`,
        then the default `Random` instance is used.
        """
        if rng is None:
            return random.getrandbits(k)
        elif isinstance(rng, random.Random):
            return rng.getrandbits(k)
        else:
            return int(rng.integers(0, 1 << k))

    def _round_at_stochastic(
        self,
        p: int | None,
        n: int,
        emin: int | None,
        rm: RoundingMode,
        num_randbits: int | None,
        rng: RNG | None,
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
            # the actual number of bits is limited by `(n + 1) - self.exp`
            num_randbits = max(0, (n + 1) - self._exp)

        # step 2. generate the randomness
        # NOTE: we always sample random bits even if `self` is representable
        # since the `rng` state should be mutated with every rounding
        randbits = self._generate_randbits(rng, num_randbits)

        # step 3. compute rounding parameters for extended-precision value
        n_rand = n - num_randbits

        # step 4. round the number to obtain the extended-precision value
        xr = self._round_at(None, n_rand, None, rm, exact)

        # step 5. split the number at the rounding position to get the rounding bits
        _, lost = xr.split(n)

        # step 6. check if rounding was exact (if so, we're done)
        if lost.is_zero():
            # just choose one of the rounding modes (RTZ)
            rand_rm = RoundingMode.RTZ
        else:
            # step 7. normalize `lost` so that `lost.n == n_rand`
            offset = lost._exp - (n_rand + 1)
            if offset > 0:
                lost_c = lost._c << offset
            elif offset < 0:
                lost_c = lost._c >> -offset
            else:
                lost_c = lost._c

            # step 8. stochastically choose a rounding mode
            round_up = randbits + lost_c >= (1 << num_randbits)
            rand_rm = RoundingMode.RAZ if round_up else RoundingMode.RTZ

        # step 9. apply rounding as normal
        return self._round_at(p, n, emin, rand_rm, exact)


    def round_at(
        self,
        n: int,
        p: int | None = None,
        rm: RoundingMode = RoundingMode.RNE,
        num_randbits: int | None = 0,
        *,
        rng: RNG | None = None,
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

        # step 1. compute `emin` for floating-point w/ subnormalization
        if p is not None and n is not None:
            emin = p + n
        else:
            emin = None

        # step 2. round at the specified position
        if num_randbits == 0:
            # non-stochastic rounding
            return self._round_at(p, n, emin, rm, exact)
        else:
            # stochastic rounding
            return self._round_at_stochastic(p, n, emin, rm, num_randbits, rng, exact)

    def round(self,
        max_p: int | None = None,
        min_n: int | None = None,
        rm: RoundingMode = RoundingMode.RNE,
        num_randbits: int | None = 0,
        *,
        rng: RNG | None = None,
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

        if max_p is None and min_n is None:
            raise ValueError(f'must specify {max_p} or {min_n}')

        # step 1. compute rounding parameters
        p, n = self._round_params(max_p, min_n)

        # step 2. compute `emin` for floating-point w/ subnormalization
        if max_p is not None and min_n is not None:
            emin = max_p + min_n
        else:
            emin = None

        # step 2. round at the specified position
        if num_randbits == 0:
            # non-stochastic rounding
            return self._round_at(p, n, emin, rm, exact)
        else:
            # stochastic rounding
            return self._round_at_stochastic(p, n, emin, rm, num_randbits, rng, exact)
