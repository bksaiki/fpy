"""
This module defines the number format type.
"""

from abc import ABC, abstractmethod
from fractions import Fraction

from ..number import Float, RealFloat

__all__ = [
    'Format',
    'OrdinalFormat',
    'SizedFormat',
    'EncodableFormat',
]


class Format(ABC):
    """
    Number format type.

    A number format describes the set of representable values
    and the encoding of those values without specifying a rounding rule.
    It is a projection of a `Context` that retains only the
    format-specific parameters.

    Use `Context.format()` to obtain the format of a context
    and `Context.from_format()` to create a context from a format.
    """

    @abstractmethod
    def is_equiv(self, other: 'Format') -> bool:
        """
        Returns if this format and another format represent
        the same set of values.
        """
        ...

    @abstractmethod
    def representable_in(self, x: Float | RealFloat) -> bool:
        """
        Returns if `x` is representable under this format.

        Representable is not the same as canonical,
        but every canonical value must be representable.
        """
        ...

    @abstractmethod
    def canonical_under(self, x: Float) -> bool:
        """
        Returns if `x` is canonical under this format.

        This function only considers relevant attributes to judge
        if a value is canonical. Thus, there may be more than
        one canonical value for a given number despite the function name.
        The result of `self.normalize()` is always canonical.
        """
        ...

    @abstractmethod
    def normal_under(self, x: Float) -> bool:
        """
        Returns if `x` is "normal" under this format.

        For IEEE-style formats, this means that `x` is finite, non-zero,
        and `x.normalize()` has full precision.
        """
        ...

    @abstractmethod
    def normalize(self, x: Float) -> Float:
        """Returns the canonical form of `x` under this format."""
        ...


class OrdinalFormat(Format):
    """
    Number format for formats that map to ordinal numbers.

    Most common number formats fall under this category.
    There exists a bijection between representable values
    and a subset of the integers.
    """

    @abstractmethod
    def to_ordinal(self, x: Float, infval: bool = False) -> int:
        """
        Maps a number to an ordinal number.

        When `infval=True`, infinities are mapped to the next (or previous)
        logical ordinal value after +/-MAX_VAL. This option is only
        valid when the format has a maximum value.
        """
        ...

    @abstractmethod
    def to_fractional_ordinal(self, x: Float) -> Fraction:
        """
        Maps a number to a (fractional) ordinal number.

        Unlike `self.to_ordinal(x)`, the argument `x` does not
        have to be representable under this format.
        If `x` is representable, then
        `self.to_ordinal(x) == self.to_fractional_ordinal(x)`.
        If `x` is not representable, then
        `self.to_fractional_ordinal(x)` is not an integer;
        it is up to the format to decide how to interpolate
        between representable numbers.

        Raises a `ValueError` when `x.is_nar()` is `True`.
        """
        ...

    @abstractmethod
    def from_ordinal(self, x: int, infval: bool = False) -> Float:
        """
        Maps an ordinal number to a number.

        When `infval=True`, infinities are mapped to the next (or previous)
        logical ordinal value after +/-MAX_VAL. This option is only
        valid when the format has a maximum value.
        """
        ...

    @abstractmethod
    def minval(self, s: bool = False) -> Float:
        """
        Returns the (signed) representable value with the minimum magnitude
        under this format.

        This value will map to +/-1 through `to_ordinal()`.
        """
        ...

    def _next_towards(self, x: Float, y: Float, allow_inf: bool = False) -> Float:
        """
        Returns the next representable value after `x` towards `y`.
        """
        if y.isinf:
            step = -1 if y.s else 1
            return self.from_ordinal(self.to_ordinal(x) + step, infval=allow_inf)
        else:
            xord = self.to_ordinal(x)
            yord = self.to_ordinal(y)
            step = 1 if xord < yord else -1
            return self.from_ordinal(xord + step, infval=allow_inf)

    def _next_away(self, x: Float, y: Float, allow_inf: bool = False) -> Float:
        """
        Returns the next representable value after `x` away from `y`.
        """
        if y.isinf:
            step = 1 if y.s else -1
            return self.from_ordinal(self.to_ordinal(x) + step, infval=allow_inf)
        else:
            xord = self.to_ordinal(x)
            yord = self.to_ordinal(y)
            step = -1 if xord < yord else 1
            return self.from_ordinal(xord + step, infval=allow_inf)

    def next_towards(self, x: Float, y: Float, allow_inf: bool = False) -> Float:
        """
        Returns the next representable value after `x` towards `y`.

        Raises a `ValueError` if
        - `x` or `y` is not representable under this format, or
        - `x` or `y` is `NaN`, or
        - `x` is infinite and `allow_inf` is `False`
        - `x == y`
        """
        if not self.representable_in(x):
            raise ValueError('x is not representable under this format', x)
        if not self.representable_in(y):
            raise ValueError('y is not representable under this format', y)
        if x.isnan:
            raise ValueError('x is NaN', x)
        if y.isnan:
            raise ValueError('y is NaN', y)
        if not allow_inf and x.isinf:
            raise ValueError('x is infinite', x)
        if x == y:
            raise ValueError('x and y are equal so there is no next value towards y', x, y)
        return self._next_towards(x, y, allow_inf)

    def next_towards_zero(self, x: Float, allow_inf: bool = False) -> Float:
        """
        Returns the next representable value after `x` towards zero.

        Raises a `ValueError` if
        - `x` is not representable under this format, or
        - `x` is `NaN`, or
        - `x` is infinite and `allow_inf` is `False`
        - `x == 0`
        """
        if not self.representable_in(x):
            raise ValueError('x is not representable under this format', x)
        if x.isnan:
            raise ValueError('x is NaN', x)
        if not allow_inf and x.isinf:
            raise ValueError('x is infinite', x)
        if x == 0:
            raise ValueError('x is zero so there is no next value towards zero', x)
        return self._next_towards(x, self.from_ordinal(0), allow_inf)

    def next_away_zero(self, x: Float, allow_inf: bool = False) -> Float:
        """
        Returns the next representable value after `x` away from zero.

        Raises a `ValueError` if
        - `x` is not representable under this format, or
        - `x` is `NaN`, or
        - `x` is infinite and `allow_inf` is `False`
        - `x == 0`
        """
        if not self.representable_in(x):
            raise ValueError('x is not representable under this format', x)
        if x.isnan:
            raise ValueError('x is NaN', x)
        if not allow_inf and x.isinf:
            raise ValueError('x is infinite', x)
        if x == 0:
            raise ValueError('x is zero so there is no next value away from zero', x)
        return self._next_away(x, self.from_ordinal(0), allow_inf)

    def next_up(self, x: Float, allow_inf: bool = False) -> Float:
        """
        Returns the next representable value after `x` towards positive infinity.

        Raises a `ValueError` if
        - `x` is not representable under this format, or
        - `x` is `NaN`, or
        - `x` is infinite and `allow_inf` is `False`
        - `x == +inf`
        """
        if not self.representable_in(x):
            raise ValueError('x is not representable under this format', x)
        if x.isnan:
            raise ValueError('x is NaN', x)
        if not allow_inf and x.isinf:
            raise ValueError('x is infinite', x)
        if x.isinf and not x.s:
            raise ValueError('x cannot be positive infinity', x)
        return self._next_towards(x, Float(isinf=True), allow_inf)

    def next_down(self, x: Float, allow_inf: bool = False) -> Float:
        """
        Returns the next representable value after `x` towards negative infinity.

        Raises a `ValueError` if
        - `x` is not representable under this format, or
        - `x` is `NaN`, or
        - `x` is infinite and `allow_inf` is `False`
        - `x == -inf`
        """
        if not self.representable_in(x):
            raise ValueError('x is not representable under this format', x)
        if x.isnan:
            raise ValueError('x is NaN', x)
        if not allow_inf and x.isinf:
            raise ValueError('x is infinite', x)
        if x.isinf and x.s:
            raise ValueError('x cannot be negative infinity', x)
        return self._next_towards(x, Float(isinf=True, s=True), allow_inf)


class SizedFormat(OrdinalFormat):
    """
    Number format for formats encodable in a fixed size.

    These formats may be mapped to ordinal numbers, and they
    have a (positive) minimum and (positive) maximum value.
    """

    @property
    def emax(self) -> int:
        """
        The normalized exponent of the maximum representable value
        under this format.
        """
        pos_e = self.largest().e
        neg_e = self.smallest().e
        return max(pos_e, neg_e)

    @abstractmethod
    def maxval(self, s: bool = False) -> Float:
        """
        Returns the (signed) representable value with the maximum magnitude
        under this format.

        If `self.maxval() == 0`, then this format cannot represent
        any finite, non-zero values.
        """
        ...

    @abstractmethod
    def infval(self, s: bool = False) -> Float:
        """
        Returns the (signed) value that is the "next" value after
        the maximum representable value under this format.
        """
        ...

    @abstractmethod
    def largest(self) -> Float:
        """
        Returns the largest representable value (towards positive infinity)
        under this format.
        """
        ...

    @abstractmethod
    def smallest(self) -> Float:
        """
        Returns the smallest representable value (towards negative infinity)
        under this format.
        """
        ...


class EncodableFormat(SizedFormat):
    """
    Number format for formats that can be encoded as bitstrings.

    Most common number formats fall under this category.
    These formats define a way to encode a number in memory.
    """

    @abstractmethod
    def total_bits(self) -> int:
        """Returns the total number of bits used to encode a number under this format."""
        ...

    @abstractmethod
    def encode(self, x: Float) -> int:
        """
        Encodes a number constructed under this format as a bitstring.
        This operation is format dependent.
        """
        ...

    @abstractmethod
    def decode(self, x: int) -> Float:
        """
        Decodes a bitstring as a number constructed under this format.
        This operation is format dependent.
        """
        ...
