"""
This module defines the rounding context type.
"""

from abc import ABC, abstractmethod
from typing import Self
from fractions import Fraction

from ...utils import is_dyadic
from ..gmputils import mpfr_value
from ..number import Float, RealFloat

from .format import Format, OrdinalFormat, SizedFormat, EncodableFormat

__all__ = [
    'Context',
    'OrdinalContext',
    'SizedContext',
    'EncodableContext'
]


class Context(ABC):
    """
    Rounding context type.

    Most mathematical operators on numbers
    can be decomposed into two steps:

    1. a mathematically-correct operation over real numbers,
    interpreting numbers as real numbers;

    2. a rounding operation to limit the number significant digits
    and decide how the "lost" digits will affect the final output.

    Thus, rounding enforces a particular "format" for numbers,
    but they should just be considered unbounded real numbers
    when in isolation. The characteristics of the rounding operation are
    summarized by this type.
    """

    def __enter__(self) -> Self:
        raise RuntimeError('do not call directly')

    def __exit__(self, *args) -> None:
        raise RuntimeError('do not call directly')

    @abstractmethod
    def with_params(self, **kwargs) -> Self:
        """Returns `self` but with updated parameters."""
        ...

    @abstractmethod
    def is_stochastic(self) -> bool:
        """
        Returns if this context is stochastic.

        Stochastic contexts are used for probabilistic rounding.
        """
        ...

    @abstractmethod
    def format(self) -> Format:
        """
        Returns the number format associated with this context.

        The format describes the set of representable values without
        the rounding rule. Format-only methods on `Context`
        (`representable_under`, etc.) default to delegating
        to the corresponding method on `self.format()`.
        """
        ...

    def is_equiv(self, other: 'Context') -> bool:
        """
        Returns if this context and another context round values to
        the same set of representable values. Two contexts are equivalent
        if they produce the same set of representable values.
        """
        if not isinstance(other, Context):
            raise TypeError(f'Expected \'Context\', got \'{type(other)}\' for other={other}')
        try:
            self_fmt = self.format()
        except NotImplementedError:
            self_fmt = None
        try:
            other_fmt = other.format()
        except NotImplementedError:
            other_fmt = None
        return self_fmt == other_fmt

    def representable_under(self, x: Float | RealFloat) -> bool:
        """
        Returns if `x` is representable under this context.

        Representable is not the same as canonical,
        but every canonical value must be representable.
        """
        return self.format().representable_in(x)

    def canonical_under(self, x: Float) -> bool:
        """
        Returns if `x` is canonical under this context.

        This function only considers relevant attributes to judge
        if a value is canonical. Thus, there may be more than
        one canonical value for a given number despite the function name.
        The result of `self.normalize()` is always canonical.
        """
        return self.format().canonical_under(x)

    def normal_under(self, x: Float) -> bool:
        """
        Returns if `x` is "normal" under this context.

        For IEEE-style contexts, this means that `x` is finite, non-zero,
        and `x.normalize()` has full precision.
        """
        return self.format().normal_under(x)

    def normalize(self, x: Float) -> Float:
        """Returns the canonical form of `x` under this context."""
        return Float(x=self.format().normalize(x), ctx=self)

    @abstractmethod
    def round_params(self) -> tuple[int | None, int | None]:
        """
        Returns the rounding parameters `(max_p, min_n)` used for rounding
        under this context.

        - (p, None) => floating-point style rounding
        - (p, n) => floating-point style rounding with subnormalization
        - (None, n) => fixed-point style rounding
        - (None, None) => real computation; no rounding

        These parameters also determine the amount of precision for
        intermediate round-to-odd operations (provided by MPFR / `gmpy2`).
        """
        ...

    @abstractmethod
    def round(self, x, *, exact: bool = False) -> Float:
        """
        Rounds any number according to this context.

        If `exact=True`, then the rounding operation will raise a `ValueError`
        if rounding produces an inexact result.
        """
        ...

    @abstractmethod
    def round_at(self, x, n: int, *, exact: bool = False) -> Float:
        """
        Rounding any number of a representable value with
        an unnormalized exponent of at minimum `n + 1`.

        Rounding is done by the following rules:

        - if `x` is representable and has an unnormalized exponent
          of at minimum `n + 1`, then `self.round_n(x, n) == x`
        - if `x` is between two representable values `i1 < x < i2`
          where both `i1` and `i2` have unnormalized exponents of at
          minimum `n + 1`,  then the context information determines
          which value is returned.

        If `exact=True`, then the rounding operation will raise a `ValueError`
        if rounding produces an inexact result.
        """
        ...

    def round_integer(self, x) -> Float:
        """
        Rounds any number to an integer according to this context.

        Rounding is done by the following rules:

        - if `x` is a representable integer, then `self.round_integer(x) == x`
        - if `x` is between two representable integers `i1 < x < i2`,
          then the context information determines which integer
          is returned.

        This is equivalent to `self.round_at(x, -1)`.
        """
        return self.round_at(x, -1)

    def _round_prepare(self, x) -> RealFloat | Float:
        """
        Initial step during rounding.

        Converts a value to a `RealFloat` or a `Float` instance that
        can be rounded under this context.

        The value produced may not be numerically equal to `x`,
        but the rounded result will be the same as if `x` was rounded directly.

        Values supported:
        - FPy values: `Float`, `RealFloat`
        - Python numbers: `int`, `float`, `Fraction`
        - Python strings: `str`
        """
        match x:
            case Float() | RealFloat():
                return x
            case float():
                return Float.from_float(x)
            case int():
                return RealFloat.from_int(x)
            case Fraction():
                if x.denominator == 1:
                    return RealFloat.from_int(int(x))
                elif is_dyadic(x):
                    return RealFloat.from_rational(x)

        # not a special case so we use MPFR as a fallback
        # round the value using RTO such that we can re-round
        p, n = self.round_params()
        return mpfr_value(x, prec=p, n=n)


class OrdinalContext(Context):
    """
    Rounding context for formats that map to ordinal numbers.

    Most common number formats fall under this category.
    There exists a bijection between representable values
    and a subset of the integers.
    """

    @abstractmethod
    def format(self) -> OrdinalFormat:
        ...

    def to_ordinal(self, x: Float, infval: bool = False) -> int:
        """
        Maps a number to an ordinal number.

        When `infval=True`, infinities are mapped to the next (or previous)
        logical ordinal value after +/-MAX_VAL. This option is only
        valid when the context has a maximum value.
        """
        return self.format().to_ordinal(x, infval)

    def to_fractional_ordinal(self, x: Float) -> Fraction:
        """
        Maps a number to a (fractional) ordinal number.

        Unlike `self.to_ordinal(x)`, the argument `x` does not
        have to be representable under this context.
        If `x` is representable, then
        `self.to_ordinal(x) == self.to_fractional_ordinal(x)`.
        If `x` is not representable, then
        `self.to_fractional_ordinal(x)` is not an integer;
        it is up to the context to decide how to interpolate
        between representable numbers.

        Raises a `ValueError` when `x.is_nar()` is `True`.
        """
        return self.format().to_fractional_ordinal(x)

    def from_ordinal(self, x: int, infval: bool = False) -> Float:
        """
        Maps an ordinal number to a number.

        When `infval=True`, infinities are mapped to the next (or previous)
        logical ordinal value after +/-MAX_VAL. This option is only
        valid when the context has a maximum value.
        """
        return Float(x=self.format().from_ordinal(x, infval), ctx=self)

    def minval(self, s: bool = False) -> Float:
        """
        Returns the (signed) representable value with the minimum magnitude
        under this context.

        This value will map to +/-1 through `to_ordinal()`.
        """
        return Float(x=self.format().minval(s), ctx=self)

    def next_towards(self, x: Float, y: Float, allow_inf: bool = False) -> Float:
        """
        Returns the next representable value after `x` towards `y`.

        Raises a `ValueError` if
        - `x` or `y` is not representable under this context, or
        - `x` or `y` is `NaN`, or
        - `x` is infinite and `allow_inf` is `False`
        - `x == y`
        """
        if not self.representable_under(x):
            raise ValueError('x is not representable under this context', x)
        if not self.representable_under(y):
            raise ValueError('y is not representable under this context', y)
        if x.isnan:
            raise ValueError('x is NaN', x)
        if y.isnan:
            raise ValueError('y is NaN', y)
        if not allow_inf and x.isinf:
            raise ValueError('x is infinite', x)
        if x == y:
            raise ValueError('x and y are equal so there is no next value towards y', x, y)
        return Float(x=self.format().next_towards(x, y, allow_inf), ctx=self)

    def next_towards_zero(self, x: Float, allow_inf: bool = False) -> Float:
        """
        Returns the next representable value after `x` towards zero.

        Raises a `ValueError` if
        - `x` is not representable under this context, or
        - `x` is `NaN`, or
        - `x` is infinite and `allow_inf` is `False`
        - `x == 0`
        """
        if not self.representable_under(x):
            raise ValueError('x is not representable under this context', x)
        if x.isnan:
            raise ValueError('x is NaN', x)
        if not allow_inf and x.isinf:
            raise ValueError('x is infinite', x)
        if x == 0:
            raise ValueError('x is zero so there is no next value towards zero', x)
        return Float(x=self.format().next_towards_zero(x, allow_inf), ctx=self)

    def next_away_zero(self, x: Float, allow_inf: bool = False) -> Float:
        """
        Returns the next representable value after `x` away from zero.

        Raises a `ValueError` if
        - `x` is not representable under this context, or
        - `x` is `NaN`, or
        - `x` is infinite and `allow_inf` is `False`
        - `x == 0`
        """
        if not self.representable_under(x):
            raise ValueError('x is not representable under this context', x)
        if x.isnan:
            raise ValueError('x is NaN', x)
        if not allow_inf and x.isinf:
            raise ValueError('x is infinite', x)
        if x == 0:
            raise ValueError('x is zero so there is no next value away from zero', x)
        return Float(x=self.format().next_away_zero(x, allow_inf), ctx=self)

    def next_up(self, x: Float, allow_inf: bool = False) -> Float:
        """
        Returns the next representable value after `x` towards positive infinity.

        Raises a `ValueError` if
        - `x` is not representable under this context, or
        - `x` is `NaN`, or
        - `x` is infinite and `allow_inf` is `False`
        - `x == +inf`
        """
        if not self.representable_under(x):
            raise ValueError('x is not representable under this context', x)
        if x.isnan:
            raise ValueError('x is NaN', x)
        if not allow_inf and x.isinf:
            raise ValueError('x is infinite', x)
        if x.isinf and not x.s:
            raise ValueError('x cannot be positive infinity', x)
        return Float(x=self.format().next_up(x, allow_inf), ctx=self)

    def next_down(self, x: Float, allow_inf: bool = False) -> Float:
        """
        Returns the next representable value after `x` towards negative infinity.

        Raises a `ValueError` if
        - `x` is not representable under this context, or
        - `x` is `NaN`, or
        - `x` is infinite and `allow_inf` is `False`
        - `x == -inf`
        """
        if not self.representable_under(x):
            raise ValueError('x is not representable under this context', x)
        if x.isnan:
            raise ValueError('x is NaN', x)
        if not allow_inf and x.isinf:
            raise ValueError('x is infinite', x)
        if x.isinf and x.s:
            raise ValueError('x cannot be negative infinity', x)
        return Float(x=self.format().next_down(x, allow_inf), ctx=self)


class SizedContext(OrdinalContext):
    """
    Rounding context for formats encodable in a fixed size.

    These formats may be mapped to ordinal numbers, and they
    have a (positive) minimum and (positive) maximum value.
    """

    @property
    def emax(self) -> int:
        """
        The normalized exponent of the maximum representable value
        under this context.
        """
        pos_e = self.largest().e
        neg_e = self.smallest().e
        return max(pos_e, neg_e)

    @abstractmethod
    def format(self) -> SizedFormat:
        ...

    def maxval(self, s: bool = False) -> Float:
        """
        Returns the (signed) representable value with the maximum magnitude
        under this context.

        If `self.maxval() == 0`, then this context cannot represent
        any finite, non-zero values.
        """
        return Float(x=self.format().maxval(s), ctx=self)

    def infval(self, s: bool = False) -> Float:
        """
        Returns the (signed) value that is the "next" value after
        the maximum representable value under this context.
        """
        return Float(x=self.format().infval(s), ctx=self)

    def largest(self) -> Float:
        """
        Returns the largest representable value (towards positive infinity)
        under this context.
        """
        return Float(x=self.format().largest(), ctx=self)

    def smallest(self) -> Float:
        """
        Returns the smallest representable value (towards negative infinity)
        under this context.
        """
        return Float(x=self.format().smallest(), ctx=self)


class EncodableContext(SizedContext):
    """
    Rounding context for formats that can be encoded as bitstrings.

    Most common number formats fall under this category.
    These formats define a way to encode a number in memory.
    """

    @abstractmethod
    def format(self) -> EncodableFormat:
        ...

    def total_bits(self) -> int:
        """Returns the total number of bits used to encode a number under this context."""
        return self.format().total_bits()

    def encode(self, x: Float) -> int:
        """
        Encodes a number constructed under this context as a bitstring.
        This operation is context dependent.
        """
        return self.format().encode(x)

    def decode(self, x: int) -> Float:
        """
        Decodes a bitstring as a a number constructed under this context.
        This operation is context dependent.
        """
        return Float(x=self.format().decode(x), ctx=self)
