"""
This module defines the rounding context type.
"""

from abc import ABC, abstractmethod
from typing import TypeAlias, Self, Union
from fractions import Fraction

from . import number
from ..utils import is_dyadic

__all__ = [
    'Context',
    'OrdinalContext',
    'SizedContext',
    'EncodableContext'
]

# avoids circular dependency issues (useful for type checking)
Float: TypeAlias = 'number.Float'
RealFloat: TypeAlias = 'number.RealFloat'

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
    def is_equiv(self, other: 'Context') -> bool:
        """
        Returns if this context and another context round values to
        the same set of representable values. Two contexts are equivalent
        if they produce the same set of representable values.
        """
        ...

    @abstractmethod
    def representable_under(self, x: Union[Float, RealFloat]) -> bool:
        """
        Returns if `x` is representable under this context.

        Representable is not the same as canonical,
        but every canonical value must be representable.
        """
        ...

    @abstractmethod
    def canonical_under(self, x: Float) -> bool:
        """
        Returns if `x` is canonical under this context.

        This function only considers relevant attributes to judge
        if a value is canonical. Thus, there may be more than
        one canonical value for a given number despite the function name.
        The result of `self.normalize()` is always canonical.
        """
        ...

    @abstractmethod
    def normal_under(self, x: Float) -> bool:
        """
        Returns if `x` is "normal" under this context.

        For IEEE-style contexts, this means that `x` is finite, non-zero,
        and `x.normalize()` has full precision.
        """
        ...

    @abstractmethod
    def normalize(self, x: Float) -> Float:
        """Returns the canonical form of `x` under this context."""
        ...

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

    def _round_prepare(self, x) -> Union[RealFloat, Float]:
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
        # get around circular import issues
        from .number import Float, RealFloat

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

        # get around circular import issues
        from .gmp import mpfr_value

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
    def to_ordinal(self, x: Float, infval: bool = False) -> int:
        """
        Maps a number to an ordinal number.

        When `infval=True`, infinities are mapped to the next (or previous)
        logical ordinal value after +/-MAX_VAL. This option is only
        valid when the context has a maximum value.
        """
        ...

    @abstractmethod
    def from_ordinal(self, x: int, infval: bool = False) -> Float:
        """
        Maps an ordinal number to a number.

        When `infval=True`, infinities are mapped to the next (or previous)
        logical ordinal value after +/-MAX_VAL. This option is only
        valid when the context has a maximum value.
        """
        ...

    @abstractmethod
    def minval(self, s: bool = False) -> Float:
        """
        Returns the (signed) representable value with the minimum magnitude
        under this context.

        This value will map to +/-1 through `to_ordinal()`.
        """
        ...


class SizedContext(OrdinalContext):
    """
    Rounding context for formats encodable in a fixed size.

    These formats may be mapped to ordinal numbers, and they
    have a (positive) minimum and (positive) maximum value.
    """

    @abstractmethod
    def maxval(self, s: bool = False) -> Float:
        """
        Returns the (signed) representable value with the maximum magnitude
        under this context.
        """
        ...


class EncodableContext(SizedContext):
    """
    Rounding context for formats that can be encoded as bitstrings.

    Most common number formats fall under this category.
    These formats define a way to encode a number in memory.
    """

    @abstractmethod
    def encode(self, x: Float) -> int:
        """
        Encodes a number constructed under this context as a bitstring.
        This operation is context dependent.
        """
        ...

    @abstractmethod
    def decode(self, x: int) -> Float:
        """
        Decodes a bitstring as a a number constructed under this context.
        This operation is context dependent.
        """
        ...
