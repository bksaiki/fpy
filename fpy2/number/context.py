"""
This module defines the rounding context type.
"""

from abc import ABC, abstractmethod
from typing import Optional, TypeAlias, Self, Union

from .round import RoundingMode

from . import number

# avoids circular dependency issues (useful for type checking)
Float: TypeAlias = 'number.Float'
RealFloat: TypeAlias = 'number.RealFloat'

class Context(ABC):
    """
    Rounding context type.

    Most mathematical operators on digital numbers
    can be decomposed into two steps:

    1. a mathematically-correct operation over real numbers,
    interpreting digital numbers as real numbers;

    2. a rounding operation to limit the number significant digits
    and decide how the "lost" digits will affect the final output.

    Thus, rounding enforces a particular "format" for digital numbers,
    but they should just be considered unbounded real numbers
    when in isolation. The characteristics of the rounding operation are
    summarized by this type.
    """

    def __enter__(self) -> Self:
        raise RuntimeError('do not call directly')

    def __exit__(self, *args) -> None:
        raise RuntimeError('do not call directly')

    @abstractmethod
    def with_rm(self, rm: RoundingMode) -> Self:
        """Returns `self` but with rounding mode `rm`."""
        ...

    @abstractmethod
    def is_representable(self, x: Union[Float, RealFloat]) -> bool:
        """
        Returns if `x` is representable under this context.

        Representable is not the same as canonical,
        but every canonical value must be representable.
        """
        ...

    @abstractmethod
    def is_canonical(self, x: Float) -> bool:
        """
        Returns if `x` is canonical under this context.

        This function only considers relevant attributes to judge
        if a value is canonical. Thus, there may be more than
        one canonical value for a given number despite the function name.
        The result of `self.normalize()` is always canonical.
        """
        ...

    @abstractmethod
    def normalize(self, x: Float) -> Float:
        """Returns the canonical form of `x` under this context."""
        ...

    @abstractmethod
    def is_normal(self, x: Float) -> bool:
        """
        Returns if `x` is "normal" under this context.

        For IEEE-style contexts, this means that `x` is finite, non-zero,
        and `x.normalize()` has full precision.
        """
        ...

    @abstractmethod
    def round_params(self) -> tuple[Optional[int], Optional[int]]:
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
    def round(self, x) -> Float:
        """Rounds any digital number according to this context."""
        ...

    @abstractmethod
    def round_at(self, x, n: int) -> Float:
        """
        Rounding any digital number of a representable value with
        an unnormalized exponent of at minimum `n + 1`.

        Rounding is done by the following rules:

        - if `x` is representable and has an unnormalized exponent
          of at minimum `n + 1`, then `self.round_n(x, n) == x`
        - if `x` is between two representable values `i1 < x < i2`
          where both `i1` and `i2` have unnormalized exponents of at
          minimum `n + 1`,  then the context information determines
          which value is returned.

        """
        ...

    def round_integer(self, x) -> Float:
        """
        Rounds any digital number to an integer according to this context.

        Rounding is done by the following rules:

        - if `x` is a representable integer, then `self.round_integer(x) == x`
        - if `x` is between two representable integers `i1 < x < i2`,
          then the context information determines which integer
          is returned.

        This is equivalent to `self.round_at(x, -1)`.
        """
        return self.round_at(x, -1)


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
        Maps a digital number to an ordinal number.

        When `infval=True`, infinities are mapped to the next (or previous)
        logical ordinal value after +/-MAX_VAL. This option is only
        valid when the context has a maximum value.
        """
        ...

    @abstractmethod
    def from_ordinal(self, x: int, infval: bool = False) -> Float:
        """
        Maps an ordinal number to a digital number.

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
        Encodes a digital number constructed under this context as a bitstring.
        This operation is context dependent.
        """
        ...

    @abstractmethod
    def decode(self, x: int) -> Float:
        """
        Decodes a bitstring as a a digital number constructed under this context.
        This operation is context dependent.
        """
        ...
