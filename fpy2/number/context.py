"""
This module defines the rounding context type.
"""

from abc import ABC, abstractmethod

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

    @abstractmethod
    def is_representable(self, x) -> bool:
        """Returns if `x` is representable under this context."""
        raise NotImplementedError('virtual method')

    @abstractmethod
    def is_canonical(self, x) -> bool:
        """
        Returns if `x` is canonical under this context.

        This function only considers relevant attributes to judge
        if a value is canonical. Thus, there may be more than
        one canonical value for a given number despite the function name.
        The result of `self.normalize()` is always canonical.
        """
        raise NotImplementedError('virtual method')

    @abstractmethod
    def normalize(self, x):
        """Returns the canonical form of `x` under this context."""
        raise NotImplementedError('virtual method')

    @abstractmethod
    def round(self, x):
        """Rounds any digital number according to this context."""
        raise NotImplementedError('virtual method')


class OrdinalContext(Context):
    """
    Rounding context for formats that map to ordinal numbers.

    Most common number formats fall under this category.
    There exists a bijection between representable values
    and a subset of the integers.
    """

    @abstractmethod
    def to_ordinal(self, x, infval: bool = False) -> int:
        """
        Maps a digital number to an ordinal number.

        When `infval=True`, infinities are mapped to the next (or previous)
        logical ordinal value after +/-MAX_VAL. This option is only
        valid when the context has a maximum value.
        """
        raise NotImplementedError('virtual method')

    @abstractmethod
    def from_ordinal(self, x: int, infval: bool = False):
        """
        Maps an ordinal number to a digital number.
        
        When `infval=True`, infinities are mapped to the next (or previous)
        logical ordinal value after +/-MAX_VAL. This option is only
        valid when the context has a maximum value.
        """
        raise NotImplementedError('virtual method')

    @abstractmethod
    def minval(self, s: bool = False):
        """
        Returns the (signed) representable value with the minimum magnitude
        under this context.

        This value will map to +/-1 through `to_ordinal()`.
        """
        raise NotImplementedError('virtual method')


class SizedContext(OrdinalContext):
    """
    Rounding context for formats encodable in a fixed size.

    These formats may be mapped to ordinal numbers, and they
    have a (positive) minimum and (positive) maximum value.
    """

    @abstractmethod
    def maxval(self, s: bool = False):
        """
        Returns the (signed) representable value with the maximum magnitude
        under this context.
        """
        raise NotImplementedError('virtual method')


class EncodableContext(SizedContext):
    """
    Rounding context for formats that can be encoded as bitstrings.

    Most common number formats fall under this category.
    These formats define a way to encode a number in memory.
    """

    @abstractmethod
    def encode(self, x) -> int:
        """
        Encodes a digital number constructed under this context as a bitstring.
        This operation is context dependent.
        """
        raise NotImplementedError('virtual method')

    @abstractmethod
    def decode(self, x: int):
        """
        Decodes a bitstring as a a digital number constructed under this context.
        This operation is context dependent.
        """
        raise NotImplementedError('virtual method')
