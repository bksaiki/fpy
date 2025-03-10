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
    def round(self, x):
        """Rounds any digital number according to this context."""
        raise NotImplementedError('virtual method')

class SizedContext(Context):
    """
    Rounding context for formats encodable in a fixed size.

    Most common number formats fall under this category.
    These formats define a way to encode a number in memory.
    """

    @abstractmethod
    def maxval(self, s: bool = False):
        """Returns the maximum (signed) real number under this context."""
        raise NotImplementedError('virtual method')

    @abstractmethod
    def minval(self, s: bool = False):
        """Returns the minimum (signed) real number under this context."""
        raise NotImplementedError('virtual method')

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
