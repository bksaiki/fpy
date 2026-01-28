"""
MPFX backend: target-specific types.
"""

from abc import ABC, abstractmethod
from typing import Iterable

__all__ = [
    'CppType',
    'CppBoolType',
    'CppDoubleType',
    'CppInt64Type',
    'CppTupleType',
    'CppListType',
    'CppContextType',
]

class CppType(ABC):
    """C++ type abstraction."""

    @abstractmethod
    def to_cpp(self) -> str:
        """Convert to C++ type string."""
        ...

class CppBoolType(CppType):
    """C++ boolean type."""

    def to_cpp(self) -> str:
        return 'bool'

class CppDoubleType(CppType):
    """C++ double type."""

    def to_cpp(self) -> str:
        return 'double'

class CppInt64Type(CppType):
    """C++ int64 type."""

    def to_cpp(self) -> str:
        return 'int64_t'

class CppTupleType(CppType):
    """C++ tuple type."""
    elts: tuple[CppType, ...]

    def __init__(self, elts: Iterable[CppType]):
        self.elts = tuple(elts)

    def to_cpp(self) -> str:
        elt_strs = ', '.join(elt.to_cpp() for elt in self.elts)
        return f'std::tuple<{elt_strs}>'

class CppListType(CppType):
    """C++ list type."""
    elt: CppType

    def __init__(self, elt: CppType):
        self.elt = elt

    def to_cpp(self) -> str:
        return f'std::vector<{self.elt.to_cpp()}>'

class CppContextType(CppType):
    """C++ context type."""

    def to_cpp(self) -> str:
        return 'mpfx::Context'
