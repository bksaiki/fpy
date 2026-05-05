"""
cpp2 backend: C++ storage types.

A *storage type* is the C++ type used to hold a value at runtime.  It
is distinct from the *rounding format* (per :mod:`format_infer`) — the
two are related but not equal: storage shapes the variable's
declaration, rounding shapes which arithmetic the result respects.

The ladder mirrors :mod:`fpy2.backend.cpp.types` but storage selection
is driven by :class:`FormatInfer` rather than concrete contexts.
"""

import enum

from typing import TypeAlias, Iterable

from ...utils import default_repr, enum_repr


@enum_repr
class CppScalar(enum.Enum):
    """Concrete C++ scalar storage types."""

    BOOL = 0
    F32 = 1
    F64 = 2
    U8 = 3
    U16 = 4
    U32 = 5
    U64 = 6
    S8 = 7
    S16 = 8
    S32 = 9
    S64 = 10

    def is_integer(self) -> bool:
        return self in INT_TYPES

    def is_float(self) -> bool:
        return self in FLOAT_TYPES

    def format(self) -> str:
        match self:
            case CppScalar.BOOL: return 'bool'
            case CppScalar.F32: return 'float'
            case CppScalar.F64: return 'double'
            case CppScalar.U8: return 'uint8_t'
            case CppScalar.U16: return 'uint16_t'
            case CppScalar.U32: return 'uint32_t'
            case CppScalar.U64: return 'uint64_t'
            case CppScalar.S8: return 'int8_t'
            case CppScalar.S16: return 'int16_t'
            case CppScalar.S32: return 'int32_t'
            case CppScalar.S64: return 'int64_t'


@default_repr
class CppList:
    """``std::vector<T>``."""
    elt: 'CppType'

    def __init__(self, elt: 'CppType'):
        self.elt = elt

    def __eq__(self, other):
        return isinstance(other, CppList) and self.elt == other.elt

    def __hash__(self):
        return hash((CppList, self.elt))

    def format(self) -> str:
        return f'std::vector<{self.elt.format()}>'


@default_repr
class CppTuple:
    """``std::tuple<T1, …, Tn>``."""
    elts: tuple['CppType', ...]

    def __init__(self, elts: Iterable['CppType']):
        self.elts = tuple(elts)

    def __eq__(self, other):
        return isinstance(other, CppTuple) and self.elts == other.elts

    def __hash__(self):
        return hash((CppTuple, self.elts))

    def format(self) -> str:
        elts = ', '.join(elt.format() for elt in self.elts)
        return f'std::tuple<{elts}>'


CppType: TypeAlias = CppScalar | CppList | CppTuple
"""All C++ storage types."""


FLOAT_TYPES = [CppScalar.F32, CppScalar.F64]
UNSIGNED_INT_TYPES = [CppScalar.U8, CppScalar.U16, CppScalar.U32, CppScalar.U64]
SIGNED_INT_TYPES = [CppScalar.S8, CppScalar.S16, CppScalar.S32, CppScalar.S64]
INT_TYPES = SIGNED_INT_TYPES + UNSIGNED_INT_TYPES
