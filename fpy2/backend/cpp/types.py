"""
C++ backend: types
"""

import enum

from typing import TypeAlias, Iterable

from ...utils import default_repr, enum_repr


@enum_repr
class CppScalar(enum.Enum):
    """
    C++ types.

    Each type represents either

    t ::= bool | real R

    where R is a concrete rounding context.
    """

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

    def format(self):
        match self:
            case CppScalar.BOOL:
                return 'bool'
            case CppScalar.F32:
                return 'float'
            case CppScalar.F64:
                return 'double'
            case CppScalar.U8:
                return 'uint8_t'
            case CppScalar.U16:
                return 'uint16_t'
            case CppScalar.U32:
                return 'uint32_t'
            case CppScalar.U64:
                return 'uint64_t'
            case CppScalar.S8:
                return 'int8_t'
            case CppScalar.S16:
                return 'int16_t'
            case CppScalar.S32:
                return 'int32_t'
            case CppScalar.S64:
                return 'int64_t'

@default_repr
class CppList:
    elt: 'CppType'

    def __init__(self, elt: 'CppType'):
        self.elt = elt

    def __eq__(self, other):
        return isinstance(other, CppList) and self.elt == other.elt

    def format(self):
        return f'std::vector<{self.elt.format()}>'

    def dim(self) -> int:
        match self.elt:
            case CppList():
                return self.elt.dim() + 1
            case _:
                return 1

@default_repr
class CppTuple:
    elts: tuple['CppType', ...]

    def __init__(self, elts: Iterable['CppType']):
        self.elts = tuple(elts)

    def __eq__(self, other):
        return isinstance(other, CppTuple) and self.elts == other.elts

    def format(self):
        elts = ', '.join(elt.format() for elt in self.elts)
        return f'std::tuple<{elts}>'


CppType: TypeAlias = CppScalar | CppList | CppTuple


FLOAT_TYPES = [
    CppScalar.F32,
    CppScalar.F64
]

INT_TYPES = [
    CppScalar.S8,
    CppScalar.S16,
    CppScalar.S32,
    CppScalar.S64,
    CppScalar.U8,
    CppScalar.U16,
    CppScalar.U32,
    CppScalar.U64
]

ALL_SCALARS = [CppScalar.BOOL] + FLOAT_TYPES + INT_TYPES
