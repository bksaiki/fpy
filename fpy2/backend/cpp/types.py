"""
cpp backend: C++ storage types.

A *storage type* is the C++ type used to hold a value at runtime.  It
is distinct from the *rounding format* (per :mod:`format_infer`) — the
two are related but not equal: storage shapes the variable's
declaration, rounding shapes which arithmetic the result respects.

The ladder mirrors :mod:`fpy2.backend.cpp.types` but storage selection
is driven by :class:`FormatInfer` rather than concrete contexts.
"""

import enum
from collections.abc import Iterable
from typing import TypeAlias

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
    """An FPy list: a shared ``std::shared_ptr<std::vector<T>>`` handle;
    where :mod:`fpy2.analysis.alias` proves nothing can observe the
    difference, a plain ``std::vector<T>``; and where the length is also
    proven, a ``std::array<T, K>``.  See :mod:`.unbox` for how both are
    decided.

    The handle exists because an FPy list is a list of *references*: binding
    shares its cells, so two names can hold the same elements and a write
    through either is visible to both.  A ``std::vector`` -- a value -- cannot
    represent that identity; a ``shared_ptr`` to one can, and a program
    embedding a kernel can hold either without depending on anything of ours.
    Refcounting needs no collector: a cycle needs a list to hold itself, and
    list types are finite, so ``xs[0] = xs`` fails to unify -- the *type*
    system rules it out, not the store rules, which would permit it.

    A *value* list whose length the backend proved is a ``size`` here and a
    ``std::array<T, K>`` in the output.  ``size`` is part of the type's
    identity, like ``boxed``: two lists differing in representation or length
    compare unequal, so neither ``storage_infer`` nor a stamped type can
    silently conflate them.  (``fpy2.types.ListType.length`` is *metadata*,
    excluded from equality -- do not repeat that here: an equality that
    ignores ``size`` would let ``std::array<double, 3>`` pass for
    ``std::array<double, 4>`` and the C++ compiler would be the first to
    notice.)

    A boxed list never carries a size: sharing already costs a heap
    allocation, so a static length buys nothing, and one representation per
    axis keeps the conversion lattice small.  Enforced here, not merely
    avoided.
    """
    elt: 'CppType'
    boxed: bool
    size: int | None

    def __init__(
        self, elt: 'CppType', boxed: bool = True, size: int | None = None,
    ):
        assert not (boxed and size is not None), (
            f'a boxed list cannot carry a static size: {size}'
        )
        self.elt = elt
        self.boxed = boxed
        self.size = size

    def __eq__(self, other):
        return (
            isinstance(other, CppList)
            and self.elt == other.elt
            and self.boxed == other.boxed
            and self.size == other.size
        )

    def __hash__(self):
        return hash((CppList, self.elt, self.boxed, self.size))

    def format(self) -> str:
        elt = self.elt.format()
        if self.boxed:
            return f'std::shared_ptr<std::vector<{elt}>>'
        if self.size is not None:
            return f'std::array<{elt}, {self.size}>'
        return f'std::vector<{elt}>'


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
