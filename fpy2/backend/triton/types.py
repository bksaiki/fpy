"""
triton backend: Triton storage types.

A *storage type* is the Triton dtype a value is held in at runtime.  As in the
cpp backend it is distinct from the *rounding format*: storage shapes the
value's declaration, rounding shapes which arithmetic the result respects.

The ladder is narrower than the cpp backend's in one direction and wider in
another: it gains ``fp16``, which C++ cannot spell, and it deliberately
excludes ``bf16`` and everything below fp16.  Excluding ``bf16`` is what makes
the float rungs a *chain* -- bf16 (es 8, prec 8) and fp16 (es 5, prec 11) are
mutually incomparable, and ordering two incomparable float rungs was the one
open question in this module.  The sub-fp16 formats remain reachable as
rounding targets through ``unfold_round``; they are simply never storage.
"""

import enum
from collections.abc import Iterable
from typing import TypeAlias

from ...utils import default_repr, enum_repr


@enum_repr
class TritonScalar(enum.Enum):
    """Concrete Triton scalar storage types."""

    BOOL = 0
    F16 = 1
    F32 = 2
    F64 = 3
    U8 = 4
    U16 = 5
    U32 = 6
    U64 = 7
    S8 = 8
    S16 = 9
    S32 = 10
    S64 = 11

    def is_integer(self) -> bool:
        return self in INT_TYPES

    def is_float(self) -> bool:
        return self in FLOAT_TYPES

    def is_signed(self) -> bool:
        """Is this a signed integer type?  ``False`` for every non-integer."""
        return self in SIGNED_INT_TYPES

    def int_bits(self) -> int | None:
        """Width an integer type's values wrap at, or ``None`` for a
        non-integer."""
        return _INT_BITS.get(self)

    def float_bits(self) -> int | None:
        """Width of a float type, or ``None`` for a non-float.  Orders the
        float types, which is what says whether a conversion narrows."""
        return _FLOAT_BITS.get(self)

    def format(self) -> str:
        """The `triton.language` dtype that spells this storage."""
        return _SPELLING[self]


_SPELLING: dict[TritonScalar, str] = {
    TritonScalar.BOOL: 'tl.int1',
    TritonScalar.F16: 'tl.float16',
    TritonScalar.F32: 'tl.float32',
    TritonScalar.F64: 'tl.float64',
    TritonScalar.U8: 'tl.uint8',
    TritonScalar.U16: 'tl.uint16',
    TritonScalar.U32: 'tl.uint32',
    TritonScalar.U64: 'tl.uint64',
    TritonScalar.S8: 'tl.int8',
    TritonScalar.S16: 'tl.int16',
    TritonScalar.S32: 'tl.int32',
    TritonScalar.S64: 'tl.int64',
}


@default_repr
class TritonTuple:
    """A tuple of values.

    Triton has no runtime tuple: a `@triton.jit` function traces Python, so a
    tuple exists at *trace* time and its elements become separate values.  It
    is kept as a storage type because FPy's types carry it and the emitter has
    to destructure it, not because anything is allocated.
    """
    elts: tuple['TritonType', ...]

    def __init__(self, elts: Iterable['TritonType']):
        self.elts = tuple(elts)

    def __eq__(self, other):
        return isinstance(other, TritonTuple) and self.elts == other.elts

    def __hash__(self):
        return hash((TritonTuple, self.elts))

    def format(self) -> str:
        return f'({", ".join(e.format() for e in self.elts)})'


TritonType: TypeAlias = TritonScalar | TritonTuple
"""All Triton storage types.

There is no list type on purpose.  A Triton value is a scalar or a tile, and an
FPy list is neither: a list of *proven* length unrolls into one value per
element (`ArraySizeInfer` + `Specialize(size_key=True)` + `ForUnroll`), and one
of unproven length is refused until the tensorization work of
the op table in :mod:`.target`.
"""


FLOAT_TYPES = [TritonScalar.F16, TritonScalar.F32, TritonScalar.F64]
UNSIGNED_INT_TYPES = [
    TritonScalar.U8, TritonScalar.U16, TritonScalar.U32, TritonScalar.U64,
]
SIGNED_INT_TYPES = [
    TritonScalar.S8, TritonScalar.S16, TritonScalar.S32, TritonScalar.S64,
]
INT_TYPES = SIGNED_INT_TYPES + UNSIGNED_INT_TYPES


_INT_BITS: dict[TritonScalar, int] = {
    TritonScalar.U8: 8, TritonScalar.U16: 16,
    TritonScalar.U32: 32, TritonScalar.U64: 64,
    TritonScalar.S8: 8, TritonScalar.S16: 16,
    TritonScalar.S32: 32, TritonScalar.S64: 64,
}
"""Value width of each integer type; see :meth:`TritonScalar.int_bits`."""

_FLOAT_BITS: dict[TritonScalar, int] = {
    TritonScalar.F16: 16, TritonScalar.F32: 32, TritonScalar.F64: 64,
}
"""Width of each float type; see :meth:`TritonScalar.float_bits`."""
