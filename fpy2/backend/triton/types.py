"""
Triton backend: storage types.

A *storage type* is the Triton dtype a value is held in at runtime.  As in the
cpp backend it is distinct from the *rounding format*: storage shapes the
value's declaration, rounding shapes which arithmetic the result respects.

Unlike the cpp backend's, the ladder has ``fp16``.  It excludes ``bf16``,
incomparable with fp16, so the float rungs form a chain, and everything below
fp16, which remains reachable as a rounding target through ``unfold_round``.
"""

import enum

from ...utils import enum_repr


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

    def int_bits(self) -> int | None:
        """Width an integer type's values wrap at, or ``None`` for a
        non-integer."""
        return _INT_BITS.get(self)

    def float_bits(self) -> int | None:
        """Width of a float type, or ``None`` for a non-float."""
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
