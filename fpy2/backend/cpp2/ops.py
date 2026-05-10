"""
cpp2 backend: operation type tables.

C++ doesn't have ad-hoc polymorphism for primitive numeric ops — each
operator only takes a fixed set of operand-type combinations.  This
module enumerates those combinations as :class:`UnaryCppOp` /
:class:`BinaryCppOp` records keyed by the FPy AST node type, and the
emitter dispatches through the resulting :class:`ScalarOpTable`.

For an FPy expression like ``a + b`` evaluated under context ``C``,
emission tries (in order):

1. **Direct match.**  If the operand storage types and the result
   storage type already match a signature, emit it as-is.
2. **Cast-to-result.**  If the operands fit in the result type per
   :func:`scalar_fits_in`, try a signature whose operand and result
   types both equal the result type — inserting an explicit
   ``static_cast`` on each narrower operand.
3. **Reject.**  Otherwise raise a ``Cpp2EmitError`` pointing at the
   AST node and the offending operand types.

This mirrors :mod:`fpy2.backend.cpp.ops` in shape but is keyed off
:class:`CppScalar` directly (no ``ScalarOpTable`` options, no
target-table indirection — Phase 5a is just about getting the types
right).
"""

from __future__ import annotations

import dataclasses
from typing import TypeAlias

from ...ast.fpyast import Abs, Add, Div, Expr, Mul, Neg, Sub

from .types import CppScalar, FLOAT_TYPES, INT_TYPES


@dataclasses.dataclass(frozen=True)
class UnaryCppOp:
    """A single supported C++ signature for a unary FPy op."""
    name: str
    is_func: bool
    arg: CppScalar
    ret: CppScalar

    def matches(self, arg: CppScalar, ret: CppScalar) -> bool:
        return self.arg == arg and self.ret == ret

    def format(self, arg: str) -> str:
        return f'{self.name}({arg})' if self.is_func else f'({self.name}{arg})'


@dataclasses.dataclass(frozen=True)
class BinaryCppOp:
    """A single supported C++ signature for a binary FPy op."""
    name: str
    is_infix: bool
    lhs: CppScalar
    rhs: CppScalar
    ret: CppScalar

    def matches(self, lhs: CppScalar, rhs: CppScalar, ret: CppScalar) -> bool:
        return self.lhs == lhs and self.rhs == rhs and self.ret == ret

    def format(self, lhs: str, rhs: str) -> str:
        if self.is_infix:
            return f'({lhs} {self.name} {rhs})'
        return f'{self.name}({lhs}, {rhs})'


UnaryOpTable: TypeAlias = dict[type[Expr], list[UnaryCppOp]]
BinaryOpTable: TypeAlias = dict[type[Expr], list[BinaryCppOp]]


@dataclasses.dataclass
class ScalarOpTable:
    """Per-op-kind tables of supported C++ signatures."""
    unary: UnaryOpTable
    binary: BinaryOpTable


# ---------------------------------------------------------------------
# Default tables.
#
# Phase 5a covers the ops the emitter already lowers inline:
# ``Neg``, ``Abs`` (unary) and ``Add``, ``Sub``, ``Mul``, ``Div``
# (binary).  Algebraic / transcendental / rounding ops land with the
# rest of Phase 5 and add their entries here.

def _make_unary_table() -> UnaryOpTable:
    return {
        Neg: (
            [UnaryCppOp('-', is_func=False, arg=ty, ret=ty)
             for ty in FLOAT_TYPES + INT_TYPES]
        ),
        Abs: (
            [UnaryCppOp('std::fabs', is_func=True, arg=ty, ret=ty)
             for ty in FLOAT_TYPES]
            + [UnaryCppOp('std::abs', is_func=True, arg=ty, ret=ty)
               for ty in INT_TYPES]
        ),
    }


def _make_binary_table() -> BinaryOpTable:
    arith_tys = FLOAT_TYPES + INT_TYPES
    return {
        Add: [BinaryCppOp('+', True, ty, ty, ty) for ty in arith_tys],
        Sub: [BinaryCppOp('-', True, ty, ty, ty) for ty in arith_tys],
        Mul: [BinaryCppOp('*', True, ty, ty, ty) for ty in arith_tys],
        Div: [BinaryCppOp('/', True, ty, ty, ty) for ty in arith_tys],
    }


def make_op_table() -> ScalarOpTable:
    """Build the default cpp2 op table."""
    return ScalarOpTable(
        unary=_make_unary_table(),
        binary=_make_binary_table(),
    )
