"""
cpp backend: per-op signature abstractions.

This module defines the *shape* of the cpp backend's operator
descriptions — the per-arity ``CppOp`` dataclasses and the
``ScalarOpTable`` that groups them — without prescribing *which*
operators are supported.  See :mod:`fpy2.backend.cpp.target` for the
default target description (the set of operators and signatures the
emitter dispatches against).

Each primitive operation is parameterized by *argument C++ types*
(:class:`CppScalar`) and an *active rounding context*
(:class:`Context`).  The split mirrors what emission actually needs:

- A signature's input slots are the concrete C++ scalar types the
  generated code feeds the operator.  ``int8_t + int8_t`` is one
  signature, ``float + float`` is another.
- The output slot is the active rounding context.  Its C++ type
  (``choose_storage(out_ctx.format())``) determines the result's
  storage; its rounding mode is enforced separately by the
  ``fesetround`` boundary emitted around ``with`` blocks.

At an op site the emitter consults:

- The **active rounding context** from
  :class:`ContextUseAnalysis.find_scope_from_use` — must equal the
  signature's ``out_ctx``.
- Each operand's **C++ storage type** from
  :class:`StorageAnalysis` — must equal the signature's input
  slot.  On mismatch the emitter falls back to the
  all-active-context signature and inserts an explicit
  ``static_cast`` per operand.
"""

from __future__ import annotations

import dataclasses
from typing import TypeAlias

from ...ast.fpyast import Expr
from ...number.context.context import Context
from .storage import choose_storage_scalar
from .types import CppScalar


@dataclasses.dataclass(frozen=True)
class UnaryCppOp:
    """A single supported C++ signature for a unary FPy op,
    parameterized by an argument C++ type and an active rounding
    context for the output.

    When ``cast_out`` is ``True`` the emitted call is wrapped in a
    ``static_cast`` to the output context's storage type — used by
    ops whose underlying C++ primitive returns a *narrower* /
    differently-typed value than the output storage demands (e.g.
    ``std::ilogb`` returns ``int`` but the output context might map
    to ``int64_t``).  The cast is sound by precondition: such
    primitives are paired with op variants that guarantee the
    returned value fits in the destination width."""
    name: str
    is_func: bool
    arg_ty: CppScalar
    out_ctx: Context
    cast_out: bool = False

    def matches(self, arg_ty: CppScalar, active_ctx: Context) -> bool:
        return self.out_ctx == active_ctx and self.arg_ty == arg_ty

    def format(self, arg: str) -> str:
        call = f'{self.name}({arg})' if self.is_func else f'({self.name}{arg})'
        if self.cast_out:
            out_ty = choose_storage_scalar(self.out_ctx.format())
            return f'static_cast<{out_ty.format()}>({call})'
        return call


@dataclasses.dataclass(frozen=True)
class BinaryCppOp:
    """A single supported C++ signature for a binary FPy op,
    parameterized by per-operand C++ types and an active rounding
    context for the output."""
    name: str
    is_infix: bool
    in1_ty: CppScalar
    in2_ty: CppScalar
    out_ctx: Context

    def matches(
        self,
        in1_ty: CppScalar,
        in2_ty: CppScalar,
        active_ctx: Context,
    ) -> bool:
        return (
            self.out_ctx == active_ctx
            and self.in1_ty == in1_ty
            and self.in2_ty == in2_ty
        )

    def format(self, lhs: str, rhs: str) -> str:
        if self.is_infix:
            return f'({lhs} {self.name} {rhs})'
        return f'{self.name}({lhs}, {rhs})'


@dataclasses.dataclass(frozen=True)
class TernaryCppOp:
    """A single supported C++ signature for a ternary FPy op
    (e.g., ``Fma``).  Same parameterization as the binary case."""
    name: str
    in1_ty: CppScalar
    in2_ty: CppScalar
    in3_ty: CppScalar
    out_ctx: Context

    def matches(
        self,
        in1_ty: CppScalar,
        in2_ty: CppScalar,
        in3_ty: CppScalar,
        active_ctx: Context,
    ) -> bool:
        return (
            self.out_ctx == active_ctx
            and self.in1_ty == in1_ty
            and self.in2_ty == in2_ty
            and self.in3_ty == in3_ty
        )

    def format(self, a: str, b: str, c: str) -> str:
        return f'{self.name}({a}, {b}, {c})'


UnaryOpTable: TypeAlias = dict[type[Expr], list[UnaryCppOp]]
BinaryOpTable: TypeAlias = dict[type[Expr], list[BinaryCppOp]]
TernaryOpTable: TypeAlias = dict[type[Expr], list[TernaryCppOp]]


@dataclasses.dataclass
class ScalarOpTable:
    """Per-op-kind tables of supported C++ signatures.

    A target description (see :mod:`fpy2.backend.cpp.target`)
    populates one of these and hands it to the emitter; the emitter
    looks up signatures by ``(op type, operand C++ types, active
    rounding context)`` at each dispatch site.
    """
    unary: UnaryOpTable
    binary: BinaryOpTable
    ternary: TernaryOpTable
