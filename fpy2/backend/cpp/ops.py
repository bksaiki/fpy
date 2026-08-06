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
import enum
from typing import TypeAlias

from ...ast.fpyast import Expr
from ...number.context.context import Context
from .storage import choose_storage_scalar
from .types import CppScalar


@enum.unique
class CppOpStyle(enum.Enum):
    """How a signature spells itself in C++."""
    CALL = 'call'      # f(a, b)
    INFIX = 'infix'    # (a op b)
    PREFIX = 'prefix'  # (op a)


@dataclasses.dataclass(frozen=True)
class CppOp:
    """One supported C++ signature for an FPy op: the input slots it takes,
    the active rounding context its output is correct under, and how it is
    spelled.

    ``cast_out`` wraps the emitted form in a ``static_cast`` to the output
    context's storage -- for primitives returning something narrower or
    otherwise different from what the output demands, such as ``std::ilogb``
    returning C ``int`` where the context maps to ``int64_t``.  Sound by
    precondition: such primitives are paired with op variants guaranteeing the
    value fits.
    """
    name: str
    in_tys: tuple[CppScalar, ...]
    out_ctx: Context
    style: CppOpStyle = CppOpStyle.CALL
    cast_out: bool = False

    @property
    def is_call(self) -> bool:
        """Does this emit ``f(args)``?  An infix or prefix operator takes its
        type from the operands instead, which is what decides whether a literal
        argument needs its own type spelled."""
        return self.style is CppOpStyle.CALL

    def matches(
        self, in_tys: tuple[CppScalar, ...], active_ctx: Context,
    ) -> bool:
        """Exactly this signature, no conversions."""
        return self.out_ctx == active_ctx and self.in_tys == in_tys

    def format(self, *args: str) -> str:
        match self.style:
            case CppOpStyle.INFIX:
                lhs, rhs = args
                out = f'({lhs} {self.name} {rhs})'
            case CppOpStyle.PREFIX:
                (arg,) = args
                out = f'({self.name}{arg})'
            case CppOpStyle.CALL:
                out = f'{self.name}({", ".join(args)})'
        if self.cast_out:
            out_ty = choose_storage_scalar(self.out_ctx.format())
            return f'static_cast<{out_ty.format()}>({out})'
        return out


UnaryOpTable: TypeAlias = dict[type[Expr], list[CppOp]]
BinaryOpTable: TypeAlias = dict[type[Expr], list[CppOp]]
TernaryOpTable: TypeAlias = dict[type[Expr], list[CppOp]]


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
