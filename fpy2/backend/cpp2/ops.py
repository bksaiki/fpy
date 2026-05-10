"""
cpp2 backend: operation type tables.

Each primitive operation is parameterized by *argument formats* and
an *active rounding context*.  Inputs carry only value-range info
(a :class:`Format`) — the rounding has already happened upstream.
The operation rounds its mathematical result under the active
context, which is why the *output* slot carries a full
:class:`Context` (format + rounding mode), not just a format.

At an op site the emitter consults:

- The **active rounding context** (from
  :class:`ContextUseAnalysis.find_scope_from_use`) — must equal the
  signature's ``out_ctx``.
- Each operand's **rounding format** (from ``FormatInfer``) — must
  fit (via :func:`format_fits_in`) within the corresponding
  signature's ``in_fmt``.

When a signature matches, the emitter uses :func:`choose_storage`
on the signature's input formats to derive the C++ scalar types and
casts each operand to that type via :meth:`_maybe_cast`.  The
rounding-mode half of the active context is enforced separately by
Phase 5b's ``fesetround`` boundary, so emission only depends on the
formats and storage selection.

This mirrors :mod:`fpy2.backend.cpp.ops` in shape but the input
slots are :class:`Format`-typed and the output is :class:`Context`-typed.
"""

from __future__ import annotations

import dataclasses
from typing import TypeAlias

from ...ast.fpyast import Abs, Add, Div, Expr, Mul, Neg, Sub
from ...analysis.format_infer import FormatBound
from ...number import (
    RM,
    INTEGER,
    SINT8, SINT16, SINT32, SINT64,
    UINT8, UINT16, UINT32, UINT64,
)
from ...number.context.context import Context
from ...number.context.format import Format
from ...number.context.ieee754 import IEEEContext

from .storage import format_fits_in


@dataclasses.dataclass(frozen=True)
class UnaryCppOp:
    """A single supported C++ signature for a unary FPy op,
    parameterized by an argument format and an active rounding
    context for the output."""
    name: str
    is_func: bool
    arg_fmt: Format
    out_ctx: Context

    def matches(self, arg_fmt: FormatBound, active_ctx: Context) -> bool:
        return (
            self.out_ctx == active_ctx
            and format_fits_in(arg_fmt, self.arg_fmt)
        )

    def format(self, arg: str) -> str:
        return f'{self.name}({arg})' if self.is_func else f'({self.name}{arg})'


@dataclasses.dataclass(frozen=True)
class BinaryCppOp:
    """A single supported C++ signature for a binary FPy op,
    parameterized by per-operand argument formats and an active
    rounding context for the output."""
    name: str
    is_infix: bool
    in1_fmt: Format
    in2_fmt: Format
    out_ctx: Context

    def matches(
        self,
        in1_fmt: FormatBound,
        in2_fmt: FormatBound,
        active_ctx: Context,
    ) -> bool:
        return (
            self.out_ctx == active_ctx
            and format_fits_in(in1_fmt, self.in1_fmt)
            and format_fits_in(in2_fmt, self.in2_fmt)
        )

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
# We enumerate same-context signatures (input formats taken from the
# context's ``.format()``) for every native context the cpp2 backend
# supports.  FP bases pair with each ``fesetround``-supported
# rounding mode (RNE/RTZ/RTP/RTN) so context equality at the
# dispatch site can match the active RM exactly.

_FP_RMS = (RM.RNE, RM.RTZ, RM.RTP, RM.RTN)


def _fp_ctxs() -> list[Context]:
    """All FP32 / FP64 contexts the cpp2 backend dispatches against
    — one per supported rounding mode."""
    return [
        IEEEContext(es, nbits, rm)
        for (es, nbits) in ((8, 32), (11, 64))
        for rm in _FP_RMS
    ]


def _int_ctxs() -> list[Context]:
    """Integer contexts the cpp2 backend dispatches against.
    ``INTEGER`` is the unbounded fallback — kept for backward compat
    with programs that don't pin a width."""
    return [
        SINT8, SINT16, SINT32, SINT64,
        UINT8, UINT16, UINT32, UINT64,
        INTEGER,
    ]


def _all_arith_ctxs() -> list[Context]:
    return _fp_ctxs() + _int_ctxs()


def _make_unary_table() -> UnaryOpTable:
    fp = _fp_ctxs()
    ints = _int_ctxs()
    same = _all_arith_ctxs()
    return {
        Neg: [UnaryCppOp('-', is_func=False, arg_fmt=c.format(), out_ctx=c)
              for c in same],
        Abs: (
            [UnaryCppOp('std::fabs', is_func=True,
                        arg_fmt=c.format(), out_ctx=c)
             for c in fp]
            + [UnaryCppOp('std::abs', is_func=True,
                          arg_fmt=c.format(), out_ctx=c)
               for c in ints]
        ),
    }


def _make_binary_table() -> BinaryOpTable:
    same = _all_arith_ctxs()
    return {
        op_cls: [
            BinaryCppOp(name, True, c.format(), c.format(), c)
            for c in same
        ]
        for op_cls, name in (
            (Add, '+'),
            (Sub, '-'),
            (Mul, '*'),
            (Div, '/'),
        )
    }


def make_op_table() -> ScalarOpTable:
    """Build the default cpp2 op table."""
    return ScalarOpTable(
        unary=_make_unary_table(),
        binary=_make_binary_table(),
    )
