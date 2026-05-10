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
the ``fesetround`` boundary emitted around ``with`` blocks, so
emission only depends on the formats and storage selection.

This mirrors :mod:`fpy2.backend.cpp.ops` in shape but the input
slots are :class:`Format`-typed and the output is :class:`Context`-typed.
"""

from __future__ import annotations

import dataclasses
from typing import TypeAlias

from ...ast.fpyast import (
    Abs, Acos, Acosh, Add, Asin, Asinh, Atan, Atan2, Atanh, Cbrt,
    Ceil, Copysign, Cos, Cosh, Div, Erf, Erfc, Exp, Exp2, Expm1,
    Expr, Fdim, Fma, Floor, Fmod, Hypot, Lgamma, Log, Log10, Log1p,
    Log2, Logb, Mul, NearbyInt, Neg, Pow, Remainder, RoundInt,
    Sin, Sinh, Sqrt, Sub, Tan, Tanh, Tgamma, Trunc,
)
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


@dataclasses.dataclass(frozen=True)
class TernaryCppOp:
    """A single supported C++ signature for a ternary FPy op
    (e.g., ``Fma``).  Same parameterization as the binary case."""
    name: str
    in1_fmt: Format
    in2_fmt: Format
    in3_fmt: Format
    out_ctx: Context

    def matches(
        self,
        in1_fmt: FormatBound,
        in2_fmt: FormatBound,
        in3_fmt: FormatBound,
        active_ctx: Context,
    ) -> bool:
        return (
            self.out_ctx == active_ctx
            and format_fits_in(in1_fmt, self.in1_fmt)
            and format_fits_in(in2_fmt, self.in2_fmt)
            and format_fits_in(in3_fmt, self.in3_fmt)
        )

    def format(self, a: str, b: str, c: str) -> str:
        return f'{self.name}({a}, {b}, {c})'


UnaryOpTable: TypeAlias = dict[type[Expr], list[UnaryCppOp]]
BinaryOpTable: TypeAlias = dict[type[Expr], list[BinaryCppOp]]
TernaryOpTable: TypeAlias = dict[type[Expr], list[TernaryCppOp]]


@dataclasses.dataclass
class ScalarOpTable:
    """Per-op-kind tables of supported C++ signatures."""
    unary: UnaryOpTable
    binary: BinaryOpTable
    ternary: TernaryOpTable


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


def _fp_unary(name: str) -> list[UnaryCppOp]:
    """Same-context FP-only unary signatures for one ``<cmath>``
    function.  One sig per FP context (FP32 / FP64 × the four
    supported rounding modes)."""
    return [
        UnaryCppOp(name, is_func=True, arg_fmt=c.format(), out_ctx=c)
        for c in _fp_ctxs()
    ]


def _fp_binary(name: str) -> list[BinaryCppOp]:
    """Same-context FP-only binary signatures for one ``<cmath>``
    function (function-call form, not infix)."""
    return [
        BinaryCppOp(name, is_infix=False,
                    in1_fmt=c.format(), in2_fmt=c.format(), out_ctx=c)
        for c in _fp_ctxs()
    ]


def _fp_ternary(name: str) -> list[TernaryCppOp]:
    """Same-context FP-only ternary signatures."""
    return [
        TernaryCppOp(name,
                     in1_fmt=c.format(), in2_fmt=c.format(),
                     in3_fmt=c.format(), out_ctx=c)
        for c in _fp_ctxs()
    ]


# ``<cmath>`` unary functions defined for FP types only.  Each entry
# pairs an FPy AST node with its C++ name; signatures cover both FP32
# and FP64 across every supported rounding mode.
_UNARY_CMATH = (
    # Roots
    (Sqrt, 'std::sqrt'),
    (Cbrt, 'std::cbrt'),
    # FP rounding to integer-valued FP
    (Ceil, 'std::ceil'),
    (Floor, 'std::floor'),
    (NearbyInt, 'std::nearbyint'),
    (RoundInt, 'std::round'),
    (Trunc, 'std::trunc'),
    # Trigonometric
    (Sin, 'std::sin'), (Cos, 'std::cos'), (Tan, 'std::tan'),
    (Asin, 'std::asin'), (Acos, 'std::acos'), (Atan, 'std::atan'),
    # Hyperbolic
    (Sinh, 'std::sinh'), (Cosh, 'std::cosh'), (Tanh, 'std::tanh'),
    (Asinh, 'std::asinh'), (Acosh, 'std::acosh'), (Atanh, 'std::atanh'),
    # Exp / log family
    (Exp, 'std::exp'), (Exp2, 'std::exp2'), (Expm1, 'std::expm1'),
    (Log, 'std::log'), (Log10, 'std::log10'),
    (Log1p, 'std::log1p'), (Log2, 'std::log2'),
    # Special functions
    (Erf, 'std::erf'), (Erfc, 'std::erfc'),
    (Lgamma, 'std::lgamma'), (Tgamma, 'std::tgamma'),
    # Numerical data
    (Logb, 'std::logb'),
)


_BINARY_CMATH = (
    (Pow, 'std::pow'),
    (Fmod, 'std::fmod'),
    (Remainder, 'std::remainder'),
    (Copysign, 'std::copysign'),
    (Fdim, 'std::fdim'),
    (Hypot, 'std::hypot'),
    (Atan2, 'std::atan2'),
)


_TERNARY_CMATH = (
    (Fma, 'std::fma'),
)


def _make_unary_table() -> UnaryOpTable:
    fp = _fp_ctxs()
    ints = _int_ctxs()
    same = _all_arith_ctxs()
    table: UnaryOpTable = {
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
    for op_cls, name in _UNARY_CMATH:
        table[op_cls] = _fp_unary(name)
    return table


def _make_binary_table() -> BinaryOpTable:
    same = _all_arith_ctxs()
    table: BinaryOpTable = {
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
    for op_cls, name in _BINARY_CMATH:
        table[op_cls] = _fp_binary(name)
    return table


def _make_ternary_table() -> TernaryOpTable:
    return {op_cls: _fp_ternary(name) for op_cls, name in _TERNARY_CMATH}


def make_op_table() -> ScalarOpTable:
    """Build the default cpp2 op table."""
    return ScalarOpTable(
        unary=_make_unary_table(),
        binary=_make_binary_table(),
        ternary=_make_ternary_table(),
    )
