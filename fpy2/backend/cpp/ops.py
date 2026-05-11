"""
cpp backend: operation type tables.

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

This mirrors :mod:`fpy2.backend.cpp.ops` in shape, but the output
slot carries a full :class:`Context` (so RM-specific signatures are
distinct) rather than a bare :class:`CppScalar`.
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
from ...number import (
    RM,
    INTEGER,
    SINT8, SINT16, SINT32, SINT64,
    UINT8, UINT16, UINT32, UINT64,
)
from ...number.context.context import Context
from ...number.context.ieee754 import IEEEContext

from .storage import choose_storage_scalar
from .types import CppScalar


@dataclasses.dataclass(frozen=True)
class UnaryCppOp:
    """A single supported C++ signature for a unary FPy op,
    parameterized by an argument C++ type and an active rounding
    context for the output."""
    name: str
    is_func: bool
    arg_ty: CppScalar
    out_ctx: Context

    def matches(self, arg_ty: CppScalar, active_ctx: Context) -> bool:
        return self.out_ctx == active_ctx and self.arg_ty == arg_ty

    def format(self, arg: str) -> str:
        return f'{self.name}({arg})' if self.is_func else f'({self.name}{arg})'


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
    """Per-op-kind tables of supported C++ signatures."""
    unary: UnaryOpTable
    binary: BinaryOpTable
    ternary: TernaryOpTable


# ---------------------------------------------------------------------
# Default tables.
#
# We enumerate same-context signatures (input C++ types taken from
# ``choose_storage_scalar(ctx.format())``) for every native context
# the cpp backend supports.  FP bases pair with each
# ``fesetround``-supported rounding mode (RNE/RTZ/RTP/RTN) so context
# equality at the dispatch site can match the active RM exactly.

_FP_RMS = (RM.RNE, RM.RTZ, RM.RTP, RM.RTN)


def _fp_ctxs() -> list[Context]:
    """All FP32 / FP64 contexts the cpp backend dispatches against
    — one per supported rounding mode."""
    return [
        IEEEContext(es, nbits, rm)
        for (es, nbits) in ((8, 32), (11, 64))
        for rm in _FP_RMS
    ]


def _int_ctxs() -> list[Context]:
    """Integer contexts the cpp backend dispatches against.
    ``INTEGER`` is the unbounded fallback — kept for backward compat
    with programs that don't pin a width."""
    return [
        SINT8, SINT16, SINT32, SINT64,
        UINT8, UINT16, UINT32, UINT64,
        INTEGER,
    ]


def _all_arith_ctxs() -> list[Context]:
    return _fp_ctxs() + _int_ctxs()


def _ty_of(ctx: Context) -> CppScalar:
    """Map a same-context signature's context to the C++ scalar that
    its inputs (and output) carry."""
    return choose_storage_scalar(ctx.format())


def _fp_unary(name: str) -> list[UnaryCppOp]:
    """Same-context FP-only unary signatures for one ``<cmath>``
    function.  One sig per FP context (FP32 / FP64 × the four
    supported rounding modes)."""
    return [
        UnaryCppOp(name, is_func=True, arg_ty=_ty_of(c), out_ctx=c)
        for c in _fp_ctxs()
    ]


def _fp_binary(name: str) -> list[BinaryCppOp]:
    """Same-context FP-only binary signatures for one ``<cmath>``
    function (function-call form, not infix)."""
    return [
        BinaryCppOp(name, is_infix=False,
                    in1_ty=_ty_of(c), in2_ty=_ty_of(c), out_ctx=c)
        for c in _fp_ctxs()
    ]


def _fp_ternary(name: str) -> list[TernaryCppOp]:
    """Same-context FP-only ternary signatures."""
    return [
        TernaryCppOp(name,
                     in1_ty=_ty_of(c), in2_ty=_ty_of(c),
                     in3_ty=_ty_of(c), out_ctx=c)
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
        Neg: [UnaryCppOp('-', is_func=False, arg_ty=_ty_of(c), out_ctx=c)
              for c in same],
        Abs: (
            [UnaryCppOp('std::fabs', is_func=True,
                        arg_ty=_ty_of(c), out_ctx=c)
             for c in fp]
            + [UnaryCppOp('std::abs', is_func=True,
                          arg_ty=_ty_of(c), out_ctx=c)
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
            BinaryCppOp(name, True, _ty_of(c), _ty_of(c), c)
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
    """Build the default cpp op table."""
    return ScalarOpTable(
        unary=_make_unary_table(),
        binary=_make_binary_table(),
        ternary=_make_ternary_table(),
    )
