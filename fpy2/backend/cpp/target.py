"""
cpp backend: default target description.

A "target description" tells the cpp emitter which primitive ops it
can dispatch and which C++ surface to emit for each one.  This module
populates the abstract :class:`ScalarOpTable` from
:mod:`fpy2.backend.cpp.ops` with the default set: ``<cmath>``
functions over IEEE-754 FP32 / FP64 (one signature per supported
``fesetround`` rounding mode), basic arithmetic over IEEE-754 floats
and every native integer width on the storage ladder, and a handful
of cross-domain hooks (``std::abs`` for ints, ``std::ilogb`` to
extract an integer exponent from a float).

Callers obtain a populated table via :func:`make_op_table`.  The
emitter consumes it as an opaque :class:`ScalarOpTable`, so future
targets can swap in a different table without changing the emitter.
"""

from __future__ import annotations

from ...ast.fpyast import (
    Abs,
    Acos,
    Acosh,
    Add,
    Asin,
    Asinh,
    Atan,
    Atan2,
    Atanh,
    Cbrt,
    Ceil,
    Copysign,
    Cos,
    Cosh,
    Div,
    Erf,
    Erfc,
    Exp,
    Exp2,
    Expm1,
    Fdim,
    Floor,
    Fma,
    Fmod,
    Hypot,
    Lgamma,
    Log,
    Log1p,
    Log2,
    Log10,
    Logb,
    Mul,
    NearbyInt,
    Neg,
    Pow,
    Remainder,
    RoundInt,
    Sin,
    Sinh,
    Sqrt,
    Sub,
    Tan,
    Tanh,
    Tgamma,
    Trunc,
)
from ...number import (
    INTEGER,
    RM,
    SINT8,
    SINT16,
    SINT32,
    SINT64,
    UINT8,
    UINT16,
    UINT32,
    UINT64,
)
from ...number.context.context import Context
from ...number.context.ieee754 import IEEEContext
from .ops import (
    BinaryCppOp,
    BinaryOpTable,
    ScalarOpTable,
    TernaryCppOp,
    TernaryOpTable,
    UnaryCppOp,
    UnaryOpTable,
)
from .storage import choose_storage_scalar
from .types import CppScalar

# ---------------------------------------------------------------------
# Native context inventory.
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


# ---------------------------------------------------------------------
# Per-arity factory helpers — generate signature sets parameterized
# by the contexts above.

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


# ---------------------------------------------------------------------
# Op → C++ name mappings.

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


# ---------------------------------------------------------------------
# Per-arity table builders.

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

    # ``logb`` under an integer output context lowers to
    # ``std::ilogb`` plus an explicit widening cast to the output's
    # storage.  ``std::ilogb`` returns C ``int`` (≥ 16 bits, usually
    # 32) — wider than any realistic binary exponent for the FP
    # inputs the cpp backend supports, so casting to the output's
    # storage type is value-preserving in practice.
    table[Logb] = table[Logb] + [
        UnaryCppOp(
            'std::ilogb', is_func=True,
            arg_ty=_ty_of(fp_ctx), out_ctx=int_ctx,
            cast_out=True,
        )
        for fp_ctx in fp
        for int_ctx in ints
    ]
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


# ---------------------------------------------------------------------
# Public entry point.

def make_op_table() -> ScalarOpTable:
    """Build the cpp backend's default :class:`ScalarOpTable`."""
    return ScalarOpTable(
        unary=_make_unary_table(),
        binary=_make_binary_table(),
        ternary=_make_ternary_table(),
    )
