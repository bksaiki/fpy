"""
triton backend: target description.

Which primitive ops the emitter may dispatch, and what Triton surface to emit
for each.  The shapes here (:class:`TritonOp`, :class:`ScalarOpTable`) parallel
the cpp backend's in :mod:`fpy2.backend.cpp.ops`; they are not shared because
the cpp ones are parameterized by ``CppScalar`` and spell C++.  Unifying them
is a refactor of the cpp backend, not of this one.

**The table is deliberately small**, and every omission is a refusal rather
than a fallback.  The contract is that if the compiler succeeds, the emitted
code must behave as the interpreter does.  On a
target whose defaults trade numerical agreement for throughput, holding that
means declining to name an operation whose Triton spelling is not the one FPy
specifies.  What is left out, and why:

- **Every transcendental.**  There is no correctly-rounded `exp`, `log`, `sin`
  or `erf` in Triton: `tl.exp` goes through libdevice `__nv_expf` and `tl.exp2`
  may lower to `ex2.approx.f32`.  The cpp backend has the same exposure and
  handles it by excluding 27 operators from its bit-exact differential check.
  Emitting none of them instead makes that exclusion list *empty*, so every
  function this backend compiles can also be checked bit-for-bit.
- **Division at FP16.**  Triton types `fp16 / fp16` as **fp32** -- there is no
  hardware divide below 32 bits -- so the result is an fp32 quotient narrowed
  back, which is a double rounding and not FP16's division.
- **Integer division.**  FPy's integer contexts round toward zero; Triton's
  ``//`` floors.  They disagree on every negative quotient.
- **``/`` and ``tl.sqrt``.**  Both are the *fast* variants.  The correctly
  rounded ones are ``tl.div_rn`` and ``tl.sqrt_rn``, and those are what the
  table names.
- **Every rounding mode but RNE.**  Triton exposes no per-instruction rounding
  modifier, so a non-RNE context reaches codegen through ``unfold_round`` or
  not at all.
"""

from __future__ import annotations

import dataclasses
import enum
from functools import cache
from typing import TypeAlias

from ...ast.fpyast import (
    Abs,
    Add,
    Ceil,
    Div,
    Expr,
    Floor,
    Fma,
    Mul,
    Neg,
    Sqrt,
    Sub,
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
from .storage import choose_storage_scalar
from .types import TritonScalar

__all__ = [
    'ScalarOpTable',
    'TritonOp',
    'TritonOpStyle',
    'castable',
    'directed_cast',
    'is_native_ctx',
    'make_op_table',
]


@enum.unique
class TritonOpStyle(enum.Enum):
    """How a signature spells itself in Triton."""
    CALL = 'call'      # f(a, b)
    INFIX = 'infix'    # (a op b)
    PREFIX = 'prefix'  # (op a)


@dataclasses.dataclass(frozen=True)
class TritonOp:
    """One supported Triton signature: input slots, the rounding context its
    output is correct under, and how it is spelled."""
    name: str
    in_tys: tuple[TritonScalar, ...]
    out_ctx: Context
    style: TritonOpStyle = TritonOpStyle.CALL

    @property
    def is_call(self) -> bool:
        return self.style is TritonOpStyle.CALL

    def matches(
        self, in_tys: tuple[TritonScalar, ...], active_ctx: Context,
    ) -> bool:
        """Exactly this signature, no conversions."""
        return self.out_ctx == active_ctx and self.in_tys == in_tys

    def format(self, *args: str) -> str:
        match self.style:
            case TritonOpStyle.INFIX:
                lhs, rhs = args
                return f'({lhs} {self.name} {rhs})'
            case TritonOpStyle.PREFIX:
                (arg,) = args
                return f'({self.name}{arg})'
            case TritonOpStyle.CALL:
                return f'{self.name}({", ".join(args)})'


UnaryOpTable: TypeAlias = dict[type[Expr], list[TritonOp]]
BinaryOpTable: TypeAlias = dict[type[Expr], list[TritonOp]]
TernaryOpTable: TypeAlias = dict[type[Expr], list[TritonOp]]


@dataclasses.dataclass
class ScalarOpTable:
    """Per-op-kind tables of supported Triton signatures."""
    unary: UnaryOpTable
    binary: BinaryOpTable
    ternary: TernaryOpTable


# ---------------------------------------------------------------------
# Native context inventory.

_FP_SHAPES = ((5, 16), (8, 32), (11, 64))
"""fp16 / fp32 / fp64 as ``(es, nbits)``.  No bf16: see :mod:`.types`."""

_DIV_SHAPES = ((8, 32), (11, 64))
"""Where a correctly-rounded divide exists.  fp16 is absent -- Triton computes
``fp16 / fp16`` in fp32."""


def _fp_ctxs(shapes=_FP_SHAPES) -> list[Context]:
    """The float contexts this backend dispatches on: round-to-nearest-even
    only, because Triton computes in no other mode.  A *cast* has one more;
    see :func:`directed_cast`."""
    return [IEEEContext(es, nbits, RM.RNE) for (es, nbits) in shapes]


_DIRECTED = {RM.RTZ: 'rz', RM.RTN: 'rd', RM.RTP: 'ru'}
"""libdevice's suffix for each directed rounding."""

_TO_F32 = {
    TritonScalar.F64: 'double2float',
    TritonScalar.S32: 'int2float',
    TritonScalar.U32: 'uint2float',
    TritonScalar.S64: 'll2float',
    TritonScalar.U64: 'ull2float',
}
"""libdevice's directed conversions into `f32`; any narrower source is exact.
`tl.cast` has no directed mode from these: the NVIDIA backend rejects one."""


def _plain(ctx: Context) -> IEEEContext | None:
    """*ctx*, where it is an IEEE format and nothing more."""
    if isinstance(ctx, IEEEContext) and ctx == IEEEContext(ctx.es, ctx.nbits, ctx.rm):
        return ctx
    return None


def castable(ctx: Context) -> bool:
    """Whether a round into *ctx* is a directed conversion from any source:
    one into `f32`, or truncation onto a format with `f32`'s exponents."""
    c = _plain(ctx)
    return c is not None and c.es == 8 and (
        (c.nbits == 32 and c.rm in _DIRECTED) or (c.nbits < 32 and c.rm is RM.RTZ))


def directed_cast(ctx: Context, src: TritonScalar) -> tuple[str, int] | None:
    """The cast from *src* that **is** *ctx*'s round: a format string over
    its argument, and how many low bits of the `f32` result then clear, since
    truncating toward zero composes.  ``None`` where there is none."""
    c = _plain(ctx)
    if c is not None and castable(c):
        spell = f'libdevice.{_TO_F32[src]}_{_DIRECTED[c.rm]}({{}})' \
            if src in _TO_F32 else '{}.to(tl.float32)'
        return spell, 32 - c.nbits
    if c is not None and (c.es, c.nbits) == (5, 16) and c.rm is RM.RTZ \
            and src is TritonScalar.F32:
        return '{}.to(tl.float16, fp_downcast_rounding="rtz")', 0
    return None


def _int_ctxs() -> list[Context]:
    return [
        SINT8, SINT16, SINT32, SINT64,
        UINT8, UINT16, UINT32, UINT64,
        INTEGER,
    ]


def _all_arith_ctxs() -> list[Context]:
    return _fp_ctxs() + _int_ctxs()


_NATIVE_CTXS = frozenset(_all_arith_ctxs())


def is_native_ctx(ctx: Context) -> bool:
    """Is a cast into *ctx*'s storage the same operation as its ``round``?

    **Deliberately a predicate on the context, not on ``(op, context)``**, even
    though this target's op coverage is not uniform over a context's operations
    -- ``Add`` at FP16 is in the table and ``Div`` at FP16 is not.

    The cpp backend can read this one predicate two ways, because ``<cmath>``
    is uniform over a context.  Here the two readings come apart, and the
    cast reading is the one that must stay:

    - As *"is a cast this context's round?"* -- which is what
      ``_require_cast_is_round`` and ``unfold_round``'s ``Round``/``Cast``
      classification ask -- the answer for FP16/FP32/FP64 under RNE is yes.
      ``x.to(tl.float16)`` **is** FP16's round-to-nearest-even.
    - As *"does the op table dispatch on this context?"* the answer is
      per-operation.

    Answering the cast question keeps a native `fp.round` to FP16 as a cast.
    Answering ``False`` instead would send it through ``unfold_round``'s
    integer lowering, which is sound but absurd for an operation the hardware
    performs directly.

    The cost is confined to a diagnostic: an operation the table lacks under an
    otherwise-native context refuses with "no matching signature" and without
    the advice to try ``unfold=DOUBLE_ROUND``.  That is a worse message, not a
    worse outcome, and the message is accurate -- no amount of double-rounding
    recovers an operation the target cannot perform at all.
    """
    return ctx in _NATIVE_CTXS


def _ty_of(ctx: Context) -> TritonScalar:
    return choose_storage_scalar(ctx.format())


# ---------------------------------------------------------------------
# Per-arity table builders.

def _same(name: str, ctxs: list[Context], arity: int,
          style: TritonOpStyle = TritonOpStyle.CALL) -> list[TritonOp]:
    """Same-context signatures for one operation over *ctxs*."""
    return [
        TritonOp(name, (_ty_of(c),) * arity, c, style=style) for c in ctxs
    ]


def _make_unary_table() -> UnaryOpTable:
    fp = _fp_ctxs()
    same = _all_arith_ctxs()
    return {
        Neg: _same('-', same, 1, TritonOpStyle.PREFIX),
        Abs: _same('tl.abs', same, 1),
        # correctly rounded; `tl.sqrt` is the fast variant
        Sqrt: _same('tl.sqrt_rn', _fp_ctxs(_DIV_SHAPES), 1),
        Ceil: _same('tl.ceil', fp, 1),
        Floor: _same('tl.floor', fp, 1),
    }


def _make_binary_table() -> BinaryOpTable:
    same = _all_arith_ctxs()
    table: BinaryOpTable = {
        Add: _same('+', same, 2, TritonOpStyle.INFIX),
        Sub: _same('-', same, 2, TritonOpStyle.INFIX),
        Mul: _same('*', same, 2, TritonOpStyle.INFIX),
    }
    # float division only, and only where a correctly-rounded one exists.
    # integer `//` floors where FPy truncates, so it is absent entirely.
    table[Div] = _same('tl.div_rn', _fp_ctxs(_DIV_SHAPES), 2)
    return table


def _make_ternary_table() -> TernaryOpTable:
    return {Fma: _same('tl.fma', _fp_ctxs(), 3)}


@cache
def make_op_table() -> ScalarOpTable:
    """The triton backend's :class:`ScalarOpTable`.

    Cached: it takes no arguments, every entry derives from the constants
    above, and building it is not cheap -- each signature's storage goes
    through ``AbstractFormat.from_format``.  Callers only read it.
    """
    return ScalarOpTable(
        unary=_make_unary_table(),
        binary=_make_binary_table(),
        ternary=_make_ternary_table(),
    )
