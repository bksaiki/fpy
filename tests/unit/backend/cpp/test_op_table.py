"""
Phase 5a tests for the cpp emitter — operation type dispatch.

Each primitive op is parameterized by *argument C++ types* and an
*output rounding context*.  Dispatch picks a signature whose
``out_ctx`` matches the active rounding context, with operand C++
storage types equal to each ``in_ty``.  When operand storages
don't match a signature directly, the all-active-context signature
is used and operands are explicit-cast to the active context's
storage.  All conversions go through ``static_cast`` (no implicit
promotion).
"""

import fpy2 as fp
import pytest

from fpy2.backend.cpp import CppCompiler, CppCompileError
from fpy2.backend.cpp.ops import make_op_table
from fpy2.backend.cpp.types import CppScalar
from fpy2.types import RealType


class TestOpTableShape:
    """``make_op_table`` returns a table covering the ops the cpp emitter emits,
    keyed by per-op-kind context signatures."""

    def test_binary_table_has_arith(self):
        t = make_op_table()
        from fpy2.ast.fpyast import Add, Sub, Mul, Div
        for op in (Add, Sub, Mul, Div):
            assert op in t.binary
            # FP64 self-application (RNE) must be present — the
            # common-case signature for ``with FP64:`` blocks.
            assert any(
                sig.in1_ty == CppScalar.F64
                and sig.in2_ty == CppScalar.F64
                and sig.out_ctx == fp.FP64
                for sig in t.binary[op]
            )

    def test_unary_table_has_neg_abs(self):
        t = make_op_table()
        from fpy2.ast.fpyast import Neg, Abs
        for op in (Neg, Abs):
            assert op in t.unary
            assert any(
                sig.arg_ty == CppScalar.F64 and sig.out_ctx == fp.FP64
                for sig in t.unary[op]
            )

    def test_abs_uses_fabs_for_floats(self):
        t = make_op_table()
        from fpy2.ast.fpyast import Abs
        sigs = [s for s in t.unary[Abs] if s.out_ctx == fp.FP64]
        assert len(sigs) == 1
        assert sigs[0].name == 'std::fabs'
        assert sigs[0].format('x') == 'std::fabs(x)'

    def test_abs_uses_std_abs_for_ints(self):
        t = make_op_table()
        from fpy2.ast.fpyast import Abs
        sigs = [s for s in t.unary[Abs] if s.out_ctx == fp.SINT32]
        assert len(sigs) == 1
        assert sigs[0].name == 'std::abs'

    def test_binary_table_has_per_rm_fp_signatures(self):
        """Each FP base gets one signature per supported rounding
        mode — the dispatch matches the active context's RM
        exactly."""
        t = make_op_table()
        from fpy2.ast.fpyast import Add
        rms = [fp.RM.RNE, fp.RM.RTZ, fp.RM.RTP, fp.RM.RTN]
        for rm in rms:
            ctx = fp.IEEEContext(11, 64, rm)
            assert any(
                sig.in1_ty == CppScalar.F64 and sig.out_ctx == ctx
                for sig in t.binary[Add]
            )


class TestDispatchDirect:
    """Direct same-type matches emit without a cast."""

    def test_double_add(self):
        @fp.fpy
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            with fp.FP64:
                return x + y

        out = CppCompiler().compile(
            f, ctx=fp.FP64,
            arg_types=[RealType(fp.FP64), RealType(fp.FP64)],
        )
        assert 'return (x + y);' in out
        assert 'static_cast' not in out


class TestDispatchCastFallback:
    """Cast-to-active fires when operand storage doesn't match the
    signature's input slot.  The cast must be lossless — lossy
    implicit casts are rejected, telling the user to round
    explicitly with ``fp.round(...)``."""

    def test_lossless_widening_casts_implicit(self):
        """``F32`` widens losslessly into ``F64``; the cast emits
        without rejection."""

        @fp.fpy
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            with fp.FP64:
                return x + y

        out = CppCompiler().compile(
            f, ctx=fp.FP64,
            arg_types=[RealType(fp.FP32), RealType(fp.FP64)],
        )
        # ``x`` is float (F32) — fits in double, so cast is emitted.
        assert 'return (static_cast<double>(x) + y);' in out

    def test_lossy_int64_to_double_rejected(self):
        """``len(xs) - 1`` with result F64: ``len(xs)`` is
        ``int64_t``, which doesn't fit losslessly in ``double``.
        The implicit cast is rejected."""

        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                n = len(xs)
                return xs[n - 1]

        from fpy2.types import ListType
        with pytest.raises(
            CppCompileError,
            match='cannot implicitly cast.*int64_t.*to.*double',
        ):
            CppCompiler().compile(
                f, ctx=fp.FP64,
                arg_types=[ListType(RealType(fp.FP64))],
            )

    def test_lossy_fp64_to_fp32_rejected(self):
        """Casting ``double`` into an ``FP32`` context is lossy
        — must be made explicit with ``fp.round(...)``."""

        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP32:
                return x + 1

        with pytest.raises(
            CppCompileError,
            match='cannot implicitly cast.*double.*to.*float',
        ):
            CppCompiler().compile(
                f, ctx=fp.FP32,
                arg_types=[RealType(fp.FP64)],
            )

    def test_explicit_round_makes_lossy_cast_legal(self):
        """Wrapping the wider operand in ``fp.round(...)`` is the
        sanctioned escape hatch — the user is explicitly opting in."""

        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP32:
                return fp.round(x) + 1

        out = CppCompiler().compile(
            f, ctx=fp.FP32,
            arg_types=[RealType(fp.FP64)],
        )
        # ``fp.round(x)`` emits the lossy ``static_cast<float>(x)``.
        assert 'static_cast<float>(x)' in out
