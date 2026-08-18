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
from fpy2.backend.cpp.target import make_op_table
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
                sig.in_tys == (CppScalar.F64, CppScalar.F64)
                and sig.out_ctx == fp.FP64
                for sig in t.binary[op]
            )

    def test_unary_table_has_neg_abs(self):
        t = make_op_table()
        from fpy2.ast.fpyast import Neg, Abs
        for op in (Neg, Abs):
            assert op in t.unary
            assert any(
                sig.in_tys == (CppScalar.F64,) and sig.out_ctx == fp.FP64
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
                sig.in_tys[0] == CppScalar.F64 and sig.out_ctx == ctx
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


class TestLossyCastAdvice:
    """A refused conversion names a fix that works.  An integer wider than the
    target float's significand is the case where the usual advice -- widen the
    context -- has no answer, since no float rung holds one."""

    @staticmethod
    def _scale(int_ctx):
        @fp.fpy
        def f(n: fp.Real, x: fp.Real) -> fp.Real:
            with fp.FP64:
                return (2 ** n) * x

        return lambda: CppCompiler().compile(
            f, ctx=fp.FP64,
            arg_types=[RealType(int_ctx), RealType(fp.FP64)],
        )

    @pytest.mark.parametrize('int_ctx', [fp.SINT64, fp.INTEGER])
    def test_a_wide_integer_exponent_names_the_bit_limit(self, int_ctx):
        with pytest.raises(CppCompileError) as exc:
            self._scale(int_ctx)()
        msg = str(exc.value)
        assert 'holds integers exactly only up to 53 bits' in msg
        assert 'narrower integer context' in msg
        # the generic advice would send the user to widen the *active* context
        assert 'format contains the operand' not in msg

    @pytest.mark.parametrize('int_ctx', [fp.SINT8, fp.SINT16, fp.SINT32])
    def test_a_narrow_integer_exponent_needs_no_advice(self, int_ctx):
        assert self._scale(int_ctx)()

    def test_the_suggested_explicit_round_compiles(self):
        """One half of the advice: accept the rounding."""

        @fp.fpy
        def f(n: fp.Real, x: fp.Real) -> fp.Real:
            with fp.FP64:
                return (2 ** fp.round(n)) * x

        assert CppCompiler().compile(
            f, ctx=fp.FP64,
            arg_types=[RealType(fp.SINT64), RealType(fp.FP64)],
        )

    def test_the_suggested_narrower_context_compiles(self):
        """The other half: bind the operand where it converts exactly."""

        @fp.fpy
        def f(n: fp.Real, x: fp.Real) -> fp.Real:
            with fp.FP64:
                with fp.SINT32:
                    m = fp.round(n)
                return (2 ** m) * x

        assert CppCompiler().compile(
            f, ctx=fp.FP64,
            arg_types=[RealType(fp.SINT64), RealType(fp.FP64)],
        )

    def test_narrowing_a_float_keeps_the_context_advice(self):
        """Not an integer source, so the bit limit says nothing."""

        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP32:
                return x + 1

        with pytest.raises(CppCompileError) as exc:
            CppCompiler().compile(
                f, ctx=fp.FP32, arg_types=[RealType(fp.FP64)],
            )
        msg = str(exc.value)
        assert 'format contains the operand' in msg
        assert 'bits' not in msg

    def test_narrowing_an_integer_keeps_the_context_advice(self):
        """Widening the active context *is* the fix here, so the bit limit --
        which only a float target has -- must not displace it."""

        @fp.fpy
        def f(n: fp.Real) -> fp.Real:
            with fp.SINT32:
                return n + 1

        with pytest.raises(CppCompileError) as exc:
            CppCompiler().compile(
                f, ctx=fp.SINT32, arg_types=[RealType(fp.SINT64)],
            )
        assert 'format contains the operand' in str(exc.value)
