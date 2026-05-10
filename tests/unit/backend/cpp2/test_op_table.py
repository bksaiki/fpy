"""
Phase 5a tests for the cpp2 emitter — operation type dispatch.

Each primitive op is parameterized by *input contexts* and an
*output context*.  Dispatch picks a signature whose ``out_ctx``
matches the active rounding context, with operand formats ⊆ each
``in_ctx.format()``.  When the operand format isn't admitted by any
sig directly, the all-active-context signature is used and operands
are explicit-cast to the active context's storage.  All conversions
go through ``static_cast`` (no implicit promotion).
"""

import fpy2 as fp
import pytest

from fpy2.backend.cpp2 import Cpp2Compiler, Cpp2CompileError
from fpy2.backend.cpp2.ops import (
    BinaryCppOp,
    UnaryCppOp,
    make_op_table,
)
from fpy2.types import RealType


class TestOpTableShape:
    """``make_op_table`` returns a table covering the ops cpp2 emits,
    keyed by per-op-kind context signatures."""

    def test_binary_table_has_arith(self):
        t = make_op_table()
        from fpy2.ast.fpyast import Add, Sub, Mul, Div
        fp64_fmt = fp.FP64.format()
        for op in (Add, Sub, Mul, Div):
            assert op in t.binary
            # FP64 self-application (RNE) must be present — the
            # common-case signature for ``with FP64:`` blocks.
            assert any(
                sig.in1_fmt == fp64_fmt
                and sig.in2_fmt == fp64_fmt
                and sig.out_ctx == fp.FP64
                for sig in t.binary[op]
            )

    def test_unary_table_has_neg_abs(self):
        t = make_op_table()
        from fpy2.ast.fpyast import Neg, Abs
        fp64_fmt = fp.FP64.format()
        for op in (Neg, Abs):
            assert op in t.unary
            assert any(
                sig.arg_fmt == fp64_fmt and sig.out_ctx == fp.FP64
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
            ctx_fmt = ctx.format()
            assert any(
                sig.in1_fmt == ctx_fmt and sig.out_ctx == ctx
                for sig in t.binary[Add]
            )


class TestDispatchDirect:
    """Direct same-type matches emit without a cast."""

    def test_double_add(self):
        @fp.fpy
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            with fp.FP64:
                return x + y

        out = Cpp2Compiler().compile(
            f, ctx=fp.FP64,
            arg_types=[RealType(fp.FP64), RealType(fp.FP64)],
        )
        assert 'return (x + y);' in out
        assert 'static_cast' not in out


class TestDispatchCastFallback:
    """Cast-to-result fires only when the implicit conversion is
    lossy.  Lossless widenings (e.g., ``U8 → F64``) stay implicit."""

    def test_int64_minus_literal_under_fp64_casts_both(self):
        """``len(xs) - 1`` with result F64.  Every conversion goes
        through an explicit ``static_cast`` (no implicit promotion)
        — both the ``int64_t`` operand and the ``U8`` literal cast
        to ``double``."""

        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                n = len(xs)
                return xs[n - 1]

        from fpy2.types import ListType
        out = Cpp2Compiler().compile(
            f, ctx=fp.FP64,
            arg_types=[ListType(RealType(fp.FP64))],
        )
        assert (
            '(static_cast<double>(n) - static_cast<double>(1))'
        ) in out

    def test_int_int_under_fp_casts_both(self):
        """Two int64 operands under FP64 both need explicit casts —
        neither fits losslessly in ``double``."""

        @fp.fpy
        def f() -> fp.Real:
            with fp.FP64:
                return sum([i * i for i in range(5)])

        out = Cpp2Compiler().compile(f, ctx=fp.FP64, arg_types=[])
        assert (
            '(static_cast<double>(i) * static_cast<double>(i))'
        ) in out

    def test_float_double_widening_casts_explicitly(self):
        """``F32`` fits losslessly in ``F64`` but we still emit the
        cast — the policy is "every conversion is explicit", not
        "every cast carries a precision warning"."""

        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP32:
                a = x + 1
            with fp.FP64:
                return a + x

        out = Cpp2Compiler().compile(
            f, ctx=fp.FP64,
            arg_types=[RealType(fp.FP64)],
        )
        # ``a`` is float, ``x`` is double, result is double — explicit
        # cast on the narrower operand even though widening to
        # double is lossless.
        assert 'return (static_cast<double>(a) + x);' in out
