"""
Phase 5a tests for the cpp2 emitter — operation type dispatch.

The emitter routes every primitive op through :class:`ScalarOpTable`.
Direct signature matches emit unchanged; cast-to-result falls back
to a same-type signature, inserting ``static_cast`` only when the
implicit C++ widening would actually drop precision.
"""

import fpy2 as fp
import pytest

from fpy2.backend.cpp2 import Cpp2Compiler, Cpp2CompileError
from fpy2.backend.cpp2.ops import (
    BinaryCppOp,
    UnaryCppOp,
    make_op_table,
)
from fpy2.backend.cpp2.types import CppScalar
from fpy2.types import RealType


class TestOpTableShape:
    """``make_op_table`` returns a table covering the ops cpp2 emits."""

    def test_binary_table_has_arith(self):
        t = make_op_table()
        from fpy2.ast.fpyast import Add, Sub, Mul, Div
        for op in (Add, Sub, Mul, Div):
            assert op in t.binary
            # F64 self-application must be present (the common case).
            assert any(
                sig.matches(CppScalar.F64, CppScalar.F64, CppScalar.F64)
                for sig in t.binary[op]
            )

    def test_unary_table_has_neg_abs(self):
        t = make_op_table()
        from fpy2.ast.fpyast import Neg, Abs
        for op in (Neg, Abs):
            assert op in t.unary
            assert any(
                sig.matches(CppScalar.F64, CppScalar.F64)
                for sig in t.unary[op]
            )

    def test_abs_uses_fabs_for_floats(self):
        t = make_op_table()
        from fpy2.ast.fpyast import Abs
        sigs = [s for s in t.unary[Abs] if s.arg == CppScalar.F64]
        assert len(sigs) == 1
        assert sigs[0].name == 'std::fabs'
        assert sigs[0].format('x') == 'std::fabs(x)'

    def test_abs_uses_std_abs_for_ints(self):
        t = make_op_table()
        from fpy2.ast.fpyast import Abs
        sigs = [s for s in t.unary[Abs] if s.arg == CppScalar.S32]
        assert len(sigs) == 1
        assert sigs[0].name == 'std::abs'


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
