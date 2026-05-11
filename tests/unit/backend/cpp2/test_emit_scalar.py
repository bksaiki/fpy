"""
Phase 2 tests for the cpp2 emitter — scalar arithmetic vertical slice.

Each test compiles a small FPy function and asserts the exact emitted
C++ source string.  This pins both the structure and the formatting,
making regressions easy to spot.

These tests don't compile the emitted C++ — that's a Phase 6 concern.
For now we just check the source is what we expect.
"""

import fpy2 as fp
import pytest

from fpy2.backend.cpp2 import Cpp2Compiler, Cpp2CompileError
from fpy2.types import RealType


@pytest.fixture
def cc():
    return Cpp2Compiler()


def _compile(cc: Cpp2Compiler, func, *, arg_ctx=None) -> str:
    """Helper: monomorphize args + body to FP64 unless told otherwise."""
    arg_ctx = arg_ctx or fp.FP64
    arg_types = [RealType(arg_ctx) for _ in func.args]
    return cc.compile(func, ctx=arg_ctx, arg_types=arg_types)


class TestScalarSlice:
    """Phase 2 — scalar arithmetic only."""

    def test_simple_add(self, cc):
        @fp.fpy
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            with fp.FP64:
                return x + y

        out = _compile(cc, f)
        assert out == (
            'double f(double x, double y) {\n'
            '    return (x + y);\n'
            '}'
        )

    def test_all_four_arith_ops(self, cc):
        @fp.fpy
        def f(a: fp.Real, b: fp.Real) -> fp.Real:
            with fp.FP64:
                t1 = a + b
                t2 = a - b
                t3 = a * b
                t4 = a / b
                return t1 + t2 + t3 + t4

        out = _compile(cc, f)
        # Each ``tN`` has a single writer, so the type is folded into
        # the assign rather than being hoisted at the function top.
        assert 'double t1 = (a + b);' in out
        assert 'double t2 = (a - b);' in out
        assert 'double t3 = (a * b);' in out
        assert 'double t4 = (a / b);' in out
        assert out.startswith('double f(double a, double b) {')
        assert out.rstrip().endswith('}')

    def test_neg_and_abs(self, cc):
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                a = -x
                return abs(a)

        out = _compile(cc, f)
        assert 'double a = (-x);' in out
        assert 'return std::fabs(a);' in out

    def test_fp32_args(self, cc):
        @fp.fpy
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            with fp.FP32:
                return x + y

        out = cc.compile(
            f, ctx=fp.FP32, arg_types=[RealType(fp.FP32), RealType(fp.FP32)]
        )
        assert out.startswith('float f(float x, float y) {')

    def test_unsupported_node_kind_errors(self, cc):
        """Anything outside the supported subset raises a clear
        Cpp2CompileError pointing at the node kind."""

        @fp.fpy
        def f() -> fp.Real:
            with fp.FP64:
                return fp.const_pi()

        with pytest.raises(Cpp2CompileError, match='does not handle NullaryOp'):
            _compile(cc, f)


class TestAssert:
    """``assert`` statements lower to ``<cassert>`` ``assert(...)``.
    With a message, the standard ``cond && \"text\"`` idiom is used so
    the message shows up in the failure output."""

    def test_assert_no_message(self, cc):
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                assert x > 0
                return x

        out = _compile(cc, f)
        assert 'assert((x > static_cast<double>(0)));' in out

    def test_assert_with_message(self, cc):
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                assert x > 0, 'x must be positive'
                return x

        out = _compile(cc, f)
        # ``stmt.msg.format()`` round-trips the AST literal, so the
        # quoted form is what ends up inside the C string.
        assert (
            "assert((x > static_cast<double>(0)) "
            "&& \"fpy assert: 'x must be positive'\");"
        ) in out

    def test_assert_message_escapes_quotes(self, cc):
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                assert x > 0, 'has \\ backslash'
                return x

        out = _compile(cc, f)
        # The Python literal escapes ``\`` as ``\\``; that doubled
        # backslash in the formatted text must in turn be C-escaped
        # to four backslashes.
        assert '&& "fpy assert: \'has \\\\\\\\ backslash\'"' in out


class TestIfExpr:
    """``cond ? ift : iff`` lowers to a C++ ternary.  When the two
    branches have different storage types, both are cast (losslessly)
    into the IfExpr's unified type."""

    def test_same_type_branches(self, cc):
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                return x if x > 0 else -x

        out = _compile(cc, f)
        assert (
            'return ((x > static_cast<double>(0)) ? x : (-x));'
        ) in out

    def test_branches_widen_to_unified(self, cc):
        """One branch is ``F32`` (arg), the other is ``F64`` (arg) —
        the IfExpr's storage is the wider type, and the narrower
        branch widens losslessly via ``static_cast``."""

        @fp.fpy
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            with fp.FP64:
                return x if y > 0 else y

        out = cc.compile(
            f, ctx=fp.FP64,
            arg_types=[RealType(fp.FP32), RealType(fp.FP64)],
        )
        # The narrow branch (F32 ``x``) widens to ``double``; the
        # wide branch (``y``) stays as-is.
        assert '? static_cast<double>(x) : y' in out
