"""
Phase 2 tests for the cpp emitter — scalar arithmetic vertical slice.

Each test compiles a small FPy function and asserts the exact emitted
C++ source string.  This pins both the structure and the formatting,
making regressions easy to spot.

These tests don't compile the emitted C++ — that's a Phase 6 concern.
For now we just check the source is what we expect.
"""

import fpy2 as fp
import pytest

from fpy2.backend.cpp import CppCompiler, CppCompileError
from fpy2.types import RealType


@pytest.fixture
def cc():
    # These tests assert specific bare-emitter output; opt out of
    # optimization so transforms like ``RoundElim`` don't reshape
    # the strings.
    return CppCompiler(optimize=False)


def _compile(cc: CppCompiler, func, *, arg_ctx=None) -> str:
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
        CppCompileError pointing at the node kind."""

        @fp.fpy
        def f() -> fp.Real:
            with fp.INTEGER:
                return fp.inf()

        with pytest.raises(
            CppCompileError, match='ConstInf is not representable in integer storage'
        ):
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


class TestNegativeZero:
    """``-0.0`` is a distinct value, and an integer storage cannot hold it.

    ``format_infer`` bounds a literal by the singleton set of its exact value,
    but a ``Fraction`` has no signed zero -- so ``-0.0`` and ``+0.0`` both give
    ``SetFormat({0})``, and the narrowest type containing that is ``uint8_t``.
    A negative-zero literal therefore reports the narrowest *float* format
    instead (``format_infer._literal_bound``).
    """

    def test_a_returned_negative_zero_keeps_its_sign(self, cc):
        """Regression: this returned `+0.0`.  The sign is observable --
        `x / -0.0` is `-inf` where `x / 0.0` is `+inf`."""
        @fp.fpy
        def f() -> fp.Real:
            return -0.0

        out = cc.compile(f, ctx=fp.FP64, arg_types=[])
        assert 'uint8_t' not in out, out
        assert 'float f()' in out or 'double f()' in out, out
        # The interpreter is the reference, and it keeps the sign.
        import math
        got = float(f(ctx=fp.FP64))
        assert got == 0.0 and math.copysign(1.0, got) < 0, got

    def test_a_negative_zero_in_a_list_literal_compiles(self, cc):
        """Regression: `std::vector<uint8_t>{-0.0}` is a hard `-Wnarrowing`
        error, so this did not compile at all."""
        @fp.fpy
        def f() -> list[fp.Real]:
            with fp.FP64:
                return [-0.0, -0.0]

        out = cc.compile(f, ctx=fp.FP64, arg_types=[])
        assert 'uint8_t' not in out, out

    def test_a_positive_zero_still_narrows(self, cc):
        """The guard is for the *negative* literal only: `+0.0` is exactly the
        integer 0, so it keeps the narrow storage and nothing regresses."""
        @fp.fpy
        def f() -> fp.Real:
            return 0.0

        assert 'uint8_t' in cc.compile(f, ctx=fp.FP64, arg_types=[])
