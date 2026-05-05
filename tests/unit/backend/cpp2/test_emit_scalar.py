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
            '}\n'
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
        """Anything outside the Phase-2 slice (here: comparisons) raises
        a clear Cpp2CompileError pointing at the node kind."""

        @fp.fpy
        def f(x: fp.Real, y: fp.Real) -> bool:
            with fp.FP64:
                return x < y

        with pytest.raises(Cpp2CompileError, match='Phase 2 does not handle'):
            _compile(cc, f)
