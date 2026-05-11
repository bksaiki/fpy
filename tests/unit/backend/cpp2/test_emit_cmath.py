"""
Phase 5d tests for the cpp2 emitter — algebraic / transcendental
ops dispatched through the op table to ``<cmath>``.

Coverage check rather than per-op exhaustive: each category (unary,
binary, ternary) gets a representative test that pins both the
function-name string and the dispatch path.  Detailed
``static_cast`` behaviour is already covered by ``test_op_table.py``.
"""

import fpy2 as fp

from fpy2.backend.cpp2 import Cpp2Compiler
from fpy2.types import RealType


class TestUnaryCmath:
    """Unary FP-only ``<cmath>`` functions."""

    def test_sqrt_sin_exp(self):
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                return fp.sqrt(x) + fp.sin(x) + fp.exp(x)

        out = Cpp2Compiler().compile(
            f, ctx=fp.FP64, arg_types=[RealType(fp.FP64)],
        )
        assert 'std::sqrt(x)' in out
        assert 'std::sin(x)' in out
        assert 'std::exp(x)' in out

    def test_fp_rounding_helpers(self):
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                return fp.ceil(x) + fp.floor(x) + fp.trunc(x)

        out = Cpp2Compiler().compile(
            f, ctx=fp.FP64, arg_types=[RealType(fp.FP64)],
        )
        assert 'std::ceil(x)' in out
        assert 'std::floor(x)' in out
        assert 'std::trunc(x)' in out

    def test_fp32_dispatch(self):
        """FP32 contexts get their own signatures — function name
        is the same, operand context differs."""

        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP32:
                return fp.sqrt(x)

        out = Cpp2Compiler().compile(
            f, ctx=fp.FP32, arg_types=[RealType(fp.FP32)],
        )
        assert 'float f(float x)' in out
        assert 'std::sqrt(x)' in out


class TestBinaryCmath:
    """Binary FP-only ``<cmath>`` functions (function-call form)."""

    def test_pow_atan2_hypot(self):
        @fp.fpy
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            with fp.FP64:
                return fp.pow(x, y) + fp.atan2(x, y) + fp.hypot(x, y)

        out = Cpp2Compiler().compile(
            f, ctx=fp.FP64,
            arg_types=[RealType(fp.FP64), RealType(fp.FP64)],
        )
        assert 'std::pow(x, y)' in out
        assert 'std::atan2(x, y)' in out
        assert 'std::hypot(x, y)' in out

    def test_copysign_fmod(self):
        @fp.fpy
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            with fp.FP64:
                return fp.copysign(x, y) + fp.fmod(x, y)

        out = Cpp2Compiler().compile(
            f, ctx=fp.FP64,
            arg_types=[RealType(fp.FP64), RealType(fp.FP64)],
        )
        assert 'std::copysign(x, y)' in out
        assert 'std::fmod(x, y)' in out


class TestTernaryCmath:
    """``Fma`` is the only currently-supported ternary."""

    def test_fma(self):
        @fp.fpy
        def f(x: fp.Real, y: fp.Real, z: fp.Real) -> fp.Real:
            with fp.FP64:
                return fp.fma(x, y, z)

        out = Cpp2Compiler().compile(
            f, ctx=fp.FP64,
            arg_types=[RealType(fp.FP64)] * 3,
        )
        assert 'return std::fma(x, y, z);' in out

    def test_fma_fp32(self):
        @fp.fpy
        def f(x: fp.Real, y: fp.Real, z: fp.Real) -> fp.Real:
            with fp.FP32:
                return fp.fma(x, y, z)

        out = Cpp2Compiler().compile(
            f, ctx=fp.FP32,
            arg_types=[RealType(fp.FP32)] * 3,
        )
        assert 'float f(float x, float y, float z)' in out
        assert 'std::fma(x, y, z)' in out


class TestCmathTableShape:
    """Quick check that the table actually carries the new ops."""

    def test_unary_table_has_transcendentals(self):
        from fpy2.ast.fpyast import Sqrt, Sin, Exp, Log, Erf, Cbrt
        from fpy2.backend.cpp2.ops import make_op_table
        t = make_op_table()
        for op in (Sqrt, Sin, Exp, Log, Erf, Cbrt):
            assert op in t.unary
            assert any(s.out_ctx == fp.FP64 for s in t.unary[op])

    def test_binary_table_has_pow_etc(self):
        from fpy2.ast.fpyast import Pow, Atan2, Hypot, Copysign
        from fpy2.backend.cpp2.ops import make_op_table
        t = make_op_table()
        for op in (Pow, Atan2, Hypot, Copysign):
            assert op in t.binary
            assert any(s.out_ctx == fp.FP64 for s in t.binary[op])

    def test_ternary_table_has_fma(self):
        from fpy2.ast.fpyast import Fma
        from fpy2.backend.cpp2.ops import make_op_table
        t = make_op_table()
        assert Fma in t.ternary
        assert any(s.out_ctx == fp.FP64 for s in t.ternary[Fma])
