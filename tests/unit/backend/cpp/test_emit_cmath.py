"""
Phase 5d tests for the cpp emitter — algebraic / transcendental
ops dispatched through the op table to ``<cmath>``.

Coverage check rather than per-op exhaustive: each category (unary,
binary, ternary) gets a representative test that pins both the
function-name string and the dispatch path.  Detailed
``static_cast`` behaviour is already covered by ``test_op_table.py``.
"""

import fpy2 as fp
import pytest

from fpy2.backend.cpp import CppCompiler
from fpy2.types import RealType


class TestUnaryCmath:
    """Unary FP-only ``<cmath>`` functions."""

    def test_sqrt_sin_exp(self):
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                return fp.sqrt(x) + fp.sin(x) + fp.exp(x)

        out = CppCompiler().compile(
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

        out = CppCompiler().compile(
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

        out = CppCompiler().compile(
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

        out = CppCompiler().compile(
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

        out = CppCompiler().compile(
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

        out = CppCompiler().compile(
            f, ctx=fp.FP64,
            arg_types=[RealType(fp.FP64)] * 3,
        )
        assert 'return std::fma(x, y, z);' in out

    def test_fma_fp32(self):
        @fp.fpy
        def f(x: fp.Real, y: fp.Real, z: fp.Real) -> fp.Real:
            with fp.FP32:
                return fp.fma(x, y, z)

        out = CppCompiler().compile(
            f, ctx=fp.FP32,
            arg_types=[RealType(fp.FP32)] * 3,
        )
        assert 'float f(float x, float y, float z)' in out
        assert 'std::fma(x, y, z)' in out


class TestCmathTableShape:
    """Quick check that the table actually carries the new ops."""

    def test_unary_table_has_transcendentals(self):
        from fpy2.ast.fpyast import Sqrt, Sin, Exp, Log, Erf, Cbrt
        from fpy2.backend.cpp.target import make_op_table
        t = make_op_table()
        for op in (Sqrt, Sin, Exp, Log, Erf, Cbrt):
            assert op in t.unary
            assert any(s.out_ctx == fp.FP64 for s in t.unary[op])

    def test_binary_table_has_pow_etc(self):
        from fpy2.ast.fpyast import Pow, Atan2, Hypot, Copysign
        from fpy2.backend.cpp.target import make_op_table
        t = make_op_table()
        for op in (Pow, Atan2, Hypot, Copysign):
            assert op in t.binary
            assert any(s.out_ctx == fp.FP64 for s in t.binary[op])

    def test_ternary_table_has_fma(self):
        from fpy2.ast.fpyast import Fma
        from fpy2.backend.cpp.target import make_op_table
        t = make_op_table()
        assert Fma in t.ternary
        assert any(s.out_ctx == fp.FP64 for s in t.ternary[Fma])


class TestNullaryConstants:
    """Nullary constants.  ``inf``/``nan`` get a ``<limits>`` spelling; the
    rest are evaluated under the active context and emitted as literals,
    since C++11 has no constant to name."""

    def test_infinity_and_nan(self):
        @fp.fpy
        def f() -> fp.Real:
            with fp.FP64:
                return fp.inf()

        @fp.fpy
        def g() -> fp.Real:
            with fp.FP64:
                return fp.nan()

        out_f = CppCompiler().compile(f, ctx=fp.FP64, arg_types=[])
        out_g = CppCompiler().compile(g, ctx=fp.FP64, arg_types=[])
        assert 'std::numeric_limits<double>::infinity()' in out_f
        assert 'std::numeric_limits<double>::quiet_NaN()' in out_g

    def test_infinity_follows_storage_width(self):
        @fp.fpy
        def f() -> fp.Real:
            with fp.FP32:
                return fp.inf()

        out = CppCompiler().compile(f, ctx=fp.FP32, arg_types=[])
        assert 'std::numeric_limits<float>::infinity()' in out

    def test_math_constants_fold_to_literals(self):
        @fp.fpy
        def f() -> fp.Real:
            with fp.FP64:
                return fp.const_pi()

        out = CppCompiler().compile(f, ctx=fp.FP64, arg_types=[])
        assert 'return 3.141592653589793;' in out
        assert 'const_pi' not in out

    def test_constant_rounds_under_its_own_context(self):
        """The literal is the constant rounded under the *active* context,
        not the widest one."""
        @fp.fpy
        def f() -> fp.Real:
            with fp.FP32:
                return fp.const_pi()

        out = CppCompiler().compile(f, ctx=fp.FP32, arg_types=[])
        assert 'return 3.1415927410125732;' in out

    def test_constant_under_integer_context(self):
        @fp.fpy
        def f() -> fp.Real:
            with fp.INTEGER:
                return fp.const_pi()

        out = CppCompiler().compile(f, ctx=fp.INTEGER, arg_types=[])
        assert 'return 3;' in out


class TestNullaryConstantsUnderReal:
    """A ``REAL`` scope has no storable format of its own, so a special value
    has to carry its own."""

    def test_infinity_compiles_under_real(self):
        @fp.fpy(ctx=fp.REAL)
        def f() -> fp.Real:
            return fp.inf()

        out = CppCompiler().compile(f, ctx=fp.REAL, arg_types=[])
        assert '::infinity()' in out

    def test_nan_compiles_under_real(self):
        @fp.fpy(ctx=fp.REAL)
        def f() -> fp.Real:
            return fp.nan()

        out = CppCompiler().compile(f, ctx=fp.REAL, arg_types=[])
        assert '::quiet_NaN()' in out

    def test_an_irrational_constant_under_real_is_refused(self):
        """No finite C++ type holds pi exactly, and REAL does not round."""
        from fpy2.backend.cpp.compiler import CppCompileError

        @fp.fpy(ctx=fp.REAL)
        def f() -> fp.Real:
            return fp.const_pi()

        with pytest.raises(CppCompileError, match='unconstrained real'):
            CppCompiler().compile(f, ctx=fp.REAL, arg_types=[])
