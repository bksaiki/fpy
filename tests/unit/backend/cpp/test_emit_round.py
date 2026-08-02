"""
Phase 5c tests for the cpp emitter — ``Round`` and ``Cast``.

- ``Round(arg)`` is a plain ``static_cast<target>(arg)``; the cast's
  rounding mode comes from Phase 5b's ``fesetround`` boundary.
- ``Cast(arg)`` (the node ``fp.cast`` and ``fp.round_exact`` both
  parse to) is the same cast plus a runtime assertion that the
  round-trip preserves the value; FP operands get a NaN-aware
  equality check.  Casting to the same type is a guaranteed no-op,
  so no cast and no assertion are emitted.
"""

import pytest

import fpy2 as fp

from fpy2.backend.cpp import CppCompiler
from fpy2.backend.cpp.compiler import CppCompileError
from fpy2.types import RealType


class TestRound:
    """Plain ``static_cast`` — ``fesetround`` from the surrounding
    ``with`` controls the rounding mode."""

    def test_round_fp64_to_fp32(self):
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP32:
                return fp.round(x)

        out = CppCompiler().compile(
            f, ctx=fp.FP32,
            arg_types=[RealType(fp.FP64)],
        )
        assert 'float f(double x)' in out
        assert 'return static_cast<float>(x);' in out

    def test_round_same_type_is_noop(self):
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                return fp.round(x)

        out = CppCompiler().compile(
            f, ctx=fp.FP64,
            arg_types=[RealType(fp.FP64)],
        )
        # Same-type round → no static_cast.
        assert 'static_cast' not in out
        assert 'return x;' in out


class TestInexactLiterals:
    """An FPy literal is an exact rational; C++ has no such thing.

    So the only inexact constant that can be emitted is one the program
    rounded, and ``fp.round`` is where the rounding happens — at compile time,
    under the context that asked for it.
    """

    def test_a_literal_the_program_never_rounded_is_refused(self):
        """``num / denom`` would be an *operation* where FPy has a constant:
        it rounds under whatever mode is set rather than the program's, and
        ``-O2`` folds it to nearest regardless."""
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                return x * fp.rational(1, 3)

        with pytest.raises(CppCompileError, match='unsupported literal'):
            CppCompiler().compile(
                f, ctx=fp.FP64, arg_types=[RealType(fp.FP64)],
            )

    def test_round_of_an_inexact_literal_is_folded(self):
        @fp.fpy
        def f() -> fp.Real:
            with fp.FP64:
                return fp.round(3.14159265359)

        out = CppCompiler().compile(f, ctx=fp.FP64, arg_types=[])
        assert 'return 3.14159265359;' in out
        assert 'static_cast' not in out

    def test_the_fold_uses_the_contexts_rounding_mode(self):
        """The reason to fold rather than cast: ``static_cast<double>(1/3)``
        gets the mode that happens to be set, and at ``-O2`` not even that."""
        toward_zero = fp.IEEEContext(11, 64, fp.RM.RTZ)

        @fp.fpy
        def f() -> fp.Real:
            with toward_zero:
                return fp.round(0.1)

        out = CppCompiler().compile(f, ctx=fp.FP64, arg_types=[])
        # 0x1.9999999999999p-4, one ulp below the nearest double to 0.1.
        assert 'return 0.09999999999999999;' in out

    def test_a_narrower_target_still_gets_its_cast(self):
        """Rounded at FP32, printed as the double that holds it exactly, then
        cast — a conversion no mode can change."""
        @fp.fpy
        def f() -> fp.Real:
            with fp.FP32:
                return fp.round(0.1)

        out = CppCompiler().compile(f, ctx=fp.FP32, arg_types=[])
        assert 'static_cast<float>(0.10000000149011612)' in out


class TestRoundExact:
    """``fp.round_exact`` (which parses to a ``Cast`` node):
    round + assert that the cast was lossless."""

    def test_fp_round_exact_uses_nan_aware_compare(self):
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP32:
                return fp.round_exact(x)

        out = CppCompiler().compile(
            f, ctx=fp.FP32,
            arg_types=[RealType(fp.FP64)],
        )
        # Cast bound to a temp.
        assert 'float _tmp1 = static_cast<float>(x);' in out
        # FP comparison includes the NaN guard.
        assert (
            'assert(x == _tmp1 || '
            '(std::isnan(x) && std::isnan(_tmp1)));'
        ) in out
        assert 'return _tmp1;' in out

    def test_int_round_exact_skips_nan_guard(self):
        """Integer operand pairs don't need ``std::isnan`` — int
        equality already handles every value."""

        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.SINT8:
                return fp.round_exact(x)

        out = CppCompiler().compile(
            f, ctx=fp.SINT8,
            arg_types=[RealType(fp.INTEGER)],
        )
        assert 'int8_t _tmp1 = static_cast<int8_t>(x);' in out
        # No std::isnan call.
        assert 'std::isnan' not in out
        assert 'assert(x == _tmp1);' in out

    def test_round_exact_same_type_is_noop(self):
        """Casting to the same type is guaranteed lossless — no
        cast and no assertion."""

        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                return fp.round_exact(x)

        out = CppCompiler().compile(
            f, ctx=fp.FP64,
            arg_types=[RealType(fp.FP64)],
        )
        assert 'static_cast' not in out
        assert 'assert' not in out
        assert 'return x;' in out


class TestCast:
    """``Cast`` rounds and asserts the result is exact — same
    emission as ``round_exact`` (they parse to the same node)."""

    def test_cast_same_type_emits_identity(self):
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                return fp.cast(x)

        out = CppCompiler().compile(
            f, ctx=fp.FP64,
            arg_types=[RealType(fp.FP64)],
        )
        # Same-type cast is a guaranteed no-op.
        assert 'static_cast' not in out
        assert 'assert' not in out
        assert 'return x;' in out

    def test_cast_cross_type_emits_assert(self):
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP32:
                return fp.cast(x)

        out = CppCompiler().compile(
            f, ctx=fp.FP32,
            arg_types=[RealType(fp.FP64)],
        )
        # Lossy-capable cast → cast bound to a temp + NaN-aware assert.
        assert 'float _tmp1 = static_cast<float>(x);' in out
        assert (
            'assert(x == _tmp1 || '
            '(std::isnan(x) && std::isnan(_tmp1)));'
        ) in out
        assert 'return _tmp1;' in out


class TestRoundElimIntegration:
    """End-to-end checks that the ``optimize=True`` pipeline runs
    :class:`fpy2.transform.RoundElim` and the emitter consumes its
    output correctly.

    The cases here aren't ones that *require* RoundElim to compile —
    they compile under ``optimize=False`` too, just less tightly.
    What the tests pin is the measurable cpp-level *difference*
    between the two modes, which is the contract callers care about
    when they flip the flag."""

    def test_round_exact_of_fitting_constant_collapses(self):
        """``fp.round_exact(1.0)`` under FP32: ``1.0`` is exactly
        FP32-representable, so the round is the identity.  Without
        optimization the emitter still produces the full
        assertion-protected cast (``static_cast<float>(1)`` bound
        to a temp + NaN-aware equality check).  With RoundElim the
        ``Cast`` node collapses to its argument and the
        whole pattern disappears."""

        @fp.fpy
        def f():
            with fp.FP32:
                return fp.round_exact(1.0)

        no_opt = CppCompiler(optimize=False).compile(
            f, ctx=fp.FP64, arg_types=[],
        )
        with_opt = CppCompiler(optimize=True).compile(
            f, ctx=fp.FP64, arg_types=[],
        )

        # Without RoundElim: the round_exact's assertion + cast
        # machinery is present.
        assert 'static_cast<float>' in no_opt
        assert 'assert(' in no_opt

        # With RoundElim: collapsed to a bare ``return 1;`` (no
        # cast, no assertion).
        assert 'static_cast' not in with_opt
        assert 'assert(' not in with_opt
        assert 'return 1;' in with_opt

    def test_arith_fits_scope_hoisted_to_real(self):
        """``(1.0 + 2.0) * 3.0`` under FP64: the unrounded result
        ``SetFormat({9.0})`` fits FP64 exactly.  RoundElim should
        hoist the chain into ``with fp.REAL:`` blocks; the emitter
        then renders the body without ``fesetround`` boundaries
        on the hoisted ops."""

        @fp.fpy
        def f():
            with fp.FP64:
                return (1.0 + 2.0) * 3.0

        with_opt = CppCompiler(optimize=True).compile(
            f, ctx=fp.FP64, arg_types=[],
        )
        # Hoisted: the per-op binds RoundElim emits surface as
        # ``_t...`` temporaries.  At least one shows up.
        assert '_t' in with_opt
        # The final return value is the last hoisted result.
        assert 'return _t' in with_opt
