"""
Phase 5c tests for the cpp emitter — ``Round``, ``RoundExact``, ``Cast``.

- ``Round(arg)`` is a plain ``static_cast<target>(arg)``; the cast's
  rounding mode comes from Phase 5b's ``fesetround`` boundary.
- ``RoundExact(arg)`` adds a runtime assertion that the round-trip
  preserves the value; FP operands get a NaN-aware equality check.
- ``Cast(arg)`` is a pure analysis annotation — emitted as the
  identity (no ``static_cast``).
"""

import fpy2 as fp

from fpy2.backend.cpp import CppCompiler
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


class TestRoundExact:
    """Round + assert that the cast was lossless."""

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
        assert 'float __cpp_tmp1 = static_cast<float>(x);' in out
        # FP comparison includes the NaN guard.
        assert (
            'assert(x == __cpp_tmp1 || '
            '(std::isnan(x) && std::isnan(__cpp_tmp1)));'
        ) in out
        assert 'return __cpp_tmp1;' in out

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
        assert 'int8_t __cpp_tmp1 = static_cast<int8_t>(x);' in out
        # No std::isnan call.
        assert 'std::isnan' not in out
        assert 'assert(x == __cpp_tmp1);' in out

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
    """``Cast`` is the identity — no generated code."""

    def test_cast_emits_identity(self):
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                return fp.cast(x)

        out = CppCompiler().compile(
            f, ctx=fp.FP64,
            arg_types=[RealType(fp.FP64)],
        )
        assert 'static_cast' not in out
        assert 'assert' not in out
        assert 'return x;' in out


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
        ``RoundExact`` node collapses to its argument and the
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
