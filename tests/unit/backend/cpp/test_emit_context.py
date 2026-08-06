"""
Phase 5b tests for the cpp emitter — context boundaries.

The active rounding context is taken from the
:class:`ContextUseAnalysis` scope at every ``FuncDef`` /
``ContextStmt`` site.  cpp only compiles programs whose contexts
are statically resolvable; symbolic context variables are rejected.

For float contexts the rounding mode must be one of the four
``fesetround``-supported modes (RNE / RTZ / RTP / RTN).  For integer
contexts the rounding mode must be RTZ — C++ integer arithmetic
already truncates toward zero, so no runtime support is needed.

Tests in this module assert specific bare-emitter output and the
rejection-mechanism behavior — both surfaces that optimizing
transforms can legally rewrite around.  Each ``CppCompiler``
construction passes ``optimize=False`` to keep the assertions
stable against transforms like :class:`fpy2.transform.RoundElim`
that would otherwise reshape the generated C++ or sidestep
rejection paths.
"""

import fpy2 as fp
import pytest

from fpy2.backend.cpp import CppCompiler, CppCompileError
from fpy2.types import RealType


_RTZ_64 = fp.IEEEContext(11, 64, fp.RM.RTZ)
_RTP_64 = fp.IEEEContext(11, 64, fp.RM.RTP)
_RNA_64 = fp.IEEEContext(11, 64, fp.RM.RNA)


class TestStaticResolution:
    """Validation gates on context use: a scope must resolve to a
    concrete, supported :class:`Context` *iff* a primitive op
    dispatches under it.  Scopes that hold an exotic context but
    have no uses compile freely."""

    def test_function_with_no_fp_doesnt_need_ctx(self):
        """A bool-returning function has no op uses, so its outer
        scope can be symbolic — no error."""

        @fp.fpy
        def f() -> bool:
            return True

        out = CppCompiler(optimize=False).compile(f)
        assert 'fesetround' not in out

    def test_concrete_with_block_resolves(self):
        @fp.fpy
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            with fp.FP64:
                return x + y

        out = CppCompiler(optimize=False).compile(
            f, ctx=fp.FP64,
            arg_types=[RealType(fp.FP64), RealType(fp.FP64)],
        )
        # FP64 default-RM (RNE) — no fesetround.
        assert 'fesetround' not in out

    def test_function_scope_unused_when_all_ops_nested(self):
        """When every op lives inside an inner ``with``, the
        function-level scope has no uses and isn't validated — so
        the outer (function-level) context can be anything,
        including a context the cpp backend doesn't natively
        support."""

        @fp.fpy(ctx=_RNA_64)              # RNA is unsupported by fesetround,
        def f(x: fp.Real) -> fp.Real:     # but this function never dispatches
            with fp.FP64:                  # under the outer scope — every op
                return x + x               # is inside ``with fp.FP64:``.

        # No error: compilation succeeds.
        out = CppCompiler(optimize=False).compile(f, arg_types=[RealType(fp.FP64)])
        # Not `return (x + x);`: the outer RM is unknown, so the inner ``with``
        # does set the mode, and returning from inside it restores first -- which
        # binds the value to a temp.  See
        # `test_return_inside_a_scope_restores_first`.
        assert '(x + x)' in out

    def test_with_block_scope_unused_compiles(self):
        """A ``with`` block whose body has no ops compiles fine
        even if the context itself is unsupported — there's nothing
        to validate."""

        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            with _RNA_64:                  # unsupported RM …
                n = 0                      # … but only a literal
            with fp.FP64:                  # assign — no op dispatches
                return xs[n]               # under the outer scope.

        from fpy2.types import ListType
        out = CppCompiler(optimize=False).compile(
            f, ctx=fp.FP64,
            arg_types=[ListType(RealType(fp.FP64))],
        )
        assert 'return xs[' in out


class TestDefaultRmIsImplicit:
    """``with FP64:`` (RM=RNE) doesn't emit fesetround when the
    surrounding mode is already RNE."""

    def test_rne_under_rne(self):
        @fp.fpy
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            with fp.FP64:
                return x + y

        out = CppCompiler(optimize=False).compile(
            f, ctx=fp.FP64,
            arg_types=[RealType(fp.FP64), RealType(fp.FP64)],
        )
        assert 'fesetround' not in out

    def test_integer_context_no_fesetround(self):
        """``with INTEGER:`` doesn't emit fesetround — integer
        arithmetic doesn't consult ``fenv``.  Requires
        ``unsafe_cast_int=True`` because ``INTEGER`` is unbounded
        and the cpp backend has no arbitrary-precision integer
        type; this test specifically exercises the opt-in path."""

        @fp.fpy
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            with fp.INTEGER:
                return x + y

        out = CppCompiler(unsafe_cast_int=True).compile(
            f, ctx=fp.INTEGER,
            arg_types=[RealType(fp.INTEGER), RealType(fp.INTEGER)],
        )
        assert 'fesetround' not in out
        assert 'int64_t f(int64_t x, int64_t y)' in out


class TestNonDefaultRmEmitsFesetround:
    """Non-RNE float contexts emit ``fesetround`` only when the
    active mode actually changes.  A concrete function-level
    annotation is the caller's contract: we trust the caller to
    deliver that RM and emit nothing at function entry."""

    def test_function_level_rtz_trusts_caller(self):
        """A concrete function-level RTZ context does *not* emit
        ``fesetround`` at entry — the caller is contractually
        delivering RTZ."""

        @fp.fpy(ctx=_RTZ_64)
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            return x + y

        out = CppCompiler(optimize=False).compile(
            f, arg_types=[RealType(fp.FP64), RealType(fp.FP64)],
        )
        assert 'fesetround' not in out

    def test_return_inside_a_scope_restores_first(self):
        """A ``return`` inside a rounding scope restores *before* returning.

        The restore a scope emits after its body cannot run when the body
        returns, so without this the mode escapes into the caller and silently
        changes arithmetic that has nothing to do with this function.  Pinned
        on the order, which is the whole content of the fix: the value must be
        computed under the scope's mode and handed back under the caller's.

        Executed end-to-end by ``_test_fenv`` in
        ``tests/infra/backend/cpp.py``, which a unit test cannot do -- the
        differential driver compares a kernel's own result, and a leaked mode
        is correct there and wrong everywhere after.
        """

        @fp.fpy(ctx=fp.FP64)
        def f(x: fp.Real) -> fp.Real:
            with _RTZ_64:
                y = x + 1.0
                return y

        out = CppCompiler(optimize=False).compile(
            f, arg_types=[RealType(fp.FP64)],
        )
        lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
        restore = next(
            i for i, ln in enumerate(lines)
            if ln.startswith('std::fesetround(') and 'FE_' not in ln
        )
        ret = next(i for i, ln in enumerate(lines) if ln.startswith('return '))
        assert restore < ret, f'restore must precede the return:\n{out}'

    def test_nested_scopes_restore_to_the_callers_mode(self):
        """Two scopes deep, the restore reaches past both.

        One restore suffices and it comes from the *outermost* scope, which
        saved the mode from before any of them were entered.

        Both scopes need an op of their own: a scope no op dispatches under is
        not emitted at all, so nesting alone would give only one save.
        """

        @fp.fpy(ctx=fp.FP64)
        def f(x: fp.Real) -> fp.Real:
            with _RTZ_64:
                a = x + 1.0
                with _RTP_64:
                    y = a + 1.0
                    return y

        out = CppCompiler(optimize=False).compile(
            f, arg_types=[RealType(fp.FP64)],
        )
        saves = [
            ln.split('=')[0].strip().split()[-1]
            for ln in out.splitlines() if 'std::fegetround()' in ln
        ]
        assert len(saves) == 2, out
        ret = out.index('return ')
        # the restore before the return names the outermost save
        assert f'std::fesetround({saves[0]});' in out[:ret], out

    def test_falling_out_of_a_scope_still_restores_at_the_end(self):
        """The counterweight: the end-of-block restore is load-bearing.

        When execution leaves the scope normally, the statement after it must
        run under the function's mode again -- so the fix for the early-return
        path must not replace the ordinary restore.
        """

        @fp.fpy(ctx=fp.FP64)
        def f(x: fp.Real) -> fp.Real:
            with _RTZ_64:
                y = x + 1.0
            z = y + 1.0
            return z

        out = CppCompiler(optimize=False).compile(
            f, arg_types=[RealType(fp.FP64)],
        )
        lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
        restore = next(
            i for i, ln in enumerate(lines)
            if ln.startswith('std::fesetround(') and 'FE_' not in ln
        )
        z_decl = next(i for i, ln in enumerate(lines) if ln.startswith('double z'))
        assert restore < z_decl, f'`z` must be computed after the restore:\n{out}'

    def test_with_block_changes_rm(self):
        """An inner ``with`` that switches to a different RM emits
        fesetround on entry and restores on exit.  The function-
        level RTZ is trusted (no entry emission)."""

        @fp.fpy(ctx=_RTZ_64)
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            a = x + y
            with fp.FP64:
                b = a + 1
            return a + b

        out = CppCompiler(optimize=False).compile(
            f, arg_types=[RealType(fp.FP64), RealType(fp.FP64)],
        )
        # Inner switches RTZ→RNE — one save / set / restore pair.
        assert out.count('std::fegetround()') == 1
        assert 'std::fesetround(FE_TONEAREST)' in out
        # No fesetround at function entry.
        assert 'std::fesetround(FE_TOWARDZERO)' not in out

    def test_with_block_same_rm_skips_fesetround(self):
        """A nested ``with`` whose RM matches the contracted
        function-level mode emits no fesetround — nothing changes
        and the caller already delivered the right mode."""

        @fp.fpy(ctx=_RTZ_64)
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            with _RTZ_64:
                return x + y

        out = CppCompiler(optimize=False).compile(
            f, arg_types=[RealType(fp.FP64), RealType(fp.FP64)],
        )
        # No fesetround anywhere — caller is contracted to deliver
        # RTZ and the inner block doesn't change that.
        assert 'fesetround' not in out

    def test_symbolic_outer_forces_inner_fesetround(self):
        """When the function-level scope is symbolic (no annotation
        and no compile-time ``ctx`` to monomorphize against), we
        don't know the caller's rounding mode — every concrete
        inner ``with`` must emit ``fesetround`` to recover
        certainty."""

        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            # Outer scope is symbolic — only the inner ``with`` uses
            # a rounding context.
            with _RTZ_64:
                return xs[0] + xs[1]

        from fpy2.types import ListType
        out = CppCompiler(optimize=False).compile(
            f, arg_types=[ListType(RealType(fp.FP64))],
        )
        # Inner RTZ block must emit fesetround — outer is unknown.
        assert 'std::fesetround(FE_TOWARDZERO)' in out
        assert 'std::fegetround()' in out


class TestRejection:
    """Errors when the context isn't statically resolvable, the
    rounding mode isn't supported, or an integer context uses a
    non-RTZ mode."""

    def test_rna_float_rejected(self):
        """RNA isn't one of the four ``fesetround`` modes."""

        @fp.fpy(ctx=_RNA_64)
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            return x + y

        with pytest.raises(
            CppCompileError,
            match='not supported by ``fesetround``',
        ):
            CppCompiler(optimize=False).compile(
                f, arg_types=[RealType(fp.FP64), RealType(fp.FP64)],
            )

    def test_integer_non_rtz_rejected(self):
        """Integer contexts must use RTZ.

        Pinned with ``optimize=False`` — ``RoundElim`` would
        otherwise hoist the integer-add out of the bad-RM scope
        (the unrounded sum of two ints is an int and fits the
        scope, so the round is identity), sidestepping the
        rejection.  The rejection mechanism is what this test
        exercises, so we keep it visible by opting out of
        optimization.

        ``enable_neg_zero=False`` for a similar reason: it
        defaults to ``True``, and a context that holds a signed
        zero has no integer storage on the ladder, so storage
        selection would reject this program before the
        rounding-mode check ever ran."""

        bad_int = fp.MPFixedContext(-1, fp.RM.RNE, enable_neg_zero=False)

        @fp.fpy(ctx=bad_int)
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            return x + y

        with pytest.raises(
            CppCompileError,
            match='must use RTZ rounding mode',
        ):
            CppCompiler(
                unsafe_cast_int=True, optimize=False,
            ).compile(
                f, arg_types=[RealType(bad_int), RealType(bad_int)],
            )

    def test_unbounded_integer_rejected_when_flag_off(self):
        """Rounding under ``fp.INTEGER`` (unbounded ``MPFixedContext``)
        is rejected when ``unsafe_cast_int=False`` — C++ has no
        arbitrary-precision integer type, so any such rounding
        silently truncates to ``int64_t``.  The default
        (``unsafe_cast_int=True``) allows it; this test pins the
        opt-out path.

        Pinned with ``optimize=False`` for the same reason as
        :meth:`test_integer_non_rtz_rejected` — ``RoundElim``
        would otherwise eliminate the (identity) integer round
        and sidestep the rejection."""

        @fp.fpy(ctx=fp.INTEGER)
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            return x + y

        with pytest.raises(
            CppCompileError,
            match=r'unbounded integer context.*unsafe_cast_int',
        ):
            CppCompiler(
                unsafe_cast_int=False, optimize=False,
            ).compile(
                f, arg_types=[RealType(fp.INTEGER), RealType(fp.INTEGER)],
            )

    def test_unbounded_integer_allowed_by_default(self):
        """The same program compiles under the default settings —
        ``unsafe_cast_int=True`` is the default."""

        @fp.fpy(ctx=fp.INTEGER)
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            return x + y

        out = CppCompiler(optimize=False).compile(
            f, arg_types=[RealType(fp.INTEGER), RealType(fp.INTEGER)],
        )
        assert 'int64_t f(int64_t x, int64_t y)' in out
        assert 'return (x + y);' in out


class TestRealScopeLosslessWidening:
    """``with fp.REAL:`` has no native C++ rounding mode.  Per-op
    dispatch instead looks for a *wider* C++ type that contains the
    exact mathematical result of the operation — emitting the op in
    that wider type makes the C++ rounding observationally identical
    to REAL rounding (i.e. a no-op).  When no such widening exists,
    the compiler rejects."""

    def test_mixed_storage_widens_through_wider_ctx(self):
        """``with fp.REAL:`` over mixed-storage operands.

        ``round(0.01)`` infers to an ``FP32`` bound (storage ``float``);
        ``round(0.0)`` infers to ``SetFormat({0})`` (storage ``uint8_t``).  The
        multiplication cannot run at ``uint8_t`` width, so the widening picks
        the ``(float, float) → float under FP32`` sig and upcasts the zero
        operand.

        The *result* is ``float``, not ``uint8_t``: ``F * {0}`` no longer infers
        ``{0}``, because IEEE 754 makes that product ``-0.0`` for a negative
        multiplicand and NaN for an infinite one -- and a ``Format`` operand
        cannot rule either out.  A ``uint8_t`` result was the old, unsound
        answer: it is what made ``0.0 * inf`` compile to
        ``static_cast<uint8_t>(NaN)``.
        """
        @fp.fpy(ctx=fp.FP32)
        def f() -> fp.Real:
            return fp.round(0.01) * fp.round(0.0) + fp.round(0.0)

        out = CppCompiler().compile(f, ctx=fp.FP32)
        assert 'float f(' in out
        assert 'uint8_t' not in out, out
        # The widening still upcasts the narrow zero operand for the product.
        assert 'static_cast<float>(' in out

    def test_sint8_mul_widens_to_int16(self):
        """Exact product of two ``SINT8`` operands fits in
        ``int16_t``; the REAL scope lowers to a widened ``*``."""

        @fp.fpy
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            with fp.SINT8:
                xq = fp.round(x)
                yq = fp.round(y)
            with fp.REAL:
                return xq * yq

        out = CppCompiler(optimize=False).compile(
            f, ctx=fp.FP64,
            arg_types=[RealType(fp.FP32), RealType(fp.FP32)],
        )
        assert 'int16_t f(float x, float y)' in out
        assert (
            'return (static_cast<int16_t>(xq) * static_cast<int16_t>(yq));'
        ) in out
        # No fesetround for the REAL scope.
        assert 'fesetround' not in out

    def test_sint8_add_widens_to_int16(self):
        """Same widening shape for addition."""

        @fp.fpy
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            with fp.SINT8:
                xq = fp.round(x)
                yq = fp.round(y)
            with fp.REAL:
                return xq + yq

        out = CppCompiler(optimize=False).compile(
            f, ctx=fp.FP64,
            arg_types=[RealType(fp.FP32), RealType(fp.FP32)],
        )
        assert (
            'return (static_cast<int16_t>(xq) + static_cast<int16_t>(yq));'
        ) in out

    def test_fp64_mul_rejected(self):
        """Exact ``double * double`` would need ~106 mantissa bits;
        no ladder entry contains that, so REAL rejects."""

        @fp.fpy
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            with fp.REAL:
                return x * y

        with pytest.raises(
            CppCompileError,
            match='no storage type on the ladder contains',
        ):
            CppCompiler(optimize=False).compile(
                f, ctx=fp.FP64,
                arg_types=[RealType(fp.FP64), RealType(fp.FP64)],
            )

    def test_real_div_rejected(self):
        """Integer division under REAL is non-dyadic — no exact
        format, no widening."""

        @fp.fpy
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            with fp.SINT8:
                xq = fp.round(x)
                yq = fp.round(y)
            with fp.REAL:
                return xq / yq

        with pytest.raises(
            CppCompileError,
            match='cannot store an unconstrained real value',
        ):
            CppCompiler(optimize=False).compile(
                f, ctx=fp.FP64,
                arg_types=[RealType(fp.FP32), RealType(fp.FP32)],
            )
