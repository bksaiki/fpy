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

        out = CppCompiler().compile(f)
        assert 'fesetround' not in out

    def test_concrete_with_block_resolves(self):
        @fp.fpy
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            with fp.FP64:
                return x + y

        out = CppCompiler().compile(
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
        out = CppCompiler().compile(f, arg_types=[RealType(fp.FP64)])
        assert 'return (x + x);' in out

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
        out = CppCompiler().compile(
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

        out = CppCompiler().compile(
            f, ctx=fp.FP64,
            arg_types=[RealType(fp.FP64), RealType(fp.FP64)],
        )
        assert 'fesetround' not in out

    def test_integer_context_no_fesetround(self):
        """``with INTEGER:`` doesn't emit fesetround — integer
        arithmetic doesn't consult ``fenv``."""

        @fp.fpy
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            with fp.INTEGER:
                return x + y

        out = CppCompiler().compile(
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

        out = CppCompiler().compile(
            f, arg_types=[RealType(fp.FP64), RealType(fp.FP64)],
        )
        assert 'fesetround' not in out

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

        out = CppCompiler().compile(
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

        out = CppCompiler().compile(
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
        out = CppCompiler().compile(
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
            CppCompiler().compile(
                f, arg_types=[RealType(fp.FP64), RealType(fp.FP64)],
            )

    def test_integer_non_rtz_rejected(self):
        """Integer contexts must use RTZ."""

        bad_int = fp.MPFixedContext(-1, fp.RM.RNE)

        @fp.fpy(ctx=bad_int)
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            return x + y

        with pytest.raises(
            CppCompileError,
            match='must use RTZ rounding mode',
        ):
            CppCompiler().compile(
                f, arg_types=[RealType(bad_int), RealType(bad_int)],
            )
