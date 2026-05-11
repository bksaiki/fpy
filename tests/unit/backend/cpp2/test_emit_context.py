"""
Phase 5b tests for the cpp2 emitter — context boundaries.

The active rounding context is taken from the
:class:`ContextUseAnalysis` scope at every ``FuncDef`` /
``ContextStmt`` site.  cpp2 only compiles programs whose contexts
are statically resolvable; symbolic context variables are rejected.

For float contexts the rounding mode must be one of the four
``fesetround``-supported modes (RNE / RTZ / RTP / RTN).  For integer
contexts the rounding mode must be RTZ — C++ integer arithmetic
already truncates toward zero, so no runtime support is needed.
"""

import fpy2 as fp
import pytest

from fpy2.backend.cpp2 import Cpp2Compiler, Cpp2CompileError
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

        out = Cpp2Compiler().compile(f)
        assert 'fesetround' not in out

    def test_concrete_with_block_resolves(self):
        @fp.fpy
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            with fp.FP64:
                return x + y

        out = Cpp2Compiler().compile(
            f, ctx=fp.FP64,
            arg_types=[RealType(fp.FP64), RealType(fp.FP64)],
        )
        # FP64 default-RM (RNE) — no fesetround.
        assert 'fesetround' not in out

    def test_function_scope_unused_when_all_ops_nested(self):
        """When every op lives inside an inner ``with``, the
        function-level scope has no uses and isn't validated — so
        the outer (function-level) context can be anything,
        including a context the cpp2 backend doesn't natively
        support."""

        @fp.fpy(ctx=_RNA_64)              # RNA is unsupported by fesetround,
        def f(x: fp.Real) -> fp.Real:     # but this function never dispatches
            with fp.FP64:                  # under the outer scope — every op
                return x + x               # is inside ``with fp.FP64:``.

        # No error: compilation succeeds.
        out = Cpp2Compiler().compile(f, arg_types=[RealType(fp.FP64)])
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
        out = Cpp2Compiler().compile(
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

        out = Cpp2Compiler().compile(
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

        out = Cpp2Compiler().compile(
            f, ctx=fp.INTEGER,
            arg_types=[RealType(fp.INTEGER), RealType(fp.INTEGER)],
        )
        assert 'fesetround' not in out
        assert 'int64_t f(int64_t x, int64_t y)' in out


class TestNonDefaultRmEmitsFesetround:
    """Non-RNE float contexts emit save / set / restore around
    their body."""

    def test_function_level_rtz(self):
        @fp.fpy(ctx=_RTZ_64)
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            return x + y

        out = Cpp2Compiler().compile(
            f, arg_types=[RealType(fp.FP64), RealType(fp.FP64)],
        )
        assert 'std::fegetround()' in out
        assert 'std::fesetround(FE_TOWARDZERO)' in out

    def test_with_block_changes_rm(self):
        """An inner ``with`` that switches to a different RM emits
        fesetround on entry and restores on exit."""

        @fp.fpy(ctx=_RTZ_64)
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            a = x + y
            with fp.FP64:
                b = a + 1
            return a + b

        out = Cpp2Compiler().compile(
            f, arg_types=[RealType(fp.FP64), RealType(fp.FP64)],
        )
        # Outer (function-level) sets RTZ.
        assert 'std::fesetround(FE_TOWARDZERO)' in out
        # Inner switches to RNE (the default), then restores.
        assert 'std::fesetround(FE_TONEAREST)' in out

    def test_with_block_same_rm_skips_fesetround(self):
        """A nested ``with`` whose RM matches the current mode
        emits no fesetround — nothing to change."""

        @fp.fpy(ctx=_RTZ_64)
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            with _RTZ_64:
                return x + y

        out = Cpp2Compiler().compile(
            f, arg_types=[RealType(fp.FP64), RealType(fp.FP64)],
        )
        # One save (function entry) + one set; no inner change.
        assert out.count('std::fegetround()') == 1
        assert out.count('std::fesetround(FE_TOWARDZERO)') == 1


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
            Cpp2CompileError,
            match='not supported by ``fesetround``',
        ):
            Cpp2Compiler().compile(
                f, arg_types=[RealType(fp.FP64), RealType(fp.FP64)],
            )

    def test_integer_non_rtz_rejected(self):
        """Integer contexts must use RTZ."""

        bad_int = fp.MPFixedContext(-1, fp.RM.RNE)

        @fp.fpy(ctx=bad_int)
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            return x + y

        with pytest.raises(
            Cpp2CompileError,
            match='must use RTZ rounding mode',
        ):
            Cpp2Compiler().compile(
                f, arg_types=[RealType(bad_int), RealType(bad_int)],
            )
