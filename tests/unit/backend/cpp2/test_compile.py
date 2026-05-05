"""
Unit tests for the cpp2 backend.

The first commit only exercises the public API surface — the compiler
is a stub that raises :class:`Cpp2CompileError`.  Subsequent phases of
the plan in ``docs/todos/backend-cpp.md`` will add coverage as code
emission lands.
"""

import fpy2 as fp
import pytest

from fpy2.backend.cpp2 import Cpp2Compiler, Cpp2CompileError


class TestCpp2CompilerStub:
    """Phase 0: import + stub-error checks."""

    def test_import_surface(self):
        """The public API is reachable from the package import."""
        assert callable(Cpp2Compiler)
        assert issubclass(Cpp2CompileError, Exception)

    def test_compile_stub_raises(self):
        """``compile()`` is a stub until later phases land."""
        from fpy2.types import RealType

        @fp.fpy
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            return x + y

        compiler = Cpp2Compiler()
        # Pin arg types so storage selection succeeds; we want to reach
        # the (still-stubbed) emission error.
        with pytest.raises(Cpp2CompileError, match='emission is under construction'):
            compiler.compile(
                f, ctx=fp.FP64, arg_types=[RealType(fp.FP64), RealType(fp.FP64)]
            )

    def test_compile_unconstrained_args_rejects(self):
        """An un-monomorphized argument can't be assigned a finite C++
        storage type — the compiler reports a clear error pointing at
        the offending name rather than silently picking ``double``."""

        @fp.fpy
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            return x + y

        compiler = Cpp2Compiler()
        with pytest.raises(Cpp2CompileError, match='cannot pick storage'):
            compiler.compile(f)

    def test_compile_rejects_non_function(self):
        """Passing a non-Function raises ``TypeError`` before the stub fires."""
        compiler = Cpp2Compiler()
        with pytest.raises(TypeError, match='Function'):
            compiler.compile('not a function')  # type: ignore[arg-type]
