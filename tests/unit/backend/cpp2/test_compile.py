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

        @fp.fpy
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            return x + y

        compiler = Cpp2Compiler()
        with pytest.raises(Cpp2CompileError, match='not yet implemented'):
            compiler.compile(f)

    def test_compile_rejects_non_function(self):
        """Passing a non-Function raises ``TypeError`` before the stub fires."""
        compiler = Cpp2Compiler()
        with pytest.raises(TypeError, match='Function'):
            compiler.compile('not a function')  # type: ignore[arg-type]
