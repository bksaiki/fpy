"""
Unit tests for the cpp backend.

The first commit only exercises the public API surface — the compiler
is a stub that raises :class:`CppCompileError`.  Subsequent phases of
the plan in ``docs/todos/backend-cpp.md`` will add coverage as code
emission lands.
"""

import fpy2 as fp
import pytest

from fpy2.backend.cpp import CppCompiler, CppCompileError


class TestCppCompilerStub:
    """Phase 0: import + stub-error checks."""

    def test_import_surface(self):
        """The public API is reachable from the package import."""
        assert callable(CppCompiler)
        assert issubclass(CppCompileError, Exception)

    def test_compile_returns_source_string(self):
        """A simple monomorphized program compiles to a non-empty C++
        source string."""
        from fpy2.types import RealType

        @fp.fpy
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            with fp.FP64:
                return x + y

        compiler = CppCompiler()
        out = compiler.compile(
            f, ctx=fp.FP64, arg_types=[RealType(fp.FP64), RealType(fp.FP64)]
        )
        assert isinstance(out, str)
        assert 'double f(double x, double y)' in out

    def test_compile_unconstrained_args_rejects(self):
        """An un-monomorphized argument can't be assigned a finite C++
        storage type — the compiler reports a clear error pointing at
        the offending name rather than silently picking ``double``."""

        @fp.fpy
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            return x + y

        compiler = CppCompiler()
        with pytest.raises(CppCompileError, match='cannot pick storage'):
            compiler.compile(f)

    def test_compile_rejects_non_function(self):
        """Passing a non-Function raises ``TypeError`` before the stub fires."""
        compiler = CppCompiler()
        with pytest.raises(TypeError, match='Function'):
            compiler.compile('not a function')  # type: ignore[arg-type]


class TestSpecializationNameCollisions:
    """Distinct :class:`FuncDef`s that share a source name (e.g.
    ``vector.zeros`` and ``matrix.zeros``) must emit distinct C++
    function names within a single translation unit — otherwise the
    C++ compiler rejects the unit with an ODR redefinition error.

    The mangler discriminates by ``func_id`` (AST identity), not by
    name; collisions get a stable ``_f<digest>`` suffix on the
    later-registered spec.
    """

    def test_cross_funcdef_same_name_disambiguates(self):
        """Two distinct FPy functions sharing a source name —
        ``fpy2.libraries.vector.zeros`` (1-D) and
        ``fpy2.libraries.matrix.zeros`` (2-D) — must emit distinct
        C++ definitions when both appear in the same translation
        unit.  Exercises the *callee* mangling path: each ``zeros``
        is reached as a callee of its own wrapper function."""
        import re
        from fpy2.libraries import matrix, vector
        from fpy2.types import RealType

        @fp.fpy
        def call_vec(n: int) -> list[fp.Real]:
            return vector.zeros(n)

        @fp.fpy
        def call_mat(n: int) -> list[list[fp.Real]]:
            return matrix.zeros(n, n)

        cc = CppCompiler(unsafe_cast_int=True)
        m = fp.Module()
        m.add(call_vec, ctx=fp.FP64, arg_types=[RealType(fp.INTEGER)])
        m.add(call_mat, ctx=fp.FP64, arg_types=[RealType(fp.INTEGER)])
        out = cc.compile_module(m)

        # After the ``Specialize`` integration, callee names are
        # mangled as ``zeros__<ctx_sha1>__<args_sha1>``.  Two distinct
        # FuncDefs sharing the source name must produce two distinct
        # mangled names (otherwise the C++ compiler rejects the unit
        # with an ODR redefinition error).  Match definitions by
        # ``<name>(`` rather than ``<ret> <name>(`` since the two
        # specializations have different return types.
        zeros_pat = re.compile(r'\bzeros__[0-9a-f]{8}__[0-9a-f]{8}\(')
        occurrences = zeros_pat.findall(out)
        # Each distinct spec appears exactly twice (one definition
        # site + one call site).  Two specs → four occurrences total
        # across two distinct names.
        assert len(occurrences) == 4, (
            f'expected 4 mangled `zeros__<ctx>__<args>(` occurrences '
            f'(2 specs * (def + call)):\n{out}'
        )
        distinct_names = set(occurrences)
        assert len(distinct_names) == 2, (
            f'expected 2 distinct mangled names for the two `zeros` '
            f'specs, got {distinct_names!r}'
        )
