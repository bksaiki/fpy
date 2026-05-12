"""
Phase 6 tests for the cpp emitter — translation-unit preamble.

``CppCompiler.compile`` returns just a function definition.  For
end-to-end compilation the caller pulls ``headers()`` / ``helpers()``
explicitly (or ``prelude()`` for both at once) and concatenates the
result with each compiled function.
"""

import fpy2 as fp

from fpy2.backend.cpp import CppCompiler
from fpy2.types import RealType


class TestHeaders:
    """Header set covers everything the emitter actually uses."""

    def test_headers_returns_a_list(self):
        cc = CppCompiler()
        headers = cc.headers()
        assert isinstance(headers, list)

    def test_headers_include_required_set(self):
        cc = CppCompiler()
        headers = cc.headers()
        # Each entry is a full ``#include`` line.
        for required in (
            '<cassert>',
            '<cfenv>',
            '<cmath>',
            '<cstddef>',
            '<cstdint>',
            '<numeric>',
            '<vector>',
            '<tuple>',
        ):
            assert any(required in h for h in headers), (
                f'missing header for {required}'
            )
            # Lines start with ``#include``.
            assert all(h.startswith('#include') for h in headers)

    def test_headers_returns_a_fresh_list(self):
        """Mutating the returned list shouldn't affect future calls."""
        cc = CppCompiler()
        h1 = cc.headers()
        h1.append('#include <bogus>')
        h2 = cc.headers()
        assert '#include <bogus>' not in h2


class TestHelpers:
    """Runtime helpers — currently empty but the slot exists."""

    def test_helpers_returns_string(self):
        cc = CppCompiler()
        assert isinstance(cc.helpers(), str)


class TestPrelude:
    """``prelude`` = headers + helpers concatenated."""

    def test_prelude_starts_with_includes(self):
        cc = CppCompiler()
        pre = cc.prelude()
        assert pre.startswith('#include')

    def test_prelude_contains_each_header(self):
        cc = CppCompiler()
        pre = cc.prelude()
        for required in ('<cassert>', '<cfenv>', '<cmath>',
                         '<cstdint>', '<vector>', '<tuple>'):
            assert required in pre


class TestCompileStillFunctionOnly:
    """``compile()`` itself does not emit headers — exact-string
    tests in the rest of the suite rely on this."""

    def test_compile_does_not_include_headers(self):
        @fp.fpy
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            with fp.FP64:
                return x + y

        out = CppCompiler().compile(
            f, ctx=fp.FP64,
            arg_types=[RealType(fp.FP64), RealType(fp.FP64)],
        )
        assert '#include' not in out
        assert out.startswith('double f')


class TestEndToEndUnit:
    """Smoke-test that combining ``prelude`` + ``compile`` produces
    a syntactically self-contained translation unit string."""

    def test_unit_combines(self):
        @fp.fpy
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            with fp.FP64:
                return x + y

        cc = CppCompiler()
        body = cc.compile(
            f, ctx=fp.FP64,
            arg_types=[RealType(fp.FP64), RealType(fp.FP64)],
        )
        unit = cc.prelude() + body
        # Order: includes first, then function.
        assert unit.index('#include <cmath>') < unit.index('double f')
