"""
Phase 6 tests for the cpp emitter — translation-unit preamble.

``CppCompiler.compile`` returns just a function definition.  For
end-to-end compilation the caller pulls ``headers()`` / ``helpers()``
explicitly (or ``prelude()`` for both at once) and concatenates the
result with each compiled function.
"""

from fpy2.backend.cpp import CppCompiler


class TestHeaders:
    """Header set covers everything the emitter actually uses."""

    def test_headers_include_required_set(self):
        cc = CppCompiler()
        headers = cc.headers()
        # Each entry is a full ``#include`` line.
        for required in (
            '<algorithm>',
            '<cassert>',
            '<cfenv>',
            '<cmath>',
            '<array>',
            '<cstddef>',
            '<cstdint>',
            '<memory>',
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




