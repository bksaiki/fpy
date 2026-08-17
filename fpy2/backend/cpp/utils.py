"""
cpp backend: utilities — headers and preamble.

The compiler's :meth:`CppCompiler.compile` returns just a function
definition (so single-function tests can use exact-string equality).
Callers that want a complete translation unit pull
:meth:`CppCompiler.headers` and :meth:`CppCompiler.helpers`
explicitly and concatenate them — same shape as the legacy
``cpp/`` backend.

Headers track exactly what the emitter uses; the list below is the whole
dependency set.

**There is no runtime.**  Everything is generated at the use site in
standard-library spellings (``std::shared_ptr<std::vector<T>>``,
``std::make_shared``), so a program embedding a kernel depends on nothing of
ours.  See :class:`.types.CppList` for why the boxed representation is a
``shared_ptr`` at all.  The conversions a *caller* needs to hand a
``std::vector`` to a boxed kernel are in ``CPP_INTEROP``, with the tests that
exercise that boundary.
"""


CPP_HEADERS: tuple[str, ...] = (
    '#include <algorithm>',
    '#include <array>',
    '#include <cassert>',
    '#include <cfenv>',
    '#include <cmath>',
    '#include <cstddef>',
    '#include <cstdint>',
    '#include <limits>',
    '#include <memory>',
    '#include <numeric>',
    '#include <vector>',
    '#include <tuple>',
)

CPP_HELPERS: str = ''
"""No support code: everything is emitted inline in ``std::`` spellings.

Kept so callers can concatenate headers, helpers and body unconditionally.
"""
