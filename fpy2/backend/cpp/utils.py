"""
cpp backend: utilities — headers and runtime helpers.

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
ours -- unconditionally, not just for the programs that happen to avoid a
helper.  See :class:`.types.CppList` for why the boxed representation is a
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
"""Empty: the backend emits no support code.

`Min`/`Max` were the last entry -- IEEE 754-2019 ``minimum``/``maximum``, which
``std::fmin``/``fmax`` are not (they ignore a NaN, and leave the ±0 choice
unspecified).  `CppEmitter._emit_ieee_min_max` now emits that inline.  Kept as a
symbol because :meth:`CppCompiler.helpers` is part of the public shape and every
caller concatenates it.
"""
