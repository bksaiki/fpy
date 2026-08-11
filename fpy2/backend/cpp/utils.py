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

The runtime is ``fpy::min``/``max`` and nothing else: everything a list needs
is generated at the use site in standard-library spellings
(``std::shared_ptr<std::vector<T>>``, ``std::make_shared``), so a program
embedding a kernel depends on nothing of ours.  See :class:`.types.CppList`
for why the boxed representation is a ``shared_ptr`` at all.  The conversions
a *caller* needs to hand a ``std::vector`` to a boxed kernel are in
``CPP_INTEROP``, with the tests that exercise that boundary.
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

# `fpy::min`/`max` are IEEE 754-2019 minimum/maximum: NaN-propagating and
# signed-zero-correct.  `std::fmin`/`fmax` are neither -- they ignore NaN, and
# libstdc++ compiles the variable-operand path to `(a < b) ? a : b`, which
# returns +0 for `fmin(-0.0, +0.0)`.  The `a == b` tie-break fixes both.
# Integer min/max use `std::min`/`max`: no NaN, no signed zero.
CPP_HELPERS: str = '''\
namespace fpy {

template <typename T>
inline T min(T a, T b) {
    if (std::isnan(a) || std::isnan(b))
        return std::numeric_limits<T>::quiet_NaN();
    if (a == b)
        return std::signbit(a) ? a : b;
    return (a < b) ? a : b;
}

template <typename T>
inline T max(T a, T b) {
    if (std::isnan(a) || std::isnan(b))
        return std::numeric_limits<T>::quiet_NaN();
    if (a == b)
        return std::signbit(a) ? b : a;
    return (a < b) ? b : a;
}

}  // namespace fpy
'''
