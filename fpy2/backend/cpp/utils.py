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

``fpy::list`` is why a runtime exists at all: an FPy list is a list of
*references*, and binding shares those cells, so a ``std::vector`` — a value —
cannot represent one.  Three choices worth not relitigating:

- A plain alias for ``std::shared_ptr``, not a wrapper class, so a program
  embedding a kernel can handle a list without depending on anything of ours.
- The control block is atomic, so copying a handle is an atomic increment.  The
  emitter passes ``const list<T>&`` wherever a name is not rebound, and a
  reference does no refcounting — that is what keeps the atomic off the hot path.
- Refcounting needs no collector: a cycle needs a list to hold itself, and list
  types are finite, so ``xs[0] = xs`` fails to unify.  It is the *type* system
  that rules this out, not the store rules, which would permit it.

Only what emitted code names lives here.  The conversions a *caller* needs to
hand a ``std::vector`` to a kernel are in ``CPP_INTEROP``, with the tests that
exercise that boundary.
"""


CPP_HEADERS: tuple[str, ...] = (
    '#include <algorithm>',
    '#include <cassert>',
    '#include <cfenv>',
    '#include <cmath>',
    '#include <cstddef>',
    '#include <cstdint>',
    '#include <initializer_list>',
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

// An FPy list: a handle to a shared, mutable sequence.  Copying a `list` shares
// its elements; only `make_list` allocates.
template <typename T>
using list = std::shared_ptr<std::vector<T> >;

template <typename T>
inline list<T> make_list(std::size_t n) {
    return std::make_shared<std::vector<T> >(n);
}

template <typename T>
inline list<T> make_list(std::size_t n, const T& x) {
    return std::make_shared<std::vector<T> >(n, x);
}

template <typename T>
inline list<T> make_list(std::initializer_list<T> il) {
    return std::make_shared<std::vector<T> >(il);
}

template <typename T, typename It>
inline list<T> make_list(It first, It last) {
    return std::make_shared<std::vector<T> >(first, last);
}

}  // namespace fpy
'''
