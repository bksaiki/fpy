"""
cpp backend: utilities — headers and runtime helpers.

The compiler's :meth:`CppCompiler.compile` returns just a function
definition (so single-function tests can use exact-string equality).
Callers that want a complete translation unit pull
:meth:`CppCompiler.headers` and :meth:`CppCompiler.helpers`
explicitly and concatenate them — same shape as the legacy
``cpp/`` backend.

Header coverage tracks what the emitter actually uses:

- ``<cassert>``: ``assert(...)`` from ``Cast``.
- ``<cfenv>``: ``std::fegetround`` / ``std::fesetround`` and the
  ``FE_*`` rounding-mode macros.
- ``<cmath>``: every ``std::fabs`` / ``std::sqrt`` / ``std::sin`` /
  ``std::isnan`` / etc. dispatched through the op table.
- ``<cstddef>``: ``size_t`` for vector indexing.
- ``<cstdint>``: fixed-width ``int8_t`` … ``uint64_t``.
- ``<numeric>``: ``std::accumulate`` for ``Sum``.
- ``<vector>``: ``std::vector<T>`` for FPy lists.
- ``<tuple>``: ``std::tuple`` / ``std::make_tuple`` / ``std::get`` for
  tuples and tuple-binding destructuring.

Helpers is currently empty — cpp doesn't yet need any custom
runtime support beyond what ``<cmath>`` / ``std::vector`` already
give us.  The slot exists so future additions (e.g., an RAII
``fenv`` guard to fix the function-level fesetround leak, or
bounds-checked subscript helpers for strict slice semantics) have a
home.
"""


CPP_HEADERS: tuple[str, ...] = (
    '#include <cassert>',
    '#include <cfenv>',
    '#include <cmath>',
    '#include <cstddef>',
    '#include <cstdint>',
    '#include <limits>',
    '#include <numeric>',
    '#include <vector>',
    '#include <tuple>',
)

# IEEE 754-2019 ``minimum`` / ``maximum`` for floating-point ``T``:
# NaN-propagating and signed-zero-correct.  ``std::fmin`` / ``std::fmax``
# follow C99 / IEEE 754-2008 ``minNum`` (NaN-ignoring), and on libstdc++
# the non-constant-folded path is just ``(a < b) ? a : b``, which loses
# the ±0 distinction (``std::fmin(-0.0, +0.0)`` with variable operands
# returns ``+0`` instead of ``-0``).  The explicit ``a == b`` tie-break
# fixes both issues.
#
# Only emitted for floating-point ``T``; integer ``min`` / ``max``
# continue to use ``std::min`` / ``std::max`` (no NaN, no signed-zero).
CPP_HELPERS: str = '''\
template <typename T>
inline T __fpy_min(T a, T b) {
    if (std::isnan(a) || std::isnan(b))
        return std::numeric_limits<T>::quiet_NaN();
    if (a == b)
        return std::signbit(a) ? a : b;
    return (a < b) ? a : b;
}

template <typename T>
inline T __fpy_max(T a, T b) {
    if (std::isnan(a) || std::isnan(b))
        return std::numeric_limits<T>::quiet_NaN();
    if (a == b)
        return std::signbit(a) ? b : a;
    return (a < b) ? b : a;
}
'''
