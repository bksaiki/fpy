"""
cpp backend: utilities — headers and runtime helpers.

The compiler's :meth:`CppCompiler.compile` returns just a function
definition (so single-function tests can use exact-string equality).
Callers that want a complete translation unit pull
:meth:`CppCompiler.headers` and :meth:`CppCompiler.helpers`
explicitly and concatenate them — same shape as the legacy
``cpp/`` backend.

Header coverage tracks what the emitter actually uses:

- ``<algorithm>``: ``std::any_of`` / ``std::all_of`` for ``AnyOf`` / ``AllOf``,
  and ``std::min`` / ``std::max`` for the integer ``min``/``max`` paths.
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

// An FPy list: a reference to a shared, mutable sequence.
//
// FPy list semantics are Python's -- assignment aliases, `xs[i] = e` mutates
// the object, and passing / returning / projecting carries the identity along.
// A bare `std::vector` is a *value*, so it cannot express that; the shared
// pointer adds the level of indirection that can.  Copying a `list` shares the
// elements; only `make_list` / `clone` allocate new ones.
//
// Deliberately a plain alias rather than a wrapper class: `std::shared_ptr`
// already has exactly the required semantics, and a host program embedding a
// generated kernel can handle one without depending on anything of ours.
//
// The control block is atomic, so a handle copy is an atomic increment.  The
// emitter therefore passes handles by `const list<T>&` wherever the name is not
// rebound -- a reference does no refcounting, and measurement showed that
// passing by value in a hot call chain is the only case where the atomic cost is
// visible at all.
//
// No cycle can be constructed -- FPy's type syntax cannot express a recursive
// list type -- so reference counting never leaks and no collector is needed.
template <typename T>
using list = std::shared_ptr<std::vector<T> >;

template <typename T>
inline list<T> make_list(std::size_t n) {
    return std::make_shared<std::vector<T> >(n);
}

template <typename T>
inline list<T> make_list(std::initializer_list<T> il) {
    return std::make_shared<std::vector<T> >(il);
}

// An independent list with the same elements -- FPy's `xs[:]`, and the explicit
// opt-out from sharing.
template <typename T>
inline list<T> clone(const list<T>& xs) {
    return std::make_shared<std::vector<T> >(*xs);
}

// A non-owning handle onto storage someone else owns, for passing a caller's
// vector into a kernel without copying it.  Sound only because the handle
// cannot outlive the call: FPy has no globals and `FreeVarElim` materializes
// captures, so a callee cannot retain it.
template <typename T>
inline list<T> borrow(std::vector<T>& v) {
    return list<T>(&v, [](std::vector<T>*) {});
}

}  // namespace fpy
'''
