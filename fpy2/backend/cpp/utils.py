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
- ``<initializer_list>``: ``fpy::make_list`` from a braced list.
- ``<limits>``: ``quiet_NaN`` in ``fpy::min`` / ``fpy::max``.
- ``<memory>``: ``std::shared_ptr`` behind ``fpy::list``.
- ``<numeric>``: ``std::accumulate`` for ``Sum``.
- ``<vector>``: the element storage inside ``fpy::list``.
- ``<tuple>``: ``std::tuple`` / ``std::make_tuple`` / ``std::get`` for
  tuples and tuple-binding destructuring.

``fpy::list`` is why the runtime exists.  FPy lists alias on assignment, so a
``std::vector`` — a value — cannot represent one; the handle supplies the
indirection.  Notes on the choices, kept here rather than in the emitted code:

- A plain alias for ``std::shared_ptr``, not a wrapper class: the standard type
  already has the required semantics, and a program embedding a generated kernel
  can then handle a list without depending on anything of ours.
- The control block is atomic, so copying a handle is an atomic increment.  The
  emitter passes ``const list<T>&`` wherever a name is not rebound (see
  ``_arg_decl``); a reference does no refcounting, which is what keeps the atomic
  off the hot path.
- Reference counting suffices with no collector: FPy has no recursive list type,
  so no location can be stored within itself and no cycle is constructible.

Interop is by conversion at the call site (``borrow`` / ``copy_in`` /
``copy_out``) rather than by emitting a wrapper per function, so the choice
between sharing and copying stays with the caller and the emitted signature is
the only signature.  Two consequences worth knowing:

- A ``vector<vector<T>>`` can only be copied.  The caller stores rows by value
  where a list stores handles, so no arrangement makes a write through either
  side visible to the other.
- A borrowed handle must not outlive its vector.  That holds because a callee
  cannot retain one: FPy has no globals, and captures are materialized before
  compilation.

Pinned by ``_test_abi`` in ``tests/infra/backend/cpp.py``.
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

// Interop: a program holding `std::vector` converts here.  A flat vector can be
// shared or copied; a nested one can only be copied.

// Must not outlive `v`.
template <typename T>
inline list<T> borrow(std::vector<T>& v) {
    return list<T>(&v, [](std::vector<T>*) {});
}

template <typename T>
inline list<T> copy_in(const std::vector<T>& v) {
    return std::make_shared<std::vector<T> >(v);
}

template <typename T>
inline list<list<T> > copy_in(const std::vector<std::vector<T> >& vs) {
    list<list<T> > out = make_list<list<T> >(vs.size());
    for (std::size_t i = 0; i < vs.size(); ++i)
        (*out)[i] = copy_in(vs[i]);
    return out;
}

template <typename T>
inline std::vector<T> copy_out(const list<T>& xs) {
    return *xs;
}

template <typename T>
inline std::vector<std::vector<T> > copy_out(const list<list<T> >& xss) {
    std::vector<std::vector<T> > out(xss->size());
    for (std::size_t i = 0; i < xss->size(); ++i)
        out[i] = *(*xss)[i];
    return out;
}

}  // namespace fpy
'''
