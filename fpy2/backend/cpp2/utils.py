"""
cpp2 backend: utilities — headers and runtime helpers.

The compiler's :meth:`Cpp2Compiler.compile` returns just a function
definition (so single-function tests can use exact-string equality).
Callers that want a complete translation unit pull
:meth:`Cpp2Compiler.headers` and :meth:`Cpp2Compiler.helpers`
explicitly and concatenate them — same shape as the legacy
``cpp/`` backend.

Header coverage tracks what the emitter actually uses:

- ``<cassert>``: ``assert(...)`` from ``RoundExact``.
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

Helpers is currently empty — cpp2 doesn't yet need any custom
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
    '#include <numeric>',
    '#include <vector>',
    '#include <tuple>',
)

CPP_HELPERS: str = ''
