"""
cpp2 backend: storage-type selection.

For each definition with a known :class:`FormatBound`, choose the
smallest C++ type from the storage ladder whose representable set
contains the bound.  When a name has multiple SSA defs (e.g., a loop
phi merging two branches), the storage type is the smallest entry that
contains every constituent format.
"""

from ...analysis.format_infer import (
    AbstractFormat,
    FormatBound,
    ListFormat,
    SetFormat,
    TupleFormat,
)
from ...analysis.format_infer.analysis import _to_abstract
from ...number import (
    FP32, FP64,
    SINT8, SINT16, SINT32, SINT64,
    UINT8, UINT16, UINT32, UINT64,
)
from ...number.context.format import Format
from ...number.context.real import REAL_FORMAT

from .types import CppList, CppScalar, CppTuple, CppType


# ----------------------------------------------------------------------
# The storage ladder.
#
# Each entry pairs a CppScalar with the AbstractFormat of the smallest
# context that fits in that scalar.  ``choose_storage_scalar`` walks the
# ladder in order and picks the first entry whose AbstractFormat
# contains the inferred bound.  Order matters: smaller types first.

def _af(fmt: Format) -> AbstractFormat:
    af = AbstractFormat.from_format(fmt)
    assert af is not None, f'expected abstractable format, got {fmt!r}'
    return af


_LADDER: tuple[tuple[CppScalar, AbstractFormat], ...] = (
    (CppScalar.U8, _af(UINT8.format())),
    (CppScalar.S8, _af(SINT8.format())),
    (CppScalar.U16, _af(UINT16.format())),
    (CppScalar.S16, _af(SINT16.format())),
    (CppScalar.U32, _af(UINT32.format())),
    (CppScalar.S32, _af(SINT32.format())),
    (CppScalar.F32, _af(FP32.format())),
    (CppScalar.U64, _af(UINT64.format())),
    (CppScalar.S64, _af(SINT64.format())),
    (CppScalar.F64, _af(FP64.format())),
)
"""Storage ladder, smallest first.  Searched linearly for the first
covering type."""


class StorageSelectionError(Exception):
    """Raised when no storage type contains the inferred format."""
    pass


def choose_storage_scalar(bound: FormatBound) -> CppScalar:
    """
    Picks the smallest scalar storage that contains *bound*.

    - ``None`` (non-numeric, e.g., the format of a comparison) → ``BOOL``.
    - :class:`SetFormat` / scalar :class:`Format` → smallest ladder
      entry whose AbstractFormat ``>= from_format(bound)``.
    - ``REAL_FORMAT`` → :exc:`StorageSelectionError` (no finite ladder
      entry covers all reals).
    """
    if bound is None:
        return CppScalar.BOOL
    if bound == REAL_FORMAT:
        raise StorageSelectionError(
            'cannot store an unconstrained real value in a finite C++ type; '
            'is the active rounding context symbolic? '
            'Try monomorphizing the function with a concrete context.'
        )
    af = _to_abstract(bound)
    if af is None:
        raise StorageSelectionError(
            f'cannot lift {bound!r} to AbstractFormat; '
            'storage selection requires a dyadic format'
        )
    for cpp_ty, ladder_af in _LADDER:
        if af <= ladder_af:
            return cpp_ty
    raise StorageSelectionError(
        f'no storage type on the ladder contains {bound!r}'
    )


def choose_storage(bound: FormatBound) -> CppType:
    """
    Recursively picks the storage type for a (possibly structured)
    :class:`FormatBound`.

    - Scalars dispatch to :func:`choose_storage_scalar`.
    - :class:`TupleFormat` becomes ``std::tuple<...>``.
    - :class:`ListFormat` becomes ``std::vector<...>``.
    """
    if isinstance(bound, TupleFormat):
        return CppTuple(tuple(choose_storage(b) for b in bound.elts))
    if isinstance(bound, ListFormat):
        return CppList(choose_storage(bound.elt))
    return choose_storage_scalar(bound)


def aggregate_storage(bounds: list[FormatBound]) -> CppType:
    """
    Picks a single storage type that contains every bound in *bounds*.

    Used when a name has multiple SSA defs (e.g., loop phi widening) —
    the variable's C++ declaration must accommodate every value
    assigned into it.

    The simplest implementation: pick storage for each bound, then take
    the supremum on the ladder.  For scalars: largest covering ladder
    type across all defs.  For structured types: structural recursion.
    """
    assert bounds, 'aggregate_storage requires at least one bound'
    storages = [choose_storage(b) for b in bounds]
    return _supremum(storages)


def _supremum(storages: list[CppType]) -> CppType:
    """
    Smallest storage that contains every storage in *storages*.

    Assumes all entries have the same structural shape (scalar / list /
    tuple).  This is enforced by the type checker upstream, so a
    mismatch indicates an analysis bug rather than user error.
    """
    head, *rest = storages
    if not rest:
        return head
    if isinstance(head, CppScalar):
        # All must be scalars.
        assert all(isinstance(s, CppScalar) for s in rest), (
            f'inconsistent storage shapes: {storages!r}'
        )
        return _scalar_sup([head] + [s for s in rest if isinstance(s, CppScalar)])
    if isinstance(head, CppList):
        assert all(isinstance(s, CppList) for s in rest)
        elts = [head.elt] + [s.elt for s in rest if isinstance(s, CppList)]
        return CppList(_supremum(elts))
    if isinstance(head, CppTuple):
        assert all(isinstance(s, CppTuple) and len(s.elts) == len(head.elts) for s in rest)
        n = len(head.elts)
        merged = []
        tuples = [head] + [s for s in rest if isinstance(s, CppTuple)]
        for i in range(n):
            merged.append(_supremum([t.elts[i] for t in tuples]))
        return CppTuple(tuple(merged))
    raise TypeError(f'unexpected CppType: {head!r}')


def _scalar_sup(scalars: list[CppScalar]) -> CppScalar:
    """Smallest scalar on the ladder that subsumes every input."""
    # Filter out BOOL specifically — mixing bool with numeric storage is
    # a typing bug, not a widening situation.
    if any(s is CppScalar.BOOL for s in scalars):
        if all(s is CppScalar.BOOL for s in scalars):
            return CppScalar.BOOL
        raise StorageSelectionError(
            f'cannot widen across BOOL and numeric storage: {scalars!r}'
        )
    # For each ladder entry, accept it iff every input is <= it.
    ladder_index = {ty: i for i, (ty, _) in enumerate(_LADDER)}
    # Gather indices and pick the max — but only if all are on the same
    # ladder.  (BOOL was excluded above; everything else is on the ladder.)
    try:
        max_idx = max(ladder_index[s] for s in scalars)
    except KeyError as e:
        raise StorageSelectionError(
            f'storage scalar not on the ladder: {e.args[0]!r}'
        )
    # The max isn't necessarily a covering type for all of them — e.g.,
    # S32 ⊔ U32 needs S64 (signed must absorb unsigned of equal width).
    # Walk the ladder from max_idx upward until we find a type that
    # covers all the ladder ABs.
    afs = []
    for s in scalars:
        for ty, af in _LADDER:
            if ty is s:
                afs.append(af)
                break
    for i in range(max_idx, len(_LADDER)):
        ty, af = _LADDER[i]
        if all(other <= af for other in afs):
            return ty
    raise StorageSelectionError(
        f'no storage type on the ladder subsumes {scalars!r}'
    )
