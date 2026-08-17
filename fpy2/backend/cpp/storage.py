"""
cpp backend: storage-type selection.

For each :class:`FormatBound`, pick the smallest C++ type from the
storage ladder whose representable set contains the bound.  Sibling
:mod:`.storage_infer` consumes the ladder to assign a single storage
type per phi-web equivalence class.

The module also exposes:

- :func:`scalar_fits_in` — ladder-level containment between two
  :class:`CppScalar`s.  Used by the cast helper to reject lossy
  implicit conversions.
- :func:`scalar_sup` — smallest ladder scalar subsuming every
  input.  Used by :meth:`_visit_compare` to pick the common type
  for a chained comparison.
- :func:`bound_fits_in_scalar` — value-level containment, for a
  conversion the type-level test refuses.
"""

from ...analysis.format_infer import (
    AbstractableFormat,
    AbstractFormat,
    FormatBound,
    ListFormat,
    SetFormat,
    TupleFormat,
    is_bottom,
)
from ...analysis.format_infer.analysis import _to_abstract
from ...number import (
    FP32,
    FP64,
    SINT8,
    SINT16,
    SINT32,
    SINT64,
    UINT8,
    UINT16,
    UINT32,
    UINT64,
)
from ...number.context.mp_fixed import MPFixedFormat
from ...number.context.real import REAL_FORMAT
from .types import CppList, CppScalar, CppTuple, CppType

# ----------------------------------------------------------------------
# The storage ladder.
#
# Each entry pairs a CppScalar with the AbstractFormat of the smallest
# context that fits in that scalar.  ``choose_storage_scalar`` walks the
# ladder in order and picks the first entry whose AbstractFormat
# contains the inferred bound.  Order matters: smaller types first.

def _af(fmt: AbstractableFormat) -> AbstractFormat:
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


_LADDER_LOOKUP = {ty: af for ty, af in _LADDER}


def scalar_fits_in(a: CppScalar, b: CppScalar) -> bool:
    """Does scalar *a* fit inside scalar *b*?

    ``BOOL`` only fits itself; the rest dispatch to ladder containment, which
    gives the standard inclusions plus integer-to-float where the mantissa is
    wide enough.
    """
    if a is CppScalar.BOOL or b is CppScalar.BOOL:
        return a is b
    return _LADDER_LOOKUP[a] <= _LADDER_LOOKUP[b]


def bound_fits_in_scalar(bound: FormatBound, ty: CppScalar) -> bool:
    """Is every value *bound* admits representable in *ty*?

    A question about values, where :func:`scalar_fits_in` asks about types --
    and the two disagree, because a bound picks the *smallest* type holding it:
    ``1`` stores as ``uint8_t``, which does not nest into ``int8_t``.
    """
    if ty is CppScalar.BOOL:
        return False
    if not isinstance(bound, AbstractableFormat | SetFormat):
        return False
    af = _to_abstract(bound)
    return af is not None and af <= _LADDER_LOOKUP[ty]


class StorageSelectionError(Exception):
    """Raised when no storage type contains the inferred format."""


def choose_storage_scalar(bound: FormatBound) -> CppScalar:
    """The smallest scalar storage containing *bound*.

    ``None`` -- a non-numeric bound, e.g. a comparison -- is ``BOOL``;
    ``REAL_FORMAT`` raises, since no finite ladder entry covers all reals.
    """
    if bound is None:
        return CppScalar.BOOL
    if bound == REAL_FORMAT:
        raise StorageSelectionError(
            'cannot store an unconstrained real value in a finite C++ type; '
            'is the active rounding context symbolic? '
            'Try monomorphizing the function with a concrete context.'
        )
    if not isinstance(bound, AbstractableFormat | SetFormat):
        raise StorageSelectionError(
            f'cannot reason about format: {bound!r}'
        )
    if is_bottom(bound):
        # A slot holding no value -- an element of a fresh `empty(...)`.  Every
        # rung contains it vacuously, so the smallest wins.  `_to_abstract`
        # cannot serve this: every format represents a `+0.0`, so none *is* the
        # empty set.
        return _LADDER[0][0]

    af = _to_abstract(bound)
    if af is None:
        raise StorageSelectionError(
            f'cannot lift {bound!r} to AbstractFormat; '
            'storage selection requires a dyadic format'
        )
    for cpp_ty, ladder_af in _LADDER:
        if af <= ladder_af:
            return cpp_ty
    if (isinstance(bound, MPFixedFormat) and bound.expmin >= 0
            and af.specials_contained_in(_LADDER_LOOKUP[CppScalar.S64])):
        # Deliberately ignores the *magnitude* bound -- an unbounded integer has
        # none, and overflow is the user's problem (see the docstring above).  It
        # must not ignore the membership flags too: `int64_t` holds no NaN, no
        # infinity and no signed zero, so a bound carrying one of those has no
        # business here even though the ladder search already rejected it.
        return CppScalar.S64
    raise StorageSelectionError(
        f'no storage type on the ladder contains {bound!r}'
    )


def choose_storage(bound: FormatBound) -> CppType:
    """The storage for a possibly structured :class:`FormatBound`: scalars via
    :func:`choose_storage_scalar`, tuples to ``std::tuple``, lists to
    ``std::vector``.
    """
    if isinstance(bound, TupleFormat):
        return CppTuple(tuple(choose_storage(b) for b in bound.elts))
    if isinstance(bound, ListFormat):
        return CppList(choose_storage(bound.elt))
    return choose_storage_scalar(bound)


def aggregate_storage(bounds: list[FormatBound]) -> CppType:
    """A single storage type containing every bound in *bounds*.

    For a name with several SSA defs, whose declaration must hold every value
    assigned into it.  Storage per bound, then the ladder supremum; structured
    types recurse.

    A bottom bound (a fresh ``empty(...)``) holds no value, so it constrains
    nothing and is dropped when any other def does -- keeping it would widen
    for nothing, since its storage is the first rung and ``u8 ⊔ s8`` is
    ``s16``.  A *partly* bottom bound still contributes its empty slots; fixing
    that needs a supremum over bounds rather than over storages.
    """
    assert bounds, 'aggregate_storage requires at least one bound'
    constraining = [b for b in bounds if not is_bottom(b)]
    storages = [choose_storage(b) for b in (constraining or bounds)]
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
        return scalar_sup([head] + [s for s in rest if isinstance(s, CppScalar)])
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


def scalar_sup(scalars: list[CppScalar]) -> CppScalar:
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
        ) from None
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
