"""
Triton backend: storage-type selection.

The backend's half of storage inference: the ordered set of formats Triton can
spell (:data:`_SIGMA`), the :class:`StorageDomain` presenting it to
:mod:`fpy2.analysis.storage_infer`, and the translation from a chosen format
into a :class:`TritonScalar`.

The float rungs are a chain, but the integer rungs are incomparable with them,
as in C++, so the ladder is not a join-semilattice: its order is the tie-break.
"""

from collections.abc import Sequence

from ...analysis.format_infer import (
    AbstractableFormat,
    AbstractFormat,
    FormatBound,
    ListFormat,
    SetFormat,
)
from ...analysis.format_infer.analysis import _to_abstract
from ...analysis.storage_infer import (
    StorageSelectionError,
    join,
    of_bound,
)
from ...number import (
    FP16,
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
from .types import TritonScalar


def _af(fmt: AbstractableFormat) -> AbstractFormat:
    af = AbstractFormat.from_format(fmt)
    assert af is not None, f'expected abstractable format, got {fmt!r}'
    return af


_SIGMA: tuple[tuple[TritonScalar, AbstractableFormat], ...] = (
    (TritonScalar.U8, UINT8.format()),
    (TritonScalar.S8, SINT8.format()),
    (TritonScalar.U16, UINT16.format()),
    (TritonScalar.S16, SINT16.format()),
    (TritonScalar.F16, FP16.format()),
    (TritonScalar.U32, UINT32.format()),
    (TritonScalar.S32, SINT32.format()),
    (TritonScalar.F32, FP32.format()),
    (TritonScalar.U64, UINT64.format()),
    (TritonScalar.S64, SINT64.format()),
    (TritonScalar.F64, FP64.format()),
)
"""The storage domain: each format, smallest first, with the Triton dtype
that spells it.

``F16`` follows ``S16``, with which it is incomparable, so integers in
[0, 2000] take ``u16`` rather than ``f16``: a count is not a float.  The cpp
ladder puts ``F32`` after ``S32`` for the same reason.
"""


_ABSTRACT: dict[TritonScalar, AbstractFormat] = {
    ty: _af(fmt) for ty, fmt in _SIGMA
}
"""Each rung as an ``AbstractFormat``, which carries ``<=``."""


def scalar_fits_in(a: TritonScalar, b: TritonScalar) -> bool:
    """Does scalar *a* fit inside scalar *b*?  ``BOOL`` fits only itself."""
    if a is TritonScalar.BOOL or b is TritonScalar.BOOL:
        return a is b
    return _ABSTRACT[a] <= _ABSTRACT[b]


def bound_fits_in_scalar(bound: FormatBound, ty: TritonScalar) -> bool:
    """Is every value *bound* admits representable in *ty*?

    A question about values, where :func:`scalar_fits_in` asks about types.
    """
    if ty is TritonScalar.BOOL:
        return False
    if not isinstance(bound, AbstractableFormat | SetFormat):
        return False
    af = _to_abstract(bound)
    return af is not None and af <= _ABSTRACT[ty]


class TritonStorageDomain:
    """The Triton backend's :class:`~.storage_infer.StorageDomain`.

    Holds no state -- the domain *is* :data:`_SIGMA`.
    """

    @property
    def sigma(self) -> Sequence[AbstractableFormat]:
        return [fmt for _ty, fmt in _SIGMA]

    def fallback(self, bound: FormatBound) -> AbstractableFormat | None:
        """``tl.int64`` for an unbounded integer format, as in the cpp backend.

        Ignores the *magnitude* bound -- an unbounded integer has none, and
        overflow is the user's problem -- but not the membership flags, since
        ``int64`` holds no NaN, no infinity and no signed zero.
        """
        if not (isinstance(bound, MPFixedFormat) and bound.expmin >= 0):
            return None
        af = _to_abstract(bound)
        if af is None or not af.specials_contained_in(_ABSTRACT[TritonScalar.S64]):
            return None
        return SINT64.format()


_SPELLING: dict[AbstractableFormat, TritonScalar] = {fmt: ty for ty, fmt in _SIGMA}


def to_triton(storage: FormatBound) -> TritonScalar:
    """A storage the analysis chose, as the Triton type that spells it.

    ``None`` means not real-valued, which is spelled as a boolean; the other
    such values, rounding contexts, the emitter refuses before asking.  A
    ``ListFormat`` is refused: a list is stored by element.
    """
    if storage is None:
        return TritonScalar.BOOL
    if isinstance(storage, ListFormat):
        raise StorageSelectionError(
            'the Triton backend has no list storage: a list is stored by element'
        )
    spelled = (
        _SPELLING.get(storage)
        if isinstance(storage, AbstractableFormat) else None
    )
    if spelled is None:
        raise StorageSelectionError(
            f'no Triton dtype spells the storage {storage!r}'
        )
    return spelled


def choose_storage_scalar(bound: FormatBound) -> TritonScalar:
    """The storage containing *bound*, spelled.

    The search lives in the analysis -- the sequence is a tie-break, not a
    presentation order -- so the backend asks rather than walking the ladder.
    """
    return to_triton(of_bound(TritonStorageDomain(), bound))


def scalar_sup(scalars: list[TritonScalar]) -> TritonScalar:
    """:func:`~.storage_infer.join` over scalars, spelled in Triton terms.

    ``BOOL`` is not on the ladder and joins only with itself; the rest defer, so
    the n-ary search lives in exactly one place.
    """
    if any(s is TritonScalar.BOOL for s in scalars):
        if all(s is TritonScalar.BOOL for s in scalars):
            return TritonScalar.BOOL
        raise StorageSelectionError(
            f'cannot widen across BOOL and numeric storage: {scalars!r}'
        )
    formats = dict(_SIGMA)
    missing = [s for s in scalars if s not in formats]
    if missing:
        raise StorageSelectionError(
            f'storage scalar not on the ladder: {missing[0]!r}'
        )
    return to_triton(join(TritonStorageDomain(), [formats[s] for s in scalars]))
