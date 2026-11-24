"""
Triton backend: storage-type selection.

The backend's half of storage inference: the ordered set of formats Triton can
spell (:data:`_SIGMA`), the :class:`StorageDomain` presenting it to
:mod:`fpy2.analysis.storage_infer`, and the translation from a chosen format
into a :class:`TritonScalar`.

The float rungs are a chain, but the integer rungs are incomparable with them,
as in C++, so the ladder is not a join-semilattice: its order is the tie-break.
"""

from ...analysis.format_infer import (
    AbstractableFormat,
    FormatBound,
    ListFormat,
    SetFormat,
    to_abstract,
)
from ...analysis.storage_infer import StorageSelectionError, of_bound
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
from ..storage import Ladder
from .types import TritonScalar

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


class TritonStorageDomain(Ladder[TritonScalar]):
    """The Triton backend's :class:`~.storage_infer.StorageDomain`: :data:`_SIGMA`."""

    def __init__(self) -> None:
        super().__init__(_SIGMA, TritonScalar.BOOL)


DOMAIN = TritonStorageDomain()
scalar_fits_in = DOMAIN.fits_in
scalar_sup = DOMAIN.sup


def bound_fits_in_scalar(bound: FormatBound, ty: TritonScalar) -> bool:
    """Is every value *bound* admits representable in *ty*?

    A question about values, where :func:`scalar_fits_in` asks about types.
    The empty set, a body that never runs, fits in any.
    """
    if isinstance(bound, SetFormat) and not bound.values:
        return True
    return DOMAIN.bound_fits_in(bound, ty)


def index_scalar(bound: FormatBound) -> TritonScalar | None:
    """The narrowest of ``tl.int32`` and ``tl.int64`` holding every value
    *bound* admits, or ``None`` (a value may not be an integer, or none fits)."""
    af = to_abstract(bound)
    if af is None:
        return None
    af = af.replace(has_neg_zero=False)  # -0 indexes as 0
    return next((ty for ty in (TritonScalar.S32, TritonScalar.S64) if af <= DOMAIN.abstract[ty]), None)


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
    spelled = DOMAIN.spell(storage)
    if spelled is None:
        raise StorageSelectionError(f'no Triton dtype spells the storage {storage!r}')
    return spelled


def choose_storage_scalar(bound: FormatBound) -> TritonScalar:
    """The storage containing *bound*, spelled.

    The search lives in the analysis -- the sequence is a tie-break, not a
    presentation order -- so the backend asks rather than walking the ladder.
    """
    return to_triton(of_bound(DOMAIN, bound))
