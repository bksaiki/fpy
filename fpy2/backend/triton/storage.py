"""
triton backend: storage-type selection.

The backend's half of storage inference: the ordered set of formats Triton can
spell (:data:`_SIGMA`), the :class:`StorageDomain` presenting it to
:mod:`fpy2.analysis.storage_infer`, and the translation from a chosen format
into a :class:`TritonType`.

**The float rungs are a chain.**  fp16 (prec 11) nests in fp32 (prec 24) nests
in fp64 (prec 53), because ``bf16`` -- which is incomparable with fp16 -- is out
of scope.  The ladder is still not a join-semilattice overall, since the integer
rungs remain incomparable with the float ones exactly as in C++, so the sequence
is still the tie-break and its order still decides which programs compile.
"""

from collections.abc import Sequence

from ...analysis.format_infer import (
    AbstractableFormat,
    AbstractFormat,
    FormatBound,
    ListFormat,
    SetFormat,
    TupleFormat,
)
from ...analysis.format_infer.analysis import _to_abstract
from ...analysis.storage_infer import (
    StorageSelectionError,
    join,
    of_bound,
)
from ...analysis.value_class import ClassBound
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
from .types import TritonScalar, TritonTuple, TritonType


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
"""The storage domain: an ordered sequence of *formats*, smallest first, each
paired with the Triton dtype that spells it.

Two placements are decisions rather than consequences.

``F16`` sits **after** ``S16``, not before ``U16``.  It has to follow ``U8`` and
``S8``, which nest in it, but against the 16-bit integer rungs it is
incomparable -- prec 11 does not hold ``u16``, and no integer type holds a
fraction.  Placing it later means a bound like "integers in [0, 2000]" takes
``u16`` rather than ``f16``, even though fp16 represents every such value
exactly.  That mirrors the cpp ladder's choice to put ``F32`` after ``S32``, and
the reason is the same: a count is not a float.

``F16`` is on the ladder at all, which the cpp ladder has no equivalent of.  It
is the rung that lets a kernel take an fp16 buffer instead of widening its
parameters at the boundary.
"""


_ABSTRACT: dict[TritonScalar, AbstractFormat] = {
    ty: _af(fmt) for ty, fmt in _SIGMA
}
"""Each rung lifted for comparison.  ``AbstractFormat`` is what carries ``<=``."""


def scalar_fits_in(a: TritonScalar, b: TritonScalar) -> bool:
    """Does scalar *a* fit inside scalar *b*?

    ``BOOL`` only fits itself; the rest dispatch to ladder containment.

    This is also the predicate that decides whether fused multiply-add is
    observable: contraction is unobservable exactly where the product's format
    fits the storage chosen for it, since "round the product then round the
    sum" and "round the sum of the exact product" then agree.  See
    measured against hardware in ``exploration/triton/``.
    """
    if a is TritonScalar.BOOL or b is TritonScalar.BOOL:
        return a is b
    return _ABSTRACT[a] <= _ABSTRACT[b]


def exact_integer_bits(ty: TritonScalar) -> int | None:
    """How wide an integer float *ty* holds exactly -- its significand -- or
    ``None`` for a non-float rung, which has no such limit."""
    return int(_ABSTRACT[ty].prec) if ty.is_float() else None


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
    """The triton backend's :class:`~.storage_infer.StorageDomain`.

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


def to_triton(storage: FormatBound) -> TritonType:
    """A storage the analysis chose, as the Triton type that spells it.

    ``None`` means **not real-valued**: format inference covers real-valued
    expressions and structures of them, and gives no format for anything else.
    A boolean is spelled; the other case is a rounding context, which is a
    foreign value the emitter refuses rather than storing.

    A ``ListFormat`` is refused rather than spelled.  Triton has no list: a
    proven-length list unrolls into one value per element before reaching here,
    and an unproven-length one is out of scope until §8 of
    Refusing names the reason; spelling it as
    a tile would silently change what the program means.
    """
    if storage is None:
        return TritonScalar.BOOL
    if isinstance(storage, TupleFormat):
        return TritonTuple(tuple(to_triton(e) for e in storage.elts))
    if isinstance(storage, ListFormat):
        raise StorageSelectionError(
            'the triton backend has no list storage: a list whose length is '
            'proven unrolls into registers, and one whose length is not is '
            'not yet supported'
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


def choose_storage(bound: FormatBound, cls: ClassBound = None) -> TritonType:
    """The storage containing *bound*, spelled.

    The search lives in the analysis -- the sequence is a tie-break, not a
    presentation order -- so the backend asks rather than walking the ladder.
    """
    return to_triton(of_bound(TritonStorageDomain(), bound, cls))


def choose_storage_scalar(bound: FormatBound) -> TritonScalar:
    """The scalar storage containing *bound*, spelled.

    A convenience for the op tables, which reason about a context's format
    rather than about a definition's class.
    """
    ty = choose_storage(bound)
    if not isinstance(ty, TritonScalar):
        raise StorageSelectionError(f'expected a scalar storage, got {ty!r}')
    return ty


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
    sup = join(TritonStorageDomain(), [formats[s] for s in scalars])
    spelled = to_triton(sup)
    assert isinstance(spelled, TritonScalar), spelled
    return spelled
