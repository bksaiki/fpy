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
- :func:`scalar_sup` — the first rung subsuming every input.  N-ary: see its
  docstring for why it must not be folded.  Used by :meth:`_visit_compare` to
  pick the common type for a chained comparison.
- :func:`bound_fits_in_scalar` — value-level containment, for a
  conversion the type-level test refuses.
- :func:`exact_integer_bits` — a float rung's exact-integer width, for
  diagnosing a refused conversion.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING

from ...analysis import Definition
from ...analysis.format_infer import (
    AbstractableFormat,
    AbstractFormat,
    FormatBound,
    ListFormat,
    SetFormat,
    TupleFormat,
    VarFormat,
    is_bottom,
)
from ...analysis.format_infer.analysis import _to_abstract
from ...ast.fpyast import Var
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

if TYPE_CHECKING:
    from .storage_infer import StorageAnalysis

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


_SIGMA: tuple[tuple[CppScalar, AbstractableFormat], ...] = (
    (CppScalar.U8, UINT8.format()),
    (CppScalar.S8, SINT8.format()),
    (CppScalar.U16, UINT16.format()),
    (CppScalar.S16, SINT16.format()),
    (CppScalar.U32, UINT32.format()),
    (CppScalar.S32, SINT32.format()),
    (CppScalar.F32, FP32.format()),
    (CppScalar.U64, UINT64.format()),
    (CppScalar.S64, SINT64.format()),
    (CppScalar.F64, FP64.format()),
)
"""The storage domain: an ordered sequence of *formats*, smallest first, each
paired with the C++ type that spells it.

A storage type is a format the target can spell -- ``CppScalar.S64`` names
exactly ``SINT64.format()`` -- so containment, widening and losslessness are all
format questions and need no separate vocabulary.

**Ordered, not a set.**  Containment over these formats is not a
join-semilattice: ``{s8, u16}`` has two incomparable minimal upper bounds,
``s32`` and ``f32``, and no least one.  The sequence is therefore the tie-break,
and a different order changes which programs compile -- preferring the integer
rung for ``{s8, u16}`` is what makes a later join with ``float`` fail.
"""


_ABSTRACT: dict[CppScalar, AbstractFormat] = {
    ty: _af(fmt) for ty, fmt in _SIGMA
}
"""Each rung lifted for comparison.  ``AbstractFormat`` is what carries ``<=``."""


def scalar_fits_in(a: CppScalar, b: CppScalar) -> bool:
    """Does scalar *a* fit inside scalar *b*?

    ``BOOL`` only fits itself; the rest dispatch to ladder containment, which
    gives the standard inclusions plus integer-to-float where the mantissa is
    wide enough.
    """
    if a is CppScalar.BOOL or b is CppScalar.BOOL:
        return a is b
    return _ABSTRACT[a] <= _ABSTRACT[b]


def exact_integer_bits(ty: CppScalar) -> int | None:
    """How wide an integer float *ty* holds exactly -- its significand -- or
    ``None`` for a non-float rung, which has no such limit.  Used to explain a
    refused integer-to-float conversion."""
    return int(_ABSTRACT[ty].prec) if ty.is_float() else None


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
    return af is not None and af <= _ABSTRACT[ty]


class StorageSelectionError(Exception):
    """Raised when no storage type contains the inferred format."""


def choose_storage_scalar(bound: FormatBound) -> CppScalar:
    """The smallest scalar storage containing *bound*.

    ``None`` -- a non-numeric bound, e.g. a comparison -- is ``BOOL``;
    ``REAL_FORMAT`` raises, since no finite ladder entry covers all reals.

    :class:`VarFormat` is ``BOOL`` as well, for a different reason: an
    unresolved kind has no observable storage, so any choice compiles.  Where it
    *is* observable -- in the signature -- ``_check_signature_monomorphic``
    refuses the spec before this runs.
    """
    if bound is None or isinstance(bound, VarFormat):
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
        return _SIGMA[0][0]

    af = _to_abstract(bound)
    if af is None:
        raise StorageSelectionError(
            f'cannot lift {bound!r} to AbstractFormat; '
            'storage selection requires a dyadic format'
        )
    for cpp_ty, _ in _SIGMA:
        if af <= _ABSTRACT[cpp_ty]:
            return cpp_ty
    if (isinstance(bound, MPFixedFormat) and bound.expmin >= 0
            and af.specials_contained_in(_ABSTRACT[CppScalar.S64])):
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
    """The first rung of :data:`_SIGMA` containing every input.

    **N-ary, and it must not be folded.**  Containment is not a
    join-semilattice, so a pairwise fold is both less precise and less total
    than one search over the whole set: ``sup([s8, u16, f32])`` is ``float``,
    where folding gives ``double`` -- ``sup([s8, u16])`` picks ``s32``, which no
    ``float`` holds -- and ``sup([s8, u32, f32])`` is ``double`` where folding
    fails outright.
    """
    if any(s is CppScalar.BOOL for s in scalars):
        # Mixing bool with numeric storage is a typing bug, not a widening.
        if all(s is CppScalar.BOOL for s in scalars):
            return CppScalar.BOOL
        raise StorageSelectionError(
            f'cannot widen across BOOL and numeric storage: {scalars!r}'
        )
    try:
        afs = [_ABSTRACT[s] for s in scalars]
    except KeyError as e:
        raise StorageSelectionError(
            f'storage scalar not on the ladder: {e.args[0]!r}'
        ) from None
    for ty, _ in _SIGMA:
        rung = _ABSTRACT[ty]
        if all(af <= rung for af in afs):
            return ty
    raise StorageSelectionError(
        f'no storage type on the ladder subsumes {scalars!r}'
    )


class CppStorageDomain:
    """The cpp backend's :class:`~.storage_infer.StorageDomain`: the ladder.

    Holds no state -- the domain *is* :data:`_SIGMA`.
    """

    @property
    def sigma(self) -> Sequence[AbstractableFormat]:
        return [fmt for _ty, fmt in _SIGMA]

    def fallback(self, bound: FormatBound) -> AbstractableFormat | None:
        """``int64_t`` for an unbounded integer format.

        Deliberately ignores the *magnitude* bound -- an unbounded integer has
        none, and overflow is the user's problem.  It must not ignore the
        membership flags too: ``int64_t`` holds no NaN, no infinity and no signed
        zero, so a bound carrying one has no business here even though the
        containment search already rejected it.
        """
        if not (isinstance(bound, MPFixedFormat) and bound.expmin >= 0):
            return None
        af = _to_abstract(bound)
        if af is None or not af.specials_contained_in(_ABSTRACT[CppScalar.S64]):
            return None
        return SINT64.format()


_SPELLING: dict[AbstractableFormat, CppScalar] = {fmt: ty for ty, fmt in _SIGMA}


def to_cpp(storage: FormatBound) -> CppType:
    """A storage the analysis chose, as the C++ type that spells it.

    The translation the backend owes: the analysis answers in formats, which say
    what a value *is*; a ``CppType`` says how this target holds one, and carries
    a representation axis (a handle, a value, a fixed length) that no format has.
    :mod:`.unbox` decides that axis and stamps it afterwards.
    """
    if storage is None:
        return CppScalar.BOOL
    if isinstance(storage, TupleFormat):
        return CppTuple(tuple(to_cpp(e) for e in storage.elts))
    if isinstance(storage, ListFormat):
        return CppList(to_cpp(storage.elt))
    spelled = (
        _SPELLING.get(storage)
        if isinstance(storage, AbstractableFormat) else None
    )
    if spelled is None:
        raise StorageSelectionError(
            f'no C++ type spells the storage {storage!r}'
        )
    return spelled


class CppStorage:
    """The analysis's answer, spelled in C++ types.

    A thin view: the analysis chose formats, this maps them once so the emitter
    reads types.  ``class_storage`` is the mutable one -- :mod:`.unbox` rewrites
    an entry to record a representation, which is a fact about how the target
    holds a value and not one the analysis has any business carrying.
    """

    def __init__(self, analysis: 'StorageAnalysis'):
        self.analysis = analysis
        self.class_storage: dict[Definition, CppType] = {
            c: to_cpp(fmt) for c, fmt in analysis.class_storage.items()
        }

    @property
    def def_class(self):
        return self.analysis.def_class

    @property
    def class_members(self):
        return self.analysis.class_members

    def storage_of(self, d: Definition) -> CppType:
        return self.class_storage[self.analysis.def_class[d]]

    def of_expr(self, e) -> CppType | None:
        """:meth:`StorageAnalysis.of_expr`, spelled.

        A ``Var`` resolves through :attr:`class_storage`, so it sees whatever
        representation :mod:`.unbox` stamped there; anything else is translated
        fresh.
        """
        if isinstance(e, Var):
            return self.storage_of(self.analysis.def_use.find_def_from_use(e))
        chosen = self.analysis.of_expr(e)
        if chosen is None and self.analysis.expr_bound.get(e) is not None:
            return None
        try:
            return to_cpp(chosen)
        except StorageSelectionError:
            return None

    def is_aggregate(self, storage: CppType) -> bool:
        return isinstance(storage, (CppList, CppTuple))
