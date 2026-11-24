"""Storage ladders and native integer contexts shared by the backends."""

import enum
from collections.abc import Sequence

from ..analysis.format_infer import (
    AbstractableFormat,
    AbstractFormat,
    FormatBound,
    to_abstract,
)
from ..analysis.storage_infer import StorageSelectionError, join, without_absent
from ..analysis.value_class import ValueClass
from ..number import (
    INTEGER,
    SINT8,
    SINT16,
    SINT32,
    SINT64,
    UINT8,
    UINT16,
    UINT32,
    UINT64,
)
from ..number.context.context import Context
from ..number.context.mp_fixed import MPFixedFormat

INT_CTXS: tuple[Context, ...] = (
    SINT8, SINT16, SINT32, SINT64,
    UINT8, UINT16, UINT32, UINT64,
    INTEGER,
)
"""The integer contexts the backends dispatch on; ``INTEGER`` is stored as ``int64``."""


def _af(fmt: AbstractableFormat) -> AbstractFormat:
    af = AbstractFormat.from_format(fmt)
    assert af is not None, f'expected abstractable format, got {fmt!r}'
    return af


_INT64 = _af(SINT64.format())


class Ladder[T: enum.Enum]:
    """A :class:`~fpy2.analysis.StorageDomain` over the scalars of a target.

    ``bool_ty`` is off the ladder and fits only itself.
    """

    formats: dict[T, AbstractableFormat]
    """Each rung's format, narrowest first."""
    bool_ty: T
    abstract: dict[T, AbstractFormat]
    """Each rung lifted for comparison: ``AbstractFormat`` carries ``<=``."""
    spelling: dict[AbstractableFormat, T]

    def __init__(self, rungs: tuple[tuple[T, AbstractableFormat], ...], bool_ty: T) -> None:
        self.formats = dict(rungs)
        self.bool_ty = bool_ty
        self.abstract = {ty: _af(fmt) for ty, fmt in rungs}
        self.spelling = {fmt: ty for ty, fmt in rungs}

    @property
    def sigma(self) -> Sequence[AbstractableFormat]:
        return list(self.formats.values())

    def fallback(self, bound: FormatBound) -> AbstractableFormat | None:
        """``int64`` for an unbounded integer format.

        Ignores the *magnitude* bound -- an unbounded integer has none, and
        overflow is the user's problem -- but not the membership flags, since
        ``int64`` holds no NaN, no infinity and no signed zero.
        """
        if not (isinstance(bound, MPFixedFormat) and bound.expmin >= 0):
            return None
        af = to_abstract(bound)
        if af is None or not af.specials_contained_in(_INT64):
            return None
        return SINT64.format()

    def spell(self, storage: FormatBound) -> T | None:
        """The rung spelling *storage*, if any."""
        return self.spelling.get(storage) if isinstance(storage, AbstractableFormat) else None

    def fits_in(self, a: T, b: T) -> bool:
        """Does scalar *a* fit inside scalar *b*?"""
        if a is self.bool_ty or b is self.bool_ty:
            return a is b
        return self.abstract[a] <= self.abstract[b]

    def bound_fits_in(self, bound: FormatBound, ty: T, cls: ValueClass | None = None) -> bool:
        """Is every value *bound* admits representable in *ty*?

        A question about values, where :meth:`fits_in` asks about types -- and
        the two disagree, because a bound picks the *smallest* type holding it:
        ``1`` stores as ``u8``, which does not nest into ``s8``.

        *cls* rules out the special values it proves absent, as storage selection
        does (:func:`~fpy2.analysis.storage_infer.without_absent`).
        """
        if ty is self.bool_ty:
            return False
        af = to_abstract(bound)
        if af is None:
            return False
        if cls is not None:
            af = without_absent(af, cls)
        return af <= self.abstract[ty]

    def sup(self, scalars: list[T]) -> T:
        """:func:`~fpy2.analysis.storage_infer.join` over scalars.

        ``bool_ty`` joins only with itself; mixing it with numeric storage is
        a typing bug, not a widening.
        """
        if self.bool_ty in scalars:
            if all(s is self.bool_ty for s in scalars):
                return self.bool_ty
            raise StorageSelectionError(
                f'cannot widen across BOOL and numeric storage: {scalars!r}'
            )
        missing = [s for s in scalars if s not in self.formats]
        if missing:
            raise StorageSelectionError(f'storage scalar not on the ladder: {missing[0]!r}')
        spelled = self.spell(join(self, [self.formats[s] for s in scalars]))
        assert spelled is not None
        return spelled
