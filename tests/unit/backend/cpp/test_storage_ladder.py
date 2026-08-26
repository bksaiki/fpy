"""The storage ladder is an ordered sequence, and its join is n-ary.

Containment over the ladder is *not* a join-semilattice: `{s8, u16}` has two
incomparable minimal upper bounds, `s32` and `f32`, and no least one.  Two
consequences a refactor could easily undo, so both are pinned here.
"""

from functools import reduce
from itertools import combinations, permutations

import pytest

from fpy2.backend.cpp.storage import (
    _ABSTRACT,
    _SIGMA,
    StorageSelectionError,
    scalar_sup,
)

_TYS = [t for t, _ in _SIGMA]


def _sup(xs):
    try:
        return scalar_sup(list(xs))
    except StorageSelectionError:
        return None


class TestTheLadderIsOrdered:
    def test_it_is_a_linear_extension_of_containment(self):
        """`ceil` takes the first containing rung, so a rung contained in an
        earlier one would never be reached."""
        idx = {t: i for i, (t, _) in enumerate(_SIGMA)}
        for a in _TYS:
            for b in _TYS:
                if a is not b and _ABSTRACT[b] <= _ABSTRACT[a]:
                    assert idx[b] < idx[a], (
                        f'{b.format()} fits in {a.format()} but comes later'
                    )

    def test_minimal_upper_bounds_are_not_unique(self):
        """Which is why the sequence is the tie-break rather than a detail."""
        from fpy2.backend.cpp.types import CppScalar as C
        assert not _ABSTRACT[C.U8] <= _ABSTRACT[C.S8]
        assert not _ABSTRACT[C.S8] <= _ABSTRACT[C.U8]


class TestTheJoinIsNAry:
    """Folding pairwise is both less precise and less total."""

    def test_folding_can_overshoot(self):
        from fpy2.backend.cpp.types import CppScalar as C
        combo = [C.S8, C.U16, C.F32]
        assert _sup(combo) is C.F32
        folded = reduce(lambda a, b: scalar_sup([a, b]), combo)
        assert folded is C.F64

    def test_folding_can_fail_where_the_join_succeeds(self):
        from fpy2.backend.cpp.types import CppScalar as C
        combo = [C.S8, C.U32, C.F32]
        assert _sup(combo) is C.F64
        with pytest.raises(StorageSelectionError):
            reduce(lambda a, b: scalar_sup([a, b]), combo)

    def test_the_join_is_order_independent(self):
        """Unlike the fold, which depends on which pair is taken first."""
        for combo in combinations(_TYS, 3):
            answers = {_sup(p) for p in permutations(combo)}
            assert len(answers) == 1, (combo, answers)
