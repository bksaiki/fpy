"""The Triton storage ladder: what its order buys, and what scope bought.

The cpp ladder's counterpart test pins that containment is not a
join-semilattice and that the sequence is therefore a tie-break.  Both hold
here too.  What is *different* is that dropping `bf16` makes the **float**
rungs a chain, which is the property that closed the one open design question
in this module -- so it is pinned rather than left to be rediscovered.
"""

import pytest

import fpy2 as fp
from fpy2.analysis.storage_infer import StorageSelectionError
from fpy2.backend.triton.storage import (
    _ABSTRACT,
    _SIGMA,
    scalar_fits_in,
    scalar_sup,
)
from fpy2.backend.triton.types import FLOAT_TYPES, TritonScalar as T

_TYS = [t for t, _ in _SIGMA]
_IDX = {t: i for i, (t, _) in enumerate(_SIGMA)}


class TestTheLadderIsOrdered:
    def test_it_is_a_linear_extension_of_containment(self):
        """The search takes the first containing rung, so a rung contained in
        an earlier one would never be reached."""
        for a in _TYS:
            for b in _TYS:
                if a is not b and _ABSTRACT[b] <= _ABSTRACT[a]:
                    assert _IDX[b] < _IDX[a], (
                        f'{b.format()} fits in {a.format()} but comes later'
                    )

    def test_minimal_upper_bounds_are_still_not_unique(self):
        """Integer rungs stay mutually incomparable, so the sequence is still
        the tie-break -- adding fp16 did not make the ladder a lattice."""
        assert not _ABSTRACT[T.U8] <= _ABSTRACT[T.S8]
        assert not _ABSTRACT[T.S8] <= _ABSTRACT[T.U8]


class TestTheFloatRungsAreAChain:
    """What dropping `bf16` bought.

    bf16 (es 8, prec 8) and fp16 (es 5, prec 11) are mutually incomparable, so
    with both on the ladder their relative order would decide which programs
    compile and neither order is obviously right.  With bf16 out of scope the
    float rungs nest totally and the question does not arise.
    """

    def test_floats_nest_totally(self):
        for a in FLOAT_TYPES:
            for b in FLOAT_TYPES:
                assert scalar_fits_in(a, b) or scalar_fits_in(b, a), (
                    f'{a.format()} and {b.format()} are incomparable'
                )

    def test_the_chain_is_f16_f32_f64(self):
        assert scalar_fits_in(T.F16, T.F32)
        assert scalar_fits_in(T.F32, T.F64)
        assert not scalar_fits_in(T.F32, T.F16)

    @pytest.mark.parametrize('ctx', [fp.BF16, fp.S1E4M3, fp.S1E5M2, fp.MX_E4M3])
    def test_narrow_and_incomparable_formats_are_not_storage(self, ctx):
        """Out of scope by decision, not by oversight.  They stay reachable as
        rounding targets through `unfold_round`; they are never storage."""
        assert ctx.format() not in [fmt for _t, fmt in _SIGMA]


class TestF16Placement:
    """`F16` sits after `S16`, and that is a choice.

    It must follow `U8`/`S8`, which nest in it.  Against the 16-bit integers it
    is incomparable, so the placement is free -- and putting it later means an
    integral bound prefers an integer rung.  Same reasoning as the cpp ladder
    putting `F32` after `S32`: a count is not a float.
    """

    def test_f16_follows_the_8_bit_integers(self):
        assert _IDX[T.U8] < _IDX[T.F16]
        assert _IDX[T.S8] < _IDX[T.F16]

    def test_f16_follows_the_16_bit_integers(self):
        assert _IDX[T.U16] < _IDX[T.F16]
        assert _IDX[T.S16] < _IDX[T.F16]

    def test_f16_is_incomparable_with_u16(self):
        """Which is why the previous test is a decision and not a consequence."""
        assert not scalar_fits_in(T.U16, T.F16)
        assert not scalar_fits_in(T.F16, T.U16)


class TestTheJoinIsNAry:
    def test_widening_across_the_incomparable_pair(self):
        assert scalar_sup([T.S8, T.U16]) is T.S32

    def test_bool_joins_only_with_itself(self):
        assert scalar_sup([T.BOOL, T.BOOL]) is T.BOOL
        with pytest.raises(StorageSelectionError):
            scalar_sup([T.BOOL, T.F32])
