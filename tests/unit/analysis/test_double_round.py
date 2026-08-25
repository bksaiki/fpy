"""
Figure 8 of *When Double Rounding is Correct*, as a predicate.

The oracle is the Lean development, `Mpfx/DoubleRounding.lean
<https://github.com/bksaiki/mpfx-lean/blob/main/Mpfx/DoubleRounding.lean>`_.
Each admitted pairing names the theorem it comes from; a pairing not listed has
no theorem and must be refused.
"""

import itertools
import math

import pytest

import fpy2 as fp
from fpy2.analysis.format_infer import (
    AbstractableFormat,
    AbstractFormat,
    derive_intermediate,
    double_round_ok,
)
from fpy2.number import RoundingMode as RM

# (final rm1, intermediate rm2) -> the theorem admitting it
ADMITTED: dict[tuple[RM, RM], str] = {
    (RM.RTZ, RM.RTZ): 'rndRTZ_RTZ',
    (RM.RAZ, RM.RAZ): 'rndRAZ_RAZ',
    (RM.RTP, RM.RTP): 'rndRTP_RTP',
    (RM.RTN, RM.RTN): 'rndRTN_RTN',
    (RM.RTO, RM.RTO): 'rndRTO_RTO',
    (RM.RTZ, RM.RTO): 'rndRTO_RTZ',
    (RM.RAZ, RM.RTO): 'rndRTO_RAZ',
    (RM.RNE, RM.RTO): 'rndRTO_RN',
    (RM.RNA, RM.RTO): 'rndRTO_RN',
}

ALL_MODES = (RM.RNE, RM.RNA, RM.RTP, RM.RTN, RM.RTZ, RM.RAZ, RM.RTO, RM.RTE)

_SORTED = sorted(ADMITTED, key=lambda p: (p[0].name, p[1].name))
_IDS = [f'{a.name}_over_{b.name}' for a, b in _SORTED]


def _fmt(ctx: fp.Context) -> AbstractFormat:
    fmt = ctx.format()
    assert isinstance(fmt, AbstractableFormat)
    return AbstractFormat.from_format(fmt)


def _fp32() -> AbstractFormat:
    return _fmt(fp.FP32)


def _wide() -> AbstractFormat:
    """An intermediate wide enough that no premise fails on width."""
    return _fmt(fp.FP64)


class TestAdmitted:
    @pytest.mark.parametrize('rm1,rm2', _SORTED, ids=_IDS)
    def test_holds_over_a_wide_intermediate(self, rm1, rm2):
        pair = (rm1, rm2)
        assert double_round_ok(_fp32(), rm1, _wide(), rm2), ADMITTED[pair]

    def test_nothing_else_is_admitted(self):
        """The admitted set exactly: 9 of the 64 mode pairs."""
        got = {
            (a, b) for a, b in itertools.product(ALL_MODES, ALL_MODES)
            if double_round_ok(_fp32(), a, _wide(), b)
        }
        assert got == set(ADMITTED)


class TestRefused:
    def test_rne_over_rne(self):
        """The one that matters most: every `fp.FP*` context is RNE, so this is
        the pairing a hand-written program falls into, and Table 2's last row
        says it is unsound however wide the intermediate."""
        assert not double_round_ok(_fp32(), RM.RNE, _wide(), RM.RNE)
        assert not double_round_ok(_fp32(), RM.RNE, _fmt(fp.FP64), RM.RNE)

    @pytest.mark.parametrize('rm1', [RM.RNE, RM.RNA, RM.RTZ, RM.RAZ, RM.RTO])
    def test_rte_as_an_intermediate(self, rm1):
        """`RTE` is in FPy's `RoundingMode` but in none of the theorems."""
        assert not double_round_ok(_fp32(), rm1, _wide(), RM.RTE)

    @pytest.mark.parametrize('rm1', [RM.RTP, RM.RTN])
    def test_a_directed_nearest_mode_over_rto(self, rm1):
        """RTP and RTN are proved only against *themselves* -- there is no
        `rndRTO_RTP`, so this declines rather than being inferred from the fact
        that they reduce to RAZ/RTZ by sign."""
        assert not double_round_ok(_fp32(), rm1, _wide(), RM.RTO)

    def test_an_intermediate_narrower_than_the_target(self):
        assert not double_round_ok(_fmt(fp.FP64), RM.RTZ, _fmt(fp.FP32), RM.RTZ)


class TestPremiseDetails:
    """The premises that are easy to transcribe wrongly."""

    def test_the_same_mode_rules_need_no_bound_bump(self):
        """`rndRTZ_RTZ` and `rndRTO_RTO` take plain containment, exactly as
        RAZ-RAZ does.  An intermediate equal to the target is enough."""
        f = _fp32()
        for rm in (RM.RTZ, RM.RAZ, RM.RTP, RM.RTN):
            assert double_round_ok(f, rm, f, rm), rm
        # RTO-RTO too, given its `p2 >= 2` side condition
        assert double_round_ok(f, RM.RTO, f, RM.RTO)

    def test_rto_over_rto_needs_two_bits(self):
        """The one side condition the proofs state rather than derive."""
        one_bit = AbstractFormat(1, -149, _fp32().pos_bound)
        assert one_bit.prec == 1
        assert not double_round_ok(one_bit, RM.RTO, one_bit, RM.RTO)
        assert double_round_ok(one_bit, RM.RTO, one_bit.with_prec_offset(1), RM.RTO)

    def test_the_two_rto_premises_differ_in_which_grid(self):
        """RTO-to-directed takes `next` in the target's grid; RTO-to-nearest in
        the once-extended grid.  So an intermediate sized for the first can be
        too narrow for the second."""
        f = _fp32()
        just_enough = f.next_bound().with_prec_offset(1).with_exp_offset(-1)
        assert double_round_ok(f, RM.RTZ, just_enough, RM.RTO)
        assert not double_round_ok(f, RM.RNE, just_enough, RM.RTO)

    def test_a_special_the_intermediate_lacks_refuses(self):
        """The theorems are about finite values; FPy tracks specials, and one
        the target has but the intermediate does not would be lost on the way
        through.  Containment enforces this itself -- `_is_contained_in` makes
        `specials_contained_in` its first condition -- so this pins the
        behaviour rather than a guard of its own."""
        f = _fp32()
        assert f.has_neg_zero
        no_neg_zero = AbstractFormat(
            f.prec + 8, f.exp - 8, f.pos_bound, neg_bound=f.neg_bound,
            has_pos_inf=True, has_neg_inf=True, has_nan=True, has_neg_zero=False,
        )
        assert not double_round_ok(f, RM.RTZ, no_neg_zero, RM.RTZ)


class TestDeriveIntermediate:
    @pytest.mark.parametrize('rm1', [RM.RNE, RM.RNA, RM.RTZ, RM.RAZ, RM.RTO])
    def test_what_it_derives_is_accepted(self, rm1):
        """The property that ties the two halves together."""
        target = fp.FP32.with_params(rm=rm1)
        via = derive_intermediate(target)
        assert via.rounding_mode() is RM.RTO
        assert double_round_ok(_fmt(target), rm1, _fmt(via), RM.RTO)

    def test_it_is_unbounded_and_strictly_wider(self):
        """Unboundedness is what makes the composition agree at the ends of the
        range: the intermediate cannot overflow or underflow, so the only
        rounding that can is the target's.  `contained_in` alone would not say
        this -- it is reflexive, so the target itself would pass."""
        via = derive_intermediate(fp.FP32)
        f1, f2 = _fmt(fp.FP32), _fmt(via)
        assert f1.contained_in(f2)
        assert f2.bound == math.inf and f2.exp == -math.inf
        assert f2.prec > f1.prec

    @pytest.mark.parametrize('name', ['SINT8', 'UINT8', 'MX_INT8', 'FP8P7', 'BF16'])
    def test_every_format_family(self, name):
        """Each derives an intermediate its own premise accepts."""
        target = getattr(fp, name)
        via = derive_intermediate(target)
        assert double_round_ok(
            _fmt(target), target.rounding_mode(), _fmt(via), RM.RTO,
        )

    def test_a_fixed_point_target(self):
        """The premises are containment checks on `A`, indifferent to which
        family the format comes from, so a fixed-point target derives a
        fixed-point intermediate."""
        via = derive_intermediate(fp.INTEGER)
        assert double_round_ok(
            _fmt(fp.INTEGER), fp.INTEGER.rounding_mode(), _fmt(via), RM.RTO,
        )

    def test_a_context_that_does_not_round(self):
        with pytest.raises(ValueError, match='does not round'):
            derive_intermediate(fp.REAL)

    @pytest.mark.parametrize('rm1', [RM.RTP, RM.RTN])
    def test_a_mode_with_no_rto_rule(self, rm1):
        with pytest.raises(ValueError, match='no double-rounding rule'):
            derive_intermediate(fp.FP32.with_params(rm=rm1))
