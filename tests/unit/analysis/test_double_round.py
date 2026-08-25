"""
Figure 8 of *When Double Rounding is Correct*, as a predicate.

The oracle is the Lean development, `Mpfx/DoubleRounding.lean
<https://github.com/bksaiki/mpfx-lean/blob/main/Mpfx/DoubleRounding.lean>`_.
Each admitted pairing names the theorem it comes from; a pairing not listed has
no theorem and must be refused.
"""

import itertools
import math
from fractions import Fraction

import pytest

import fpy2 as fp
from fpy2.analysis.format_infer import (
    AbstractableFormat,
    AbstractFormat,
    DoubleRoundOp,
    derive_intermediate,
    double_round_ok,
    double_round_op_ok,
)
from fpy2.number import RealFloat, RoundingMode as RM

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

    def test_a_stochastic_target(self):
        """No composition reproduces a rounding that is not a function of its
        input, so there is no intermediate to hand back."""
        with pytest.raises(ValueError, match='stochastically'):
            derive_intermediate(fp.FP32.with_params(num_randbits=2))

    def test_a_context_that_does_not_round(self):
        with pytest.raises(ValueError, match='does not round'):
            derive_intermediate(fp.REAL)

    @pytest.mark.parametrize('rm1', [RM.RTP, RM.RTN])
    def test_a_mode_with_no_rto_rule(self, rm1):
        with pytest.raises(ValueError, match='no double-rounding rule'):
            derive_intermediate(fp.FP32.with_params(rm=rm1))


class TestAgainstArithmetic:
    """The predicate checked against actual rounding, not against the
    derivation that shares its helpers.

    Every other numeric test here routes through `derive_intermediate`, so an
    error the predicate and the derivation share would be self-consistent and
    invisible.  This rounds twice versus once over a grid fine enough to
    straddle every midpoint of the target, and requires that an accepted pair
    really does agree.
    """

    FINALS = (RM.RNE, RM.RNA, RM.RTZ, RM.RAZ, RM.RTO, RM.RTP, RM.RTN)

    @staticmethod
    def _values(p: int) -> list[Fraction]:
        s, out = p + 3, []
        for binade in (0, 1, 4):
            lo, hi = 2 ** s, 2 ** (s + 1)
            for m in range(lo, hi, max(1, (hi - lo) // 48)):
                v = Fraction(m, 2 ** s) * Fraction(2) ** binade
                out += [v, -v]
        return out

    @staticmethod
    def _agrees(target, via, xs) -> bool:
        return all(str(target.round(x)) == str(target.round(via.round(x)))
                   for x in xs)

    @pytest.mark.parametrize('p1', [3, 4])
    def test_an_accepted_pair_agrees_with_rounding_once(self, p1):
        for rm1 in self.FINALS:
            target = fp.MPFloatContext(p1, rm1)
            f1 = _fmt(target)
            xs = self._values(p1)
            for dp in (0, 1, 2, 3):
                for rm2 in (rm1, RM.RTO):
                    via = fp.MPFloatContext(p1 + dp, rm2)
                    if double_round_ok(f1, rm1, _fmt(via), rm2):
                        assert self._agrees(target, via, xs), (
                            f'p1={p1} {rm1.name} over p2={p1 + dp} {rm2.name}'
                        )

    def test_the_sweep_can_detect_a_bad_pair(self):
        """Otherwise the test above would pass on a predicate that accepts
        nothing -- or on one that accepts everything, if the sweep were too
        coarse to notice."""
        target = fp.MPFloatContext(3, RM.RNE)
        too_narrow = fp.MPFloatContext(3, RM.RTO)
        assert not double_round_ok(_fmt(target), RM.RNE, _fmt(too_narrow), RM.RTO)
        assert not self._agrees(target, too_narrow, self._values(3))

def _flx(p: int) -> AbstractFormat:
    """A precision-only format: no minimum quantum, no bound (Flocq `FLX`)."""
    return _fmt(fp.MPFloatContext(p, RM.RNE))


class TestOperationRules:
    """Roux 2014's operation-specific rules, `Mpfx/DoubleRounding{Add,Div,Sqrt}
    .lean`.  They hold only for the results of one operation, but admit nearest
    over nearest -- which Figure 8 never does."""

    NEED = {DoubleRoundOp.ADD: lambda p: 2 * p + 1,
            DoubleRoundOp.DIV: lambda p: 2 * p,
            DoubleRoundOp.SQRT: lambda p: 2 * p + 2}

    @pytest.mark.parametrize('op', list(DoubleRoundOp), ids=lambda o: o.value)
    @pytest.mark.parametrize('p1', [2, 3, 8, 24])
    @pytest.mark.parametrize('flt', [False, True], ids=['flx', 'flt'])
    def test_the_precision_bound_is_exact(self, op, p1, flt):
        """Sound at the stated width, refused one digit below -- these bounds are
        tight, and `div`'s counterexample is Roux's Remark 30.

        Both families: each rule is proved twice, and only the FLX statements
        drop the exponent conditions, so a bound transcribed wrongly in one
        branch is invisible from the other.  The FLT intermediate here carries an
        exponent margin deep enough for every rule, leaving precision the only
        thing in play.
        """
        need = self.NEED[op](p1)

        def f2(p: int) -> AbstractFormat:
            return AbstractFormat(p, -4 * p1 - 2, float('inf')) if flt else _flx(p)

        f1 = AbstractFormat(p1, 0, float('inf')) if flt else _flx(p1)
        assert double_round_op_ok(op, f1, RM.RNE, f2(need), RM.RNE)
        assert not double_round_op_ok(op, f1, RM.RNE, f2(need - 1), RM.RNE)

    @pytest.mark.parametrize('op', list(DoubleRoundOp), ids=lambda o: o.value)
    @pytest.mark.parametrize('rm1,rm2', [
        (RM.RTZ, RM.RNE), (RM.RNE, RM.RTZ), (RM.RTO, RM.RNE), (RM.RNE, RM.RTO),
        (RM.RTP, RM.RTP), (RM.RTE, RM.RNE),
    ])
    def test_only_nearest_over_nearest(self, op, rm1, rm2):
        """Proved for `.nearest` on both sides.  Not conservatism: `add` over a
        far wider intermediate still disagrees with a directed target."""
        assert not double_round_op_ok(op, _flx(3), rm1, _flx(64), rm2)

    @pytest.mark.parametrize('op', list(DoubleRoundOp), ids=lambda o: o.value)
    @pytest.mark.parametrize('rm1', [RM.RNE, RM.RNA])
    @pytest.mark.parametrize('rm2', [RM.RNE, RM.RNA])
    def test_the_tie_breaks_are_independent(self, op, rm1, rm2):
        assert double_round_op_ok(op, _flx(4), rm1, _flx(64), rm2)

    @pytest.mark.parametrize('op', list(DoubleRoundOp), ids=lambda o: o.value)
    def test_the_degenerate_format(self, op):
        """`IsUndefined`: one digit and no minimum quantum leaves nearest-even
        with no answer, so the theorems exclude it.  Nearest-away is fine."""
        assert not double_round_op_ok(op, _flx(1), RM.RNE, _flx(64), RM.RNE)
        assert double_round_op_ok(op, _flx(1), RM.RNA, _flx(64), RM.RNE)

    @staticmethod
    def _at_exp(e: int, prec: int = 64, **specials) -> AbstractFormat:
        """A wide FLT intermediate whose least quantum is `e`.

        It carries every special by default: the rules require the target's to
        survive, so a bare format would fail that check rather than the one
        under test.
        """
        flags = {'has_pos_inf': True, 'has_neg_inf': True, 'has_nan': True,
                 'has_neg_zero': True}
        return AbstractFormat(prec, e, float('inf'), **{**flags, **specials})

    def test_the_addition_exponent_condition(self):
        """`exp2 <= exp1`, stated over `WithBot`, so it spans both families."""
        f1 = _fmt(fp.FP32)
        assert isinstance(f1.exp, int)
        assert double_round_op_ok(DoubleRoundOp.ADD, f1, RM.RNE, _fmt(fp.FP64), RM.RNE)
        assert double_round_op_ok(
            DoubleRoundOp.ADD, f1, RM.RNE, self._at_exp(f1.exp), RM.RNE,
        )
        assert not double_round_op_ok(
            DoubleRoundOp.ADD, f1, RM.RNE, self._at_exp(f1.exp + 1), RM.RNE,
        )

    def test_the_division_underflow_margin(self):
        """A quotient is generally irrational, so `div` needs room *below* the
        target's least quantum: `exp2 <= exp1 - p1 - 2`."""
        f1 = _fmt(fp.FP32)
        p1, e1 = f1.prec, f1.exp
        assert isinstance(p1, int) and isinstance(e1, int)
        for off, ok in [(0, True), (1, False)]:
            assert double_round_op_ok(
                DoubleRoundOp.DIV, f1, RM.RNE, self._at_exp(e1 - p1 - 2 + off), RM.RNE,
            ) is ok

    def test_the_sqrt_underflow_disjunction(self):
        """`sqrt`'s Table II bound is a *disjunction*: either room below the
        target's quantum, or twice as much exponent range.  Either alone
        suffices, and only failing both refuses."""
        f1 = _fmt(fp.FP32)
        p1, e1 = f1.prec, f1.exp
        assert isinstance(p1, int) and isinstance(e1, int)
        first, second = e1 - p1 - 2, (e1 - 4 * p1 - 2) // 2
        assert first < second      # the second disjunct is the weaker one
        for e2, ok in [(first, True), (second, True), (second + 1, False)]:
            assert double_round_op_ok(
                DoubleRoundOp.SQRT, f1, RM.RNE, self._at_exp(e2), RM.RNE,
            ) is ok, e2

    @pytest.mark.parametrize('op,spans', [(DoubleRoundOp.ADD, True),
                                          (DoubleRoundOp.DIV, False),
                                          (DoubleRoundOp.SQRT, False)])
    def test_a_mixed_exponent_family(self, op, spans):
        """`div` and `sqrt` are proved for FLX and FLT separately, with no mixed
        statement, so an unbounded exponent on one side only is refused rather
        than assumed sound."""
        assert double_round_op_ok(op, _flx(3), RM.RNE, _flx(64), RM.RNE)
        assert double_round_op_ok(op, _fmt(fp.FP32), RM.RNE, _fmt(fp.FP64), RM.RNE)
        assert double_round_op_ok(op, _fmt(fp.FP32), RM.RNE, _flx(64), RM.RNE) is spans

    @pytest.mark.parametrize('op', list(DoubleRoundOp), ids=lambda o: o.value)
    @pytest.mark.parametrize('missing', ['has_nan', 'has_pos_inf', 'has_neg_zero'])
    def test_a_special_the_intermediate_lacks(self, op, missing):
        """The premises are about finite values, but a program carries specials
        through the split: one the target has and the intermediate has not comes
        back changed, or raises.  Figure 8 gets this from containment; these
        rules need it stated, since they deliberately do not ask for
        containment."""
        f1 = _fmt(fp.FP32)
        assert isinstance(f1.exp, int)
        deep = f1.exp - 200
        assert double_round_op_ok(op, f1, RM.RNE, self._at_exp(deep), RM.RNE)
        assert not double_round_op_ok(
            op, f1, RM.RNE, self._at_exp(deep, **{missing: False}), RM.RNE,
        )

    def test_a_fixed_point_format(self):
        """The theorems take a finite precision, which a fixed-point format has
        not, so its rounding is Figure 8's business alone."""
        for op in DoubleRoundOp:
            assert not double_round_op_ok(
                op, _fmt(fp.INTEGER), RM.RNE, _flx(64), RM.RNE,
            )


class TestDeriveForOperation:
    """`derive_intermediate(target, op)`: the width that operation's own rule
    asks for, as a context ready to install."""

    @pytest.mark.parametrize('op', list(DoubleRoundOp), ids=lambda o: o.value)
    @pytest.mark.parametrize('name', ['FP32', 'FP16', 'BF16', 'FP8P4', 'FP8P7'])
    def test_what_it_derives_is_accepted(self, op, name):
        """The property tying the two halves together, for every float family."""
        target = getattr(fp, name)
        via = derive_intermediate(target, op)
        assert double_round_op_ok(
            op, _fmt(target), target.rounding_mode(),
            _fmt(via), via.rounding_mode(),
        )

    @pytest.mark.parametrize('op', list(DoubleRoundOp), ids=lambda o: o.value)
    def test_it_rounds_to_nearest_like_the_target(self, op):
        """Not round-to-odd: these rules are nearest-only, and the tie-breaks are
        independent, so the target's own mode serves."""
        for rm in (RM.RNE, RM.RNA):
            target = fp.FP32.with_params(rm=rm)
            assert derive_intermediate(target, op).rounding_mode() is rm

    @pytest.mark.parametrize('op', list(DoubleRoundOp), ids=lambda o: o.value)
    def test_it_is_narrower_than_the_round_to_odd_one_is_wide(self, op):
        """The point of these rules: a *hardware* format can serve.  FP64 meets
        every one of them for an FP32 target."""
        assert double_round_op_ok(
            op, _fmt(fp.FP32), RM.RNE, _fmt(fp.FP64), RM.RNE,
        )
        assert _fmt(derive_intermediate(fp.FP32, op)).prec <= 50

    @pytest.mark.parametrize('op', list(DoubleRoundOp), ids=lambda o: o.value)
    def test_an_flx_target_takes_an_flx_intermediate(self, op):
        """`div` and `sqrt` are proved per family, so the derivation must not
        hand back a minimum quantum the target has not got."""
        target = fp.MPFloatContext(11, RM.RNE)
        via = derive_intermediate(target, op)
        assert _fmt(via).exp == -math.inf
        assert double_round_op_ok(op, _fmt(target), RM.RNE, _fmt(via), RM.RNE)

    @pytest.mark.parametrize('op', list(DoubleRoundOp), ids=lambda o: o.value)
    def test_a_directed_target(self, op):
        with pytest.raises(ValueError, match='round-to-nearest'):
            derive_intermediate(fp.FP32.with_params(rm=RM.RTZ), op)

    @pytest.mark.parametrize('op', list(DoubleRoundOp), ids=lambda o: o.value)
    def test_a_fixed_point_target(self, op):
        """No finite precision, so no rule -- unlike the round-to-odd
        derivation, which handles fixed point.  A nearest fixed-point target, so
        that precision is what refuses it rather than the mode."""
        target = fp.MPFixedContext(-1, RM.RNE)
        with pytest.raises(ValueError, match='finite precision'):
            derive_intermediate(target, op)
        assert derive_intermediate(target) is not None

    def test_a_sqrt_target_representing_nothing_below_one(self):
        """`rndSqrt_FLT` states `emin1 <= 0`; a format whose least quantum sits
        above one does not satisfy it."""
        target = fp.MPSFloatContext(8, 12, RM.RNE)
        assert _fmt(target).exp > 0
        with pytest.raises(ValueError, match='exp1 <= 0'):
            derive_intermediate(target, DoubleRoundOp.SQRT)
        assert derive_intermediate(target, DoubleRoundOp.ADD) is not None

    def test_the_default_is_still_round_to_odd(self):
        assert derive_intermediate(fp.FP32).rounding_mode() is RM.RTO


class TestOperationRulesAgainstArithmetic:
    """The predicate against actual rounding, per operation.

    :class:`TestOperationRules` checks the transcribed inequalities; this rounds
    twice versus once over every pair of operands *in the target format*, which
    is the premise the rules are stated under.
    """

    OPS = {DoubleRoundOp.ADD: (fp.add, 2), DoubleRoundOp.DIV: (fp.div, 2),
           DoubleRoundOp.SQRT: (fp.sqrt, 1)}

    @staticmethod
    def _values(p: int) -> list[fp.Float]:
        out = [fp.Float(x=RealFloat())]
        for m in range(2 ** (p - 1), 2 ** p):
            for e in range(-3, 2):
                out.append(fp.Float(x=RealFloat(m=m, exp=e)))
                out.append(fp.Float(x=RealFloat(s=True, c=m, exp=e)))
        return out

    @classmethod
    def _mismatches(cls, op, target, via) -> int:
        fn, arity = cls.OPS[op]
        xs = cls._values(target.pmax)
        pairs = ((x,) for x in xs) if arity == 1 else itertools.product(xs, xs)
        bad = 0
        for args in pairs:
            if arity == 1 and args[0].s:
                continue
            if op is DoubleRoundOp.DIV and args[1].is_zero():
                continue
            if str(fn(*args, target)) != str(target.round(fn(*args, via))):
                bad += 1
        return bad

    @pytest.mark.parametrize('op', list(DoubleRoundOp), ids=lambda o: o.value)
    @pytest.mark.parametrize('p1', [3, 4])
    def test_an_accepted_pair_agrees_with_rounding_once(self, op, p1):
        target = fp.MPFloatContext(p1, RM.RNE)
        f1 = _fmt(target)
        for dp in range(0, 2 * p1 + 4):
            via = fp.MPFloatContext(p1 + dp, RM.RNE)
            if double_round_op_ok(op, f1, RM.RNE, _fmt(via), RM.RNE):
                assert self._mismatches(op, target, via) == 0, \
                    f'{op.value}: p1={p1} over p2={p1 + dp}'

    @pytest.mark.parametrize('op', list(DoubleRoundOp), ids=lambda o: o.value)
    def test_the_sweep_can_detect_a_bad_pair(self, op):
        """One digit under the stated bound really does break, so the test above
        is not passing on a predicate that accepts nothing."""
        p1 = 3
        need = TestOperationRules.NEED[op](p1)
        target = fp.MPFloatContext(p1, RM.RNE)
        assert not double_round_op_ok(
            op, _fmt(target), RM.RNE, _fmt(fp.MPFloatContext(need - 1, RM.RNE)), RM.RNE,
        )
        assert self._mismatches(
            op, target, fp.MPFloatContext(need - 1, RM.RNE),
        ) > 0

