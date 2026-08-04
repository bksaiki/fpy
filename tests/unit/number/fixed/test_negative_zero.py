"""
Testing that two's complement fixed-point formats have no negative zero.

Two's complement has a single encoding of zero, so `-0` is not representable
under a `FixedContext`, and `FixedContext.round()` must never produce one.
Sign-magnitude (`SMFixedContext`) does have a distinct `-0`, so it must be
unaffected.
"""

import fpy2 as fp
import pytest

from hypothesis import given, strategies as st

from ...generators import real_floats

_ZERO = fp.Float(s=False, exp=0, c=0)
_NEG_ZERO = fp.Float(s=True, exp=0, c=0)

_ROUNDING_MODES = [
    fp.RoundingMode.RNE,
    fp.RoundingMode.RNA,
    fp.RoundingMode.RTP,
    fp.RoundingMode.RTN,
    fp.RoundingMode.RTZ,
    fp.RoundingMode.RAZ,
    fp.RoundingMode.RTO,
    fp.RoundingMode.RTE,
]

_INT_CONTEXTS = [fp.SINT8, fp.SINT16, fp.SINT32, fp.UINT8, fp.UINT16, fp.UINT32]


class TestNegativeZeroNotRepresentable():

    @pytest.mark.parametrize('ctx', _INT_CONTEXTS)
    def test_negative_zero_unrepresentable(self, ctx: fp.FixedContext):
        assert not ctx.format().representable_in(_NEG_ZERO)
        assert not ctx.representable_under(_NEG_ZERO)

    @pytest.mark.parametrize('ctx', _INT_CONTEXTS)
    def test_positive_zero_representable(self, ctx: fp.FixedContext):
        assert ctx.format().representable_in(_ZERO)
        assert ctx.representable_under(_ZERO)

    @given(st.booleans(), st.integers(min_value=1, max_value=8),
           st.integers(min_value=-4, max_value=4), st.integers(min_value=-8, max_value=8))
    def test_any_negative_zero_encoding(self, signed: bool, nbits: int, scale: int, exp: int):
        # a negative zero at any exponent, in any two's complement format
        ctx = fp.FixedContext(signed, scale, nbits + 1)
        assert not ctx.format().representable_in(fp.Float(s=True, exp=exp, c=0))
        assert not ctx.format().representable_in(fp.RealFloat(s=True, exp=exp, c=0))
        assert ctx.format().representable_in(fp.Float(s=False, exp=exp, c=0))

    @pytest.mark.parametrize('ctx', _INT_CONTEXTS)
    def test_negative_zero_rejected_by_encode(self, ctx: fp.FixedContext):
        # `encode` would otherwise silently drop the sign, since `-0` and `+0`
        # share the encoding `0`
        assert ctx.encode(_ZERO) == 0
        with pytest.raises(ValueError):
            ctx.encode(_NEG_ZERO)


class TestSignMagnitudeKeepsNegativeZero():
    """Sign-magnitude has a distinct `-0` encoding, so it must be unaffected."""

    def test_negative_zero_representable(self):
        ctx = fp.SMFixedContext(0, 8)
        assert ctx.format().representable_in(_NEG_ZERO)
        assert ctx.representable_under(_NEG_ZERO)

    def test_negative_zero_roundtrips(self):
        ctx = fp.SMFixedContext(0, 8)
        encoded = ctx.encode(_NEG_ZERO)
        assert encoded != ctx.encode(_ZERO)
        assert ctx.decode(encoded).s
        assert ctx.decode(encoded).is_zero()


class TestRoundNeverProducesNegativeZero():

    @pytest.mark.parametrize('rm', _ROUNDING_MODES)
    @pytest.mark.parametrize('v', [-0.0, -1e-9, -0.25, -0.5, -0.75, -0.999])
    def test_round_small_negatives(self, rm: fp.RoundingMode, v: float):
        # every one of these rounds to zero in a scale-0 format
        ctx = fp.FixedContext(True, 0, 8, rm)
        r = ctx.round(v)
        assert not (r.is_zero() and r.s), f'round({v}) under {rm} gave -0'
        assert ctx.representable_under(r)

    @pytest.mark.parametrize('rm', _ROUNDING_MODES)
    @given(real_floats(prec_max=8, exp_min=-8, exp_max=8))
    def test_round_is_always_representable(self, rm: fp.RoundingMode, x: fp.RealFloat):
        # the core invariant: `round` lands on a representable value
        ctx = fp.FixedContext(True, 0, 8, rm)
        r = ctx.round(x)
        assert ctx.representable_under(r)
        assert not (r.is_zero() and r.s)

    @pytest.mark.parametrize('rm', _ROUNDING_MODES)
    @given(real_floats(prec_max=8, exp_min=-8, exp_max=8))
    def test_rounded_values_encode(self, rm: fp.RoundingMode, x: fp.RealFloat):
        # a rounded value must always be encodable, which is what a stricter
        # `representable_in` would otherwise break
        ctx = fp.FixedContext(True, 0, 8, rm)
        r = ctx.round(x)
        assert ctx.decode(ctx.encode(r)) == r

    @pytest.mark.parametrize('rm', _ROUNDING_MODES)
    def test_round_preserves_negative_magnitudes(self, rm: fp.RoundingMode):
        # normalizing `-0` must not disturb genuinely negative results;
        # these are all exactly representable, so rounding is the identity
        ctx = fp.FixedContext(True, 0, 8, rm)
        for v in (-1.0, -2.0, -64.0, -127.0, -128.0):
            r = ctx.round(v)
            assert r.s, f'round({v}) under {rm} lost its sign'
            assert r == v, f'round({v}) under {rm} gave {float(r)}'

    @pytest.mark.parametrize('rm', _ROUNDING_MODES)
    def test_round_of_negative_half(self, rm: fp.RoundingMode):
        # -0.5 lands on 0, -1, or (for RTO) -1 depending on the mode, but
        # never on -0
        ctx = fp.FixedContext(True, 0, 8, rm)
        r = ctx.round(-0.5)
        assert r == 0 or r == -1, f'round(-0.5) under {rm} gave {float(r)}'
        assert not (r.is_zero() and r.s)

    def test_round_preserves_flags(self):
        # the fixup must not drop status flags from the rounded result
        ctx = fp.FixedContext(True, 0, 8, fp.RoundingMode.RTZ)
        r = ctx.round(-0.5)
        assert r.is_zero() and not r.s
        assert r.inexact

    @pytest.mark.parametrize('rm', _ROUNDING_MODES)
    def test_round_at_never_produces_negative_zero(self, rm: fp.RoundingMode):
        # `round_at` shares the same path as `round`
        ctx = fp.FixedContext(True, 0, 8, rm)
        for v in (-0.5, -0.25, -1e-9):
            r = ctx.round_at(v, 0)
            assert not (r.is_zero() and r.s), f'round_at({v}, 0) under {rm} gave -0'
            assert ctx.representable_under(r)
