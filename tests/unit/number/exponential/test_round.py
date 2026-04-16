import fpy2 as fp

from hypothesis import given, strategies as st

from ...generators import real_floats, floats, rounding_modes

class RoundTestCase():
    """Testing rounding behavior of `ExpContext`."""

    @given(real_floats(prec_max=8, exp_min=-256, exp_max=256), st.integers(1, 8), st.integers(-8, 8), rounding_modes())
    def test_round_real(self, x: fp.RealFloat, n: int, eoffset: int, rm: fp.RoundingMode):
        ctx = fp.ExpContext(n, eoffset, rm=rm)
        rounded = ctx.round(x)
        assert isinstance(rounded, fp.Float), f'x={x}, rounded={rounded}'
        assert ctx.representable_under(fp.Float(x=rounded, ctx=None)), f'x={x}, rounded={rounded}'
        if x.is_zero():
            assert rounded.isnan, f'x={x}, rounded={rounded}'
        elif x.s:
            assert rounded.isnan, f'x={x}, rounded={rounded}'

    @given(floats(prec_max=8, exp_min=-256, exp_max=256), st.integers(1, 8), st.integers(-8, 8), rounding_modes())
    def test_round_float(self, x: fp.Float, n: int, eoffset: int, rm: fp.RoundingMode):
        ctx = fp.ExpContext(n, eoffset, rm=rm)
        rounded = ctx.round(x)
        assert isinstance(rounded, fp.Float), f'x={x}, rounded={rounded}'
        assert ctx.representable_under(rounded), f'x={x}, rounded={rounded}'
        if x.is_zero():
            assert rounded.isnan, f'x={x}, rounded={rounded}'
        elif x.s:
            assert rounded.isnan, f'x={x}, rounded={rounded}'
        elif x.isinf:
            assert rounded.isnan, f'x={x}, rounded={rounded}'
