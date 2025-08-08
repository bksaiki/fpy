import fpy2 as fp
import unittest

from hypothesis import given, strategies as st

from ...generators import real_floats, floats, rounding_modes

class RoundTestCase(unittest.TestCase):
    """Testing rounding behavior of `ExpContext`."""

    @given(real_floats(prec=8, exp_min=-256, exp_max=256), st.integers(1, 8), st.integers(-8, 8), rounding_modes())
    def test_round_real(self, x: fp.RealFloat, n: int, eoffset: int, rm: fp.RoundingMode):
        ctx = fp.ExpContext(n, eoffset, rm=rm)
        rounded = ctx.round(x)
        self.assertIsInstance(rounded, fp.Float, f'x={x}, rounded={rounded}')
        self.assertTrue(ctx.representable_under(fp.Float(x=rounded, ctx=None)), f'x={x}, rounded={rounded}')
        if x.is_zero():
            self.assertTrue(rounded.isnan, f'x={x}, rounded={rounded}')
        elif x.s:
            self.assertTrue(rounded.isnan, f'x={x}, rounded={rounded}')

    @given(floats(prec=8, exp_min=-256, exp_max=256), st.integers(1, 8), st.integers(-8, 8), rounding_modes())
    def test_round_float(self, x: fp.Float, n: int, eoffset: int, rm: fp.RoundingMode):
        ctx = fp.ExpContext(n, eoffset, rm=rm)
        rounded = ctx.round(x)
        self.assertIsInstance(rounded, fp.Float, f'x={x}, rounded={rounded}')
        self.assertTrue(ctx.representable_under(rounded), f'x={x}, rounded={rounded}')
        if x.is_zero():
            self.assertTrue(rounded.isnan, f'x={x}, rounded={rounded}')
        elif x.s:
            self.assertTrue(rounded.isnan, f'x={x}, rounded={rounded}')
        elif x.isinf:
            self.assertTrue(rounded.isnan, f'x={x}, rounded={rounded}')
