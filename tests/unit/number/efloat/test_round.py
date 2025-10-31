import unittest

import fpy2 as fp
from hypothesis import given

from ...generators import floats

_common: list[fp.EFloatContext] = [
    fp.S1E5M2, fp.S1E4M3,
    fp.MX_E5M2, fp.MX_E4M3, fp.MX_E3M2, fp.MX_E2M3, fp.MX_E2M1,
    fp.FP8P1, fp.FP8P2, fp.FP8P3, fp.FP8P4, fp.FP8P5, fp.FP8P6, fp.FP8P7
]

class TestRound(unittest.TestCase):
    """Test rounding behavior."""

    @given(floats(prec_max=fp.FP16.pmax, exp_min=fp.FP16.expmin, exp_max=fp.FP16.expmax))
    def test_round(self, x: fp.Float):
        # iterate over common contexts
        for ctx in _common:
            y = ctx.round(x)
            y = fp.Float(x=y, ctx=None) # strip context for testing
            self.assertIsInstance(y, fp.Float, f'ctx={ctx}, x={x}, y={y}')
            self.assertTrue(ctx.representable_under(y), f'ctx={ctx}, x={x}, y={y}')

class TestRoundSaturate(unittest.TestCase):
    """Test saturation behavior."""

    def test_saturate(self):
        ctx = fp.IEEEContext(5, 8, overflow=fp.OV.SATURATE)
        maxval = ctx.maxval()
        eps = fp.Float(c=1, exp=maxval.exp)

        self.assertEqual(ctx.round(maxval), maxval)
        self.assertEqual(ctx.round(fp.add(maxval, eps)), maxval)
