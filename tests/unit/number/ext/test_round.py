import unittest

import fpy2 as fp


class TestRoundSaturate(unittest.TestCase):
    """Test saturation behavior."""

    def test_saturate(self):
        ctx = fp.IEEEContext(5, 8, overflow=fp.OV.SATURATE)
        maxval = ctx.maxval()
        eps = fp.Float(c=1, exp=maxval.exp)

        self.assertEqual(ctx.round(maxval), maxval)
        self.assertEqual(ctx.round(fp.add(maxval, eps)), maxval)
