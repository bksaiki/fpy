import unittest

from fpy2 import (
    ExtFloatContext,
    RealFloat, Float,
    S1E5M2, S1E4M3,
    MX_E5M2, MX_E4M3, MX_E3M2, MX_E2M3, MX_E2M1,
    FP8P1, FP8P2, FP8P3, FP8P4, FP8P5, FP8P6, FP8P7
)


_common: list[ExtFloatContext] = [
    S1E5M2, S1E4M3,
    MX_E5M2, MX_E4M3, MX_E3M2, MX_E2M3, MX_E2M1,
    FP8P1, FP8P2, FP8P3, FP8P4, FP8P5, FP8P6, FP8P7
]

class DecodeTestCase(unittest.TestCase):
    """Testing `ExtFloatContext.decode()`"""

    def test_common(self):
        # iterate over common contexts
        for ctx in _common:
            # for ctx, decode all possible encodings
            for i in range(1 << ctx.nbits):
                x = ctx.decode(i)
                self.assertIsInstance(x, Float, f'i={i}, x={x}')
                self.assertTrue(x.is_representable(), f'i={i}, x={x}')

class EncodeTestCase(unittest.TestCase):
    """Testing `ExtFloatContext.encode()`"""

    def test_common(self):
        # iterate over common contexts
        for ctx in _common:
            # for ctx, encode all possible values
            xs: list[Float] = []
            for s in (True, False):
                # TODO: fix?
                expmax = ctx.expmin if ctx.expmax < ctx.expmin else ctx.expmax
                for exp in range(ctx.expmin, expmax + 1):
                    for c in range(0, 1 << ctx.pmax - 1):
                        xr = RealFloat(s, exp, c)
                        if ctx.representable_under(xr):
                            xs.append(Float(x=xr, ctx=ctx))
            # run encoding
            for x in xs:
                i = ctx.encode(x)
                self.assertIsInstance(i, int, f'x={x}, i={i}')
                self.assertGreaterEqual(i, 0, f'x={x}, i={i}')
                self.assertLess(i, 1 << ctx.nbits, f'x={x}, i={i}')


class EncodeRoundTripTestCase(unittest.TestCase):
    """Testing `ExtFloatContext.encode()` and `ExtFloatContext.decode()`"""

    def test_common(self):
        # iterate over common contexts
        for ctx in _common:
            # for ctx, decode all possible encodings
            for i in range(1 << ctx.nbits):
                x = ctx.decode(i)
                j = ctx.encode(x)
                if x.isnan:
                    y = ctx.decode(j)
                    self.assertTrue(y.isnan, f'i={i}, j={j}, y={y}')
                else:
                    self.assertEqual(i, j, f'i={i}, j={j}')
