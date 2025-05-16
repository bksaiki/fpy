import unittest

from fpy2 import (
    ExtContext,
    Float,
    S1E5M2, S1E4M3,
    MX_E5M2, MX_E4M3, MX_E3M2, MX_E2M3, MX_E2M1,
    FP8P1, FP8P2, FP8P3, FP8P4, FP8P5, FP8P6, FP8P7
)

_common: list[ExtContext] = [
    S1E5M2, S1E4M3,
    MX_E5M2, MX_E4M3, MX_E3M2, MX_E2M3, MX_E2M1,
    FP8P1, FP8P2, FP8P3, FP8P4, FP8P5, FP8P6, FP8P7
]

class ToOrdinalTestCase(unittest.TestCase):
    """Testing `IEEEContext.to_ordinal()`"""

    def test_common(self):
        # iterate over common contexts
        for ctx in _common:
            # for ctx, encode all possible values
            for s in (True, False):
                for exp in range(ctx.expmin, ctx.expmax + 1):
                    for c in range(0, 1 << ctx.pmax):
                        x = Float(s, exp, c, ctx=ctx)
                        if ctx.is_representable(x):
                            i = ctx.to_ordinal(x)
                            self.assertIsInstance(i, int, f'x={x}, i={i}')
                            self.assertGreaterEqual(i, -(1 << ctx.nbits - 1), f'x={x}, i={i}')
                            self.assertLess(i, 1 << ctx.nbits - 1, f'x={x}, i={i}')


class OrdinalRoundTripTestCase(unittest.TestCase):
    """Testing `ExtContext.to_ordinal()` and `ExtContext.from_ordinal()`"""

    def test_common(self):
        # iterate over common contexts
        for ctx in _common:
            # for ctx, encode all possible values
            for s in (True, False):
                for exp in range(ctx.expmin, ctx.expmax + 1):
                    for c in range(0, 1 << ctx.pmax):
                        x = Float(s, exp, c, ctx=ctx)
                        if ctx.is_representable(x):
                            # run ordinal conversion
                            i = ctx.to_ordinal(x)
                            y = ctx.from_ordinal(i)
                            self.assertIsInstance(y, Float, f'x={x}, i={i}, y={y}')
                            self.assertEqual(x, y, f'x={x}, i={i}, y={y}')
