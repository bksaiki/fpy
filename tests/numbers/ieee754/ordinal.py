import unittest
import random

from fpy2.number import Float, IEEEContext, RM

class ToOrdinalTestCase(unittest.TestCase):
    """Testing `IEEEContext.to_ordinal()`"""

    def test_native(self, num_encodings: int = 10_000):
        # rounding context for native Python floats
        fp64 = IEEEContext(11, 64, RM.RNE)
        # sample 10_000 random floating-point values
        random.seed(1)
        xs: list[Float] = []
        for _ in range(num_encodings):
            s = True if random.randint(0, 1) == 0 else False
            exp = random.randint(fp64.expmin, fp64.expmax)
            c = random.randint(0, (1 << fp64.pmax) - 1)
            x = Float(s, exp, c, ctx=fp64)
            assert fp64.is_representable(x)
            xs.append(x)
        # run encoding
        for x in xs:
            i = fp64.to_ordinal(x)
            self.assertIsInstance(i, int, f'x={x}, i={i}')
            self.assertGreaterEqual(i, -(1 << fp64.nbits - 1), f'x={x}, i={i}')
            self.assertLess(i, 1 << fp64.nbits - 1, f'x={x}, i={i}')

    def test_small(self, es_max: int = 6, nbits_max: int = 8):
        # iterate over possible contexts
        for es in range(2, es_max+1):
            for nbits in range(es + 2, nbits_max+1):
                ctx = IEEEContext(es, nbits, RM.RNE)
                # for ctx, encode all possible values
                for s in (True, False):
                    for exp in range(ctx.expmin, ctx.expmax):
                        for c in range(0, 1 << ctx.pmax - 1):
                            x = Float(s, exp, c, ctx=ctx)
                            assert ctx.is_representable(x)

                            i = ctx.to_ordinal(x)
                            self.assertIsInstance(i, int, f'x={x}, i={i}')
                            self.assertGreaterEqual(i, -(1 << ctx.nbits - 1), f'x={x}, i={i}')
                            self.assertLess(i, 1 << ctx.nbits - 1, f'x={x}, i={i}')

class OrdinalRoundTripTestCase(unittest.TestCase):
    """Ensure `IEEEContext.to_ordinal()` and `IEEEContext.from_ordinal()` roundtrips."""
    
    def test_native(self, num_encodings: int = 10_000):
        # rounding context for native Python floats
        fp64 = IEEEContext(11, 64, RM.RNE)
        # sample 10_000 random encodings
        random.seed(1)
        xs: list[Float] = []
        for _ in range(num_encodings):
            s = True if random.randint(0, 1) == 0 else False
            exp = random.randint(fp64.expmin, fp64.expmax)
            c = random.randint(0, (1 << fp64.pmax) - 1)
            x = Float(s, exp, c, ctx=fp64)
            assert fp64.is_representable(x)
            xs.append(x)
        # run encoding
        for x in xs:
            i = fp64.to_ordinal(x)
            y = fp64.from_ordinal(i)
            self.assertIsInstance(y, Float, f'i={i}, x={x}')
            self.assertEqual(x, y, f'x={x}, y={y}')
