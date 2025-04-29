import unittest
import random

from fpy2 import Float, IEEEContext, RM, FP64

class DecodeTestCase(unittest.TestCase):
    """Testing `IEEEContext.decode()`"""

    def test_native(self, num_encodings: int = 10_000):
        # sample 10_000 random encodings
        random.seed(1)
        encodings = [random.randint(0, (1 << FP64.nbits) - 1) for _ in range(num_encodings)]
        # run decode
        for i in encodings:
            x = FP64.decode(i)
            self.assertIsInstance(x, Float, f'i={i}, x={x}')

    def test_small(self, es_max: int = 6, nbits_max: int = 8):
        # iterate over possible contexts
        for es in range(2, es_max+1):
            for nbits in range(es + 2, nbits_max+1):
                ctx = IEEEContext(es, nbits, RM.RNE)
                # for ctx, decode all possible encodings
                for i in range(1 << ctx.nbits):
                    x = ctx.decode(i)
                    self.assertIsInstance(x, Float, f'i={i}, x={x}')

class EncodeTestCase(unittest.TestCase):
    """Testing `IEEEContext.encode()`"""

    def test_native(self, num_encodings: int = 10_000):
        # sample 10_000 random floating-point values
        random.seed(1)
        xs: list[Float] = []
        for _ in range(num_encodings):
            s = random.choice([False, True])
            exp = random.randint(FP64.expmin, FP64.expmax)
            c = random.randint(0, (1 << FP64.pmax) - 1)
            x = Float(s, exp, c, ctx=FP64)
            assert FP64.is_representable(x)
            xs.append(x)
        # run encoding
        for x in xs:
            i = FP64.encode(x)
            self.assertIsInstance(i, int, f'x={x}, i={i}')
            self.assertGreaterEqual(i, 0, f'x={x}, i={i}')
            self.assertLess(i, 1 << FP64.nbits, f'x={x}, i={i}')

    def test_small(self, es_max: int = 6, nbits_max: int = 8):
        # iterate over possible contexts
        for es in range(2, es_max+1):
            for nbits in range(es + 2, nbits_max+1):
                ctx = IEEEContext(es, nbits, RM.RNE)
                # for ctx, encode all possible values
                for s in (True, False):
                    for exp in range(ctx.expmin, ctx.expmax + 1):
                        for c in range(0, 1 << ctx.pmax - 1):
                            x = Float(s, exp, c, ctx=ctx)
                            assert ctx.is_representable(x)

                            i = ctx.encode(x)
                            self.assertIsInstance(i, int, f'x={x}, i={i}')
                            self.assertGreaterEqual(i, 0, f'x={x}, i={i}')
                            self.assertLess(i, 1 << ctx.nbits, f'x={x}, i={i}')


class EncodeRoundTripTestCase(unittest.TestCase):
    """Ensure `IEEEContext.decode()` and `IEEEContext.encode()` roundtrips."""

    def test_native(self, num_encodings: int = 10_000):
        # sample 10_000 random encodings
        random.seed(1)
        encodings = [random.randint(0, (1 << FP64.nbits) - 1) for _ in range(num_encodings)]
        # run decode
        for i in encodings:
            x = FP64.decode(i)
            j = FP64.encode(x)
            if x.isnan:
                # mapped to +/-qNaN(0)
                y = FP64.decode(j)
                self.assertTrue(y.isnan, f'i={i}, j={j}, y={y}')
            else:
                self.assertEqual(i, j, f'i={i}, j={j}')

    def test_small(self, es_max: int = 6, nbits_max: int = 8):
        # iterate over possible contexts
        for es in range(2, es_max+1):
            for nbits in range(es + 2, nbits_max+1):
                ctx = IEEEContext(es, nbits, RM.RNE)
                # for ctx, decode all possible encodings
                for i in range(1 << ctx.nbits):
                    x = ctx.decode(i)
                    j = ctx.encode(x)
                    if x.isnan:
                        # mapped to +/-qNaN(0)
                        y = ctx.decode(j)
                        self.assertTrue(y.isnan, f'i={i}, j={j}, y={y}')
                    else:
                        self.assertEqual(i, j, f'i={i}, j={j}')
