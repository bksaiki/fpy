import unittest
import random

from fpy2.number import Float, IEEEContext, RM

def _maxval_ordinal(ctx: IEEEContext):
    # [emin, emax] + subnormal
    num_binades = ctx.emax - ctx.emin + 2
    return (num_binades << (ctx.pmax - 1)) - 1


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
        # run ordinal conversion
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

    def test_values_native(self):
        # rounding context for native Python floats
        fp64 = IEEEContext(11, 64, RM.RNE)

        tests = [
            (fp64.zero(), 0),
            (fp64.zero(True), 0),
            (fp64.minval(), 1),
            (fp64.minval(True), -1),
            (fp64.maxval(), _maxval_ordinal(fp64)),
            (fp64.maxval(True), -_maxval_ordinal(fp64))
        ]

        for x, expect in tests:
            i = fp64.to_ordinal(x)
            self.assertEqual(i, expect, f'x={x}, i={i}, expect={expect}')

    def test_values_small(self, es_max: int = 6, nbits_max: int = 8):
        # rounding context for native Python floats
        # iterate over possible contexts
        for es in range(2, es_max+1):
            for nbits in range(es + 2, nbits_max+1):
                ctx = IEEEContext(es, nbits, RM.RNE)

                tests = [
                    (ctx.zero(), 0),
                    (ctx.zero(True), 0),
                    (ctx.minval(), 1),
                    (ctx.minval(True), -1),
                    (ctx.maxval(), _maxval_ordinal(ctx)),
                    (ctx.maxval(True), -_maxval_ordinal(ctx))
                ]

                for x, expect in tests:
                    i = ctx.to_ordinal(x)
                    self.assertEqual(i, expect, f'x={x}, i={i}, expect={expect}')


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
        # run ordinal conversion
        for x in xs:
            i = fp64.to_ordinal(x)
            y = fp64.from_ordinal(i)
            self.assertIsInstance(y, Float, f'x={x}, i={i}, y={y}')
            self.assertEqual(x, y, f'x={x}, i={i}, y={y}')

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

                            # run ordinal conversion
                            i = ctx.to_ordinal(x)
                            y = ctx.from_ordinal(i)
                            self.assertIsInstance(y, Float, f'x={x}, i={i}, y={y}')
                            self.assertEqual(x, y, f'x={x}, i={i}, y={y}')


