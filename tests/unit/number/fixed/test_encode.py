import unittest

import fpy2 as fp

class EncodeTestCase(unittest.TestCase):
    """Testing `IEEEContext.encode()`"""

    def test_unsigned(self):
        ctx = fp.FixedContext(False, 0, 4)
        for i in range(1 << ctx.nbits):
            x = fp.Float.from_int(i, ctx=ctx)
            encoded = ctx.encode(x)
            self.assertIsInstance(encoded, int, f'x={x}, encoded={encoded}')
            self.assertEqual(encoded, i, f'x={x}, encoded={encoded}')

    def test_signed(self):
        ctx = fp.FixedContext(True, 0, 4)
        # non-negative values
        for i in range(1 << (ctx.nbits - 1)):
            x = fp.Float.from_int(i, ctx=ctx)
            encoded = ctx.encode(x)
            self.assertIsInstance(encoded, int, f'x={x}, encoded={encoded}')
            self.assertEqual(encoded, i, f'x={x}, encoded={encoded}')
        # negative values
        for i in range(-(1 << (ctx.nbits - 1)), 0):
            x = fp.Float.from_int(i, ctx=ctx)
            encoded = ctx.encode(x)
            self.assertIsInstance(encoded, int, f'x={x}, encoded={encoded}')
            self.assertEqual(encoded, (1 << (ctx.nbits - 1)) - abs(i), f'x={x}, encoded={encoded}')

class DecodeTestCase(unittest.TestCase):
    """Testing `IEEEContext.decode()`"""

    def test_unsigned(self):
        ctx = fp.FixedContext(False, 0, 4)
        for i in range(1 << ctx.nbits):
            decoded = ctx.decode(i)
            self.assertIsInstance(decoded, fp.Float, f'i={i}, decoded={decoded}')
            self.assertEqual(decoded.c, i, f'i={i}, decoded={decoded}')

    def test_signed(self):
        ctx = fp.FixedContext(True, 0, 4)
        # non-negative values
        for i in range(1 << (ctx.nbits - 1)):
            decoded = ctx.decode(i)
            self.assertIsInstance(decoded, fp.Float, f'i={i}, decoded={decoded}')
            self.assertEqual(decoded.c, i, f'i={i}, decoded={decoded}')
        # negative values
        for i in range(1 << (ctx.nbits - 1), 1 << ctx.nbits):
            decoded = ctx.decode(i)
            self.assertIsInstance(decoded, fp.Float, f'i={i}, decoded={decoded}')
            self.assertEqual(decoded.c, i - (1 << (ctx.nbits - 1)), f'i={i}, decoded={decoded}')
