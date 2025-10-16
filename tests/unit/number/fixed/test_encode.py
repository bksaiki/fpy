import unittest

import fpy2 as fp

class EncodeTestCase(unittest.TestCase):
    """Testing `FixedContext.encode()`"""

    def test_encode_u4(self):
        ctx = fp.FixedContext(False, 0, 4)
        for i in range(1 << ctx.nbits):
            x = fp.Float.from_int(i, ctx=ctx)
            encoded = ctx.encode(x)
            self.assertIsInstance(encoded, int, f'x={x}, encoded={encoded}')
            self.assertEqual(encoded, i, f'x={x}, encoded={encoded}')

    def test_encode_u4_scale2(self):
        ctx = fp.FixedContext(False, 2, 4)
        for i in range(1 << ctx.nbits):
            x = fp.Float(False, 2, i, ctx=ctx)
            encoded = ctx.encode(x)
            self.assertIsInstance(encoded, int, f'x={x}, encoded={encoded}')
            self.assertEqual(encoded, i, f'x={x}, encoded={encoded}')

    def test_encode_s4(self):
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

    def test_encode_s4_scale2(self):
        ctx = fp.FixedContext(True, 2, 4)
        # non-negative values
        for i in range(1 << (ctx.nbits - 1)):
            x = fp.Float(False, 2, i, ctx=ctx)
            encoded = ctx.encode(x)
            self.assertIsInstance(encoded, int, f'x={x}, encoded={encoded}')
            self.assertEqual(encoded, i, f'x={x}, encoded={encoded}')
        # negative values
        for i in range(-(1 << (ctx.nbits - 1)), 0):
            x = fp.Float(True, 2, abs(i), ctx=ctx)
            encoded = ctx.encode(x)
            self.assertIsInstance(encoded, int, f'x={x}, encoded={encoded}')
            self.assertEqual(encoded, (1 << (ctx.nbits - 1)) - abs(i), f'x={x}, encoded={encoded}')


class DecodeTestCase(unittest.TestCase):
    """Testing `FixedContext.decode()`"""

    def test_decode_u4(self):
        ctx = fp.FixedContext(False, 0, 4)
        for i in range(1 << ctx.nbits):
            decoded = ctx.decode(i)
            self.assertIsInstance(decoded, fp.Float, f'i={i}, decoded={decoded}')
            self.assertEqual(int(decoded), i, f'i={i}, decoded={decoded}')

    def test_decode_u4_scale2(self):
        ctx = fp.FixedContext(False, 2, 4)
        for i in range(1 << ctx.nbits):
            decoded = ctx.decode(i)
            self.assertIsInstance(decoded, fp.Float, f'i={i}, decoded={decoded}')
            self.assertEqual(decoded, fp.Float(exp=2, c=i), f'i={i}, decoded={decoded}')

    def test_decode_s4(self):
        ctx = fp.FixedContext(True, 0, 4)
        # non-negative values
        for i in range(1 << (ctx.nbits - 1)):
            decoded = ctx.decode(i)
            self.assertIsInstance(decoded, fp.Float, f'i={i}, decoded={decoded}')
            self.assertEqual(int(decoded), i, f'i={i}, decoded={decoded}')
        # negative values
        for i in range(1 << (ctx.nbits - 1), 1 << ctx.nbits):
            decoded = ctx.decode(i)
            self.assertIsInstance(decoded, fp.Float, f'i={i}, decoded={decoded}')
            self.assertEqual(int(decoded), (1 << (ctx.nbits - 1)) - i, f'i={i}, decoded={decoded}')

    def test_decode_s4_scale2(self):
        ctx = fp.FixedContext(True, 2, 4)
        # non-negative values
        for i in range(1 << (ctx.nbits - 1)):
            decoded = ctx.decode(i)
            self.assertIsInstance(decoded, fp.Float, f'i={i}, decoded={decoded}')
            self.assertEqual(decoded, fp.Float(False, 2, i), f'i={i}, decoded={decoded}')
        # negative values
        for i in range(1 << (ctx.nbits - 1), 1 << ctx.nbits):
            decoded = ctx.decode(i)
            self.assertIsInstance(decoded, fp.Float, f'i={i}, decoded={decoded}')
            self.assertEqual(decoded, fp.Float(True, 2, i - (1 << (ctx.nbits - 1))), f'i={i}, decoded={decoded}')
