import fpy2 as fp
import math
import unittest

from hypothesis import given, strategies as st

class TestFloatMethods(unittest.TestCase):

    @given(st.integers())
    def test_from_int(self, x):
        y = fp.Float.from_int(x)
        self.assertIsInstance(y, fp.Float)
        self.assertEqual(x, int(y))

    @given(st.floats(allow_nan=True, allow_infinity=True, allow_subnormal=True))
    def test_from_float(self, x):
        y = fp.Float.from_float(x)
        self.assertIsInstance(y, fp.Float)
        z = float(y)
        self.assertTrue(math.isnan(z) if math.isnan(x) else z == x)
