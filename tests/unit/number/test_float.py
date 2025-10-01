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

    @given(st.fractions(min_value=-1, max_value=1, max_denominator=1000))
    def test_from_rational(self, x):
        if fp.utils.is_dyadic(x):
            y = fp.Float.from_rational(x)
            self.assertIsInstance(y, fp.Float)
            self.assertEqual(x, y)
        else:
            with self.assertRaises(ValueError):
                fp.Float.from_rational(x)

    @given(st.integers())
    def test_as_int(self, x):
        y = fp.Float.from_int(x)
        self.assertEqual(x, int(y))

    @given(st.floats(allow_nan=False, allow_infinity=False, allow_subnormal=True))
    def test_as_float(self, x):
        y = fp.Float.from_float(x)
        self.assertEqual(x, float(y))

    @given(st.fractions(min_value=-1, max_value=1, max_denominator=1000))
    def test_as_rational(self, x):
        if fp.utils.is_dyadic(x):
            y = fp.Float.from_rational(x)
            self.assertEqual(x, y.as_rational())
        else:
            with self.assertRaises(ValueError):
                fp.Float.from_rational(x)
