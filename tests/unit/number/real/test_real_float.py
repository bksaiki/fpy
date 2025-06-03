import unittest

from fpy2 import RealFloat

class TestRealFloatMethods(unittest.TestCase):
    """Testing `RealFloat.is_more_significant()`"""

    def test_is_more_significant(self):
        x = RealFloat(c=7, exp=0)
        self.assertTrue(x.is_more_significant(-2))
        self.assertTrue(x.is_more_significant(-1))
        self.assertFalse(x.is_more_significant(0))
        self.assertFalse(x.is_more_significant(1))

        y = RealFloat(c=3, exp=-2)
        self.assertTrue(y.is_more_significant(-4))
        self.assertTrue(y.is_more_significant(-3))
        self.assertFalse(y.is_more_significant(-2))
        self.assertFalse(y.is_more_significant(-1))
        self.assertFalse(y.is_more_significant(0))
