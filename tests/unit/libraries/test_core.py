import unittest

from fpy2 import *
from fpy2.libraries.core import logb

class TestCore(unittest.TestCase):
    """Testing core functionality"""

    def assertNumEqual(self, a: Float, b: Float):
        """Assert that two numbers are equal, handling NaN and Inf"""
        if a.isnan or b.isnan:
            return a.isnan == b.isnan
        else:
            return a == b

    def test_logb(self):
        """Testing `logb` primitive"""
        self.assertNumEqual(logb(Float.from_int(1)), Float.from_int(0))
        self.assertNumEqual(logb(Float.from_int(2)), Float.from_int(1))
        self.assertNumEqual(logb(Float.from_int(4)), Float.from_int(2))
        self.assertNumEqual(logb(Float.from_int(8)), Float.from_int(3))

        self.assertNumEqual(logb(Float(isnan=True)), Float(isnan=True))
        self.assertNumEqual(logb(Float(isinf=True)), Float(isinf=True))
        self.assertNumEqual(logb(Float(s=True, isinf=True)), Float(isinf=True))
        self.assertNumEqual(logb(Float.from_int(0)), Float(isinf=True, s=True))
