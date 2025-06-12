import unittest

from fpy2 import *
from fpy2.libraries.core import logb, modf

class TestCore(unittest.TestCase):
    """Testing core functionality"""

    def assertNumEqual(self, a: Float, b: Float):
        """Assert that two numbers are equal, handling NaN and Inf"""
        if a.isnan or b.isnan:
            return a.isnan == b.isnan
        else:
            return a == b

    def assertArrayEqual(self, a: NDArray, b: NDArray):
        """Assert that two arrays are equal, handling NaN and Inf"""
        return a.shape == b.shape and all(self.assertNumEqual(a[i], b[i]) for i in range(a.size))

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

    def test_modf(self):
        """Testing `modf` primitive"""
        self.assertArrayEqual(modf(Float.from_int(0)), NDArray((Float.from_int(0), Float.from_int(0))))
        self.assertArrayEqual(modf(Float.from_float(-0.0)), NDArray((Float.from_float(-0.0), Float.from_float(-0.0))))
        self.assertArrayEqual(modf(Float.from_int(1)), NDArray((Float.from_int(1), Float.from_int(0))))
        self.assertArrayEqual(modf(Float.from_float(1.5)), NDArray((Float.from_int(1), Float.from_float(0.5))))

        self.assertArrayEqual(modf(Float(isnan=True)), NDArray((Float(isnan=True), Float(isnan=True))))
        self.assertArrayEqual(modf(Float(isinf=True, s=True)), NDArray((Float(s=True), Float(isinf=True, s=True))))
        self.assertArrayEqual(modf(Float(s=True, isinf=True)), NDArray((Float(s=True), Float(isinf=True, s=True))))
        self.assertArrayEqual(modf(Float(s=False, isinf=True)), NDArray((Float(s=False), Float(isinf=True, s=False))))
