import pytest

from fpy2 import *
from fpy2.libraries.core import split, logb, modf, ldexp, isinteger

class TestCore():
    """Testing core functionality"""

    def assertNumEqual(self, a: Float, b: Float):
        """Assert that two numbers are equal, handling NaN and Inf"""
        if a.isnan or b.isnan:
            assert a.isnan == b.isnan
        else:
            assert a == b, f"Expected {a} to be equal to {b}"

    def assertArrayEqual(self, a: list, b: list):
        """Assert that two arrays are equal, handling NaN and Inf"""
        return len(a) == len(b) and all(self.assertNumEqual(x, y) for x, y in zip(a, b))

    def test_split(self):
        """Testing `split` function"""
        self.assertArrayEqual(split(Float.from_int(0), Float.from_int(-1)), [Float.from_int(0), Float.from_int(0)])
        self.assertArrayEqual(split(Float.from_float(-0.0), Float.from_int(-1)), [Float.from_float(-0.0), Float.from_float(-0.0)])
        self.assertArrayEqual(split(Float.from_int(1), Float.from_int(-1)), [Float.from_int(1), Float.from_int(0)])
        self.assertArrayEqual(split(Float.from_float(1.5), Float.from_int(-1)), [Float.from_int(1), Float.from_float(0.5)])

        self.assertArrayEqual(split(Float(isnan=True), Float.from_int(0)), [Float(isnan=True), Float(isnan=True)])
        self.assertArrayEqual(split(Float(isinf=True), Float.from_int(0)), [Float(isinf=True), Float(isinf=True)])
        self.assertArrayEqual(split(Float(s=True, isinf=True), Float.from_int(0)), [Float(s=True, isinf=True), Float(isinf=True, s=True)])

    def test_logb(self):
        """Testing `logb` function"""
        self.assertNumEqual(logb(Float.from_int(1)), Float.from_int(0))
        self.assertNumEqual(logb(Float.from_int(2)), Float.from_int(1))
        self.assertNumEqual(logb(Float.from_int(4)), Float.from_int(2))
        self.assertNumEqual(logb(Float.from_int(8)), Float.from_int(3))

        self.assertNumEqual(logb(Float(isnan=True)), Float(isnan=True))
        self.assertNumEqual(logb(Float(isinf=True)), Float(isinf=True))
        self.assertNumEqual(logb(Float(s=True, isinf=True)), Float(isinf=True))
        self.assertNumEqual(logb(Float.from_int(0)), Float(isinf=True, s=True))

    def test_modf(self):
        """Testing `modf` function"""
        self.assertArrayEqual(modf(Float.from_int(0)), [Float.from_int(0), Float.from_int(0)])
        self.assertArrayEqual(modf(Float.from_float(-0.0)), [Float.from_float(-0.0), Float.from_float(-0.0)])
        self.assertArrayEqual(modf(Float.from_int(1)), [Float.from_int(1), Float.from_int(0)])
        self.assertArrayEqual(modf(Float.from_float(1.5)), [Float.from_int(1), Float.from_float(0.5)])

        self.assertArrayEqual(modf(Float(isnan=True)), [Float(isnan=True), Float(isnan=True)])
        self.assertArrayEqual(modf(Float(isinf=True, s=True)), [Float(s=True), Float(isinf=True, s=True)])
        self.assertArrayEqual(modf(Float(s=True, isinf=True)), [Float(s=True), Float(isinf=True, s=True)])
        self.assertArrayEqual(modf(Float(s=False, isinf=True)), [Float(s=False), Float(isinf=True, s=False)])

    def test_ldexp(self):
        """Testing `ldexp` function"""
        self.assertNumEqual(ldexp(Float.from_int(1), Float.from_int(0)), Float.from_int(1))
        self.assertNumEqual(ldexp(Float.from_int(1), Float.from_int(1)), Float.from_int(2))
        self.assertNumEqual(ldexp(Float.from_int(1), Float.from_int(-1)), Float.from_float(0.5))

        with pytest.raises(ValueError):
            ldexp(Float.from_int(1), Float.from_float(0.5))

        self.assertNumEqual(ldexp(Float(isnan=True), Float.from_int(0)), Float(isnan=True))
        self.assertNumEqual(ldexp(Float(isinf=True), Float.from_int(0)), Float(isinf=True))
        self.assertNumEqual(ldexp(Float(s=True, isinf=True), Float.from_int(0)), Float(s=True, isinf=True))

    def test_isinteger(self):
        """Testing `isinteger` function"""
        assert isinteger(Float.from_int(0))
        assert isinteger(Float.from_int(1))
        assert not isinteger(Float.from_float(1.5))
        assert not isinteger(Float(isnan=True))
        assert not isinteger(Float(isinf=True))
        assert not isinteger(Float(s=True, isinf=True))
