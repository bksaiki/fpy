import fpy2 as fp
import math
import unittest

from fractions import Fraction
from hypothesis import given, strategies as st

from ...generators import real_floats, rounding_modes


class TestRealFloatConstructors(unittest.TestCase):
    """Testing `RealFloat` constructors."""

    @given(st.integers())
    def test_from_int(self, a: int):
        x = fp.RealFloat.from_int(a)
        self.assertIsInstance(x, fp.RealFloat)
        self.assertEqual(x, a)

    @given(st.floats(allow_infinity=False, allow_nan=False))
    def test_from_float(self, a: float):
        x = fp.RealFloat.from_float(a)
        self.assertIsInstance(x, fp.RealFloat)
        self.assertEqual(x, a)

    @given(st.fractions(min_value=-1e6, max_value=1e6, max_denominator=1_000_000).filter(
        lambda x: fp.utils.is_dyadic(x)
    ))
    def test_from_rational(self, a: Fraction):
        x = fp.RealFloat.from_rational(a)
        self.assertIsInstance(x, fp.RealFloat)
        self.assertEqual(x, a)


class TestRealFloatReprMethods(unittest.TestCase):
    """Testing `RealFloat` representation methods"""

    def test_is_more_significant(self):
        x = fp.RealFloat(c=7, exp=0)
        self.assertTrue(x.is_more_significant(-2))
        self.assertTrue(x.is_more_significant(-1))
        self.assertFalse(x.is_more_significant(0))
        self.assertFalse(x.is_more_significant(1))

        y = fp.RealFloat(c=3, exp=-2)
        self.assertTrue(y.is_more_significant(-4))
        self.assertTrue(y.is_more_significant(-3))
        self.assertFalse(y.is_more_significant(-2))
        self.assertFalse(y.is_more_significant(-1))
        self.assertFalse(y.is_more_significant(0))



class TestRealFloatArithmetic(unittest.TestCase):
    """Testing `RealFloat` arithmetic operations."""

    @given(st.floats(allow_infinity=False, allow_nan=False))
    def test_abs(self, a: float):
        expect = abs(Fraction(a))
        actual = abs(fp.RealFloat.from_float(a))
        self.assertEqual(actual, expect)

    @given(st.floats(allow_infinity=False, allow_nan=False))
    def test_neg(self, a: float):
        expect = -Fraction(a)
        actual = -fp.RealFloat.from_float(a)
        self.assertEqual(actual, expect)

    @given(st.floats(allow_infinity=False, allow_nan=False))
    def test_trunc(self, a: float):
        expect = math.trunc(Fraction(a))
        actual = math.trunc(fp.RealFloat.from_float(a))
        self.assertEqual(actual, expect)

    @given(st.floats(allow_infinity=False, allow_nan=False))
    def test_floor(self, a: float):
        expect = math.floor(Fraction(a))
        actual = math.floor(fp.RealFloat.from_float(a))
        self.assertEqual(actual, expect)

    @given(st.floats(allow_infinity=False, allow_nan=False))
    def test_ceil(self, a: float):
        expect = math.ceil(Fraction(a))
        actual = math.ceil(fp.RealFloat.from_float(a))
        self.assertEqual(actual, expect)

    @given(st.floats(allow_infinity=False, allow_nan=False))
    def test_round(self, a: float):
        expect = round(Fraction(a))
        actual = round(fp.RealFloat.from_float(a))
        self.assertEqual(actual, expect)

    @given(
        st.floats(allow_infinity=False, allow_nan=False),
        st.floats(allow_infinity=False, allow_nan=False)
    )
    def test_add(self, a: float, b: float):
        expect = Fraction(a) + Fraction(b)
        actual = fp.RealFloat.from_float(a) + fp.RealFloat.from_float(b)
        self.assertEqual(actual, expect)

    @given(
        st.floats(allow_infinity=False, allow_nan=False),
        st.integers()
        | st.floats(allow_infinity=False, allow_nan=False)
        | st.fractions().filter(lambda x: fp.utils.is_dyadic(x))
    )
    def test_add_mixed(self, a: float, b: int | float | Fraction):
        actual = fp.RealFloat.from_float(a) + b
        self.assertIsInstance(actual, fp.RealFloat)

    @given(
        st.floats(allow_infinity=False, allow_nan=False),
        st.floats(allow_infinity=False, allow_nan=False)
    )
    def test_sub(self, a: float, b: float):
        expect = Fraction(a) - Fraction(b)
        actual = fp.RealFloat.from_float(a) - fp.RealFloat.from_float(b)
        self.assertEqual(actual, expect)

    @given(
        st.floats(allow_infinity=False, allow_nan=False),
        st.integers()
        | st.floats(allow_infinity=False, allow_nan=False)
        | st.fractions().filter(lambda x: fp.utils.is_dyadic(x))
    )
    def test_sub_mixed(self, a: float, b: int | float | Fraction):
        actual = fp.RealFloat.from_float(a) - b
        self.assertIsInstance(actual, fp.RealFloat)

    @given(
        st.floats(allow_infinity=False, allow_nan=False),
        st.floats(allow_infinity=False, allow_nan=False)
    )
    def test_mul(self, a: float, b: float):
        expect = Fraction(a) * Fraction(b)
        actual = fp.RealFloat.from_float(a) * fp.RealFloat.from_float(b)
        self.assertEqual(actual, expect)

    @given(
        st.floats(allow_infinity=False, allow_nan=False),
        st.integers()
        | st.floats(allow_infinity=False, allow_nan=False)
        | st.fractions().filter(lambda x: fp.utils.is_dyadic(x))
    )
    def test_mul_mixed(self, a: float, b: int | float | Fraction):
        actual = fp.RealFloat.from_float(a) * b
        self.assertIsInstance(actual, fp.RealFloat)

    @given(
        st.floats(allow_infinity=False, allow_nan=False).filter(lambda x: x != 0.0),
        st.integers(min_value=0, max_value=1000)
    )
    def test_pow(self, a: float, b: int):
        expect = Fraction(a) ** b
        actual = fp.RealFloat.from_float(a) ** b
        self.assertEqual(actual, expect)
