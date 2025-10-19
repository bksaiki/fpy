import fpy2 as fp
import math
import unittest

from fractions import Fraction
from hypothesis import given, strategies as st


def _cvt_to_fraction(x: float | int | Fraction):
    match x:
        case float():
            if math.isnan(x) or math.isinf(x):
                return x
            else:
                return Fraction(x)
        case int():
            return Fraction(x)
        case Fraction():
            return x
        case _:
            raise TypeError(f"Unsupported type: {type(x)}")

def _cvt_to_real(x: float | int | Fraction | fp.Float):
    match x:
        case fp.Float():
            return x
        case float():
            return fp.Float.from_float(x)
        case int():
            return fp.Float.from_int(x)
        case Fraction():
            return fp.Float.from_rational(x)
        case _:
            raise TypeError(f"Unsupported type: {type(x)}")


class TestFloatMethods(unittest.TestCase):

    def assertEqualOrNan(
        self,
        first: float | int | Fraction | fp.Float,
        second: float | int | Fraction | fp.Float,
        msg=None
    ):
        a = _cvt_to_real(first)
        b = _cvt_to_real(second)
        if a.isnan:
            self.assertTrue(b.isnan, msg=msg)
        else:
            self.assertEqual(a, b, msg=msg)

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

    @given(st.floats())
    def test_abs(self, a: float):
        expect = abs(_cvt_to_fraction(a))
        actual = abs(fp.Float.from_float(a))
        self.assertEqualOrNan(actual, expect)

    @given(st.floats())
    def test_neg(self, a: float):
        expect = -_cvt_to_fraction(a)
        actual = -fp.Float.from_float(a)
        self.assertEqualOrNan(actual, expect)

    @given(st.floats())
    def test_trunc(self, a: float):
        x = fp.Float.from_float(a)
        if x.is_nar():
            with self.assertRaises(ValueError):
                math.trunc(x)
        else:
            expect = math.trunc(_cvt_to_fraction(a))
            actual = math.trunc(x)
            self.assertEqual(actual, expect)

    @given(st.floats())
    def test_floor(self, a: float):
        x = fp.Float.from_float(a)
        if x.is_nar():
            with self.assertRaises(ValueError):
                math.floor(x)
        else:
            expect = math.floor(_cvt_to_fraction(a))
            actual = math.floor(x)
            self.assertEqual(actual, expect)

    @given(st.floats())
    def test_ceil(self, a: float):
        x = fp.Float.from_float(a)
        if x.is_nar():
            with self.assertRaises(ValueError):
                math.ceil(x)
        else:
            expect = math.ceil(_cvt_to_fraction(a))
            actual = math.ceil(x)
            self.assertEqual(actual, expect)

    @given(st.floats())
    def test_round(self, a: float):
        x = fp.Float.from_float(a)
        if x.is_nar():
            with self.assertRaises(ValueError):
                round(x)
        else:
            expect = round(_cvt_to_fraction(a))
            actual = round(x)
            self.assertEqual(actual, expect)

    @given(st.floats(), st.floats())
    def test_add(self, a: float, b: float):
        expect = _cvt_to_fraction(a) + _cvt_to_fraction(b)
        actual = fp.Float.from_float(a) + fp.Float.from_float(b)
        self.assertEqualOrNan(actual, expect)

    @given(
        st.floats(),
        st.integers()
        | st.floats()
        | st.fractions().filter(lambda x: fp.utils.is_dyadic(x))
    )
    def test_add_mixed(self, a: float, b: int | float | Fraction):
        expect = _cvt_to_fraction(a) + _cvt_to_fraction(b)
        actual = fp.Float.from_float(a) + b
        self.assertEqualOrNan(actual, expect)

    @given(st.floats(), st.floats())
    def test_sub(self, a: float, b: float):
        expect = _cvt_to_fraction(a) - _cvt_to_fraction(b)
        actual = fp.Float.from_float(a) - fp.Float.from_float(b)
        self.assertEqualOrNan(actual, expect)

    @given(
        st.floats(),
        st.integers()
        | st.floats()
        | st.fractions().filter(lambda x: fp.utils.is_dyadic(x))
    )
    def test_sub_mixed(self, a: float, b: int | float | Fraction):
        expect = _cvt_to_fraction(a) - _cvt_to_fraction(b)
        actual = fp.Float.from_float(a) - b
        self.assertEqualOrNan(actual, expect)

    @given(st.floats(), st.floats())
    def test_mul(self, a: float, b: float):
        expect = _cvt_to_fraction(a) * _cvt_to_fraction(b)
        actual = fp.Float.from_float(a) * fp.Float.from_float(b)
        self.assertEqualOrNan(actual, expect)

    @given(
        st.floats(),
        st.integers()
        | st.floats()
        | st.fractions().filter(lambda x: fp.utils.is_dyadic(x))
    )
    def test_mul_mixed(self, a: float, b: int | float | Fraction):
        expect = _cvt_to_fraction(a) * _cvt_to_fraction(b)
        actual = fp.Float.from_float(a) * b
        self.assertEqualOrNan(actual, expect)
