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


    @given(real_floats(prec=64, exp_max=512, exp_min=-512), st.integers(-512, 512))
    def test_split(self, x: fp.RealFloat, n: int):
        hi, lo = x.split(n)
        self.assertIsInstance(hi, fp.RealFloat)
        self.assertIsInstance(lo, fp.RealFloat)
        self.assertEqual(x, hi + lo, f'x={x}, n={n}, hi={hi}, lo={lo}')
        self.assertTrue(hi.is_more_significant(n), f'x={x}, n={n}, hi={hi}, lo={lo}')
        self.assertLessEqual(lo.e, n, f'x={x}, n={n}, hi={hi}, lo={lo}')

    @given(
        real_floats(prec=128, exp_min=-512, exp_max=512),
        st.one_of(st.integers(0, 64), st.none()),
        st.one_of(st.integers(-512, 512), st.none())
    )
    def test_normalize(self, x: fp.RealFloat, p: int | None, n: int | None):
        try:
            y = x.normalize(p=p, n=n)
            self.assertIsInstance(y, fp.RealFloat)
            self.assertEqual(x, y, f'x={x}, p={p}, n={n}, y={y}')
            if p is not None:
                self.assertLessEqual(y.p, p, f'x={x}, p={p}, n={n}, y={y}')
            if n is not None:
                self.assertGreater(y.exp, n, f'x={x}, p={p}, n={n}, y={y}')
        except ValueError:
            self.assertTrue(p is not None or n is not None, f'x={x}, p={p}, n={n}')

            # compute the split point
            match p, n:
                case int(), None:
                    n = x.e - p
                case None, int():
                    pass
                case int(), int():
                    n = max(x.e - p, n)
                case _:
                    raise RuntimeError('unreachable')

            _, lo = x.split(n)
            self.assertNotEqual(lo, 0, f'x={x}, p={p}, n={n}')


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
