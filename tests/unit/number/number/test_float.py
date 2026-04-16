import pytest
import fpy2 as fp
import math

from fractions import Fraction
from hypothesis import given, strategies as st

from ...generators import floats


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


class FloatTestCast():

    def assertEqualOrNan(
        self,
        first: float | int | Fraction | fp.Float,
        second: float | int | Fraction | fp.Float,
        msg=None
    ):
        a = _cvt_to_real(first)
        b = _cvt_to_real(second)
        if a.isnan:
            assert b.isnan, msg
        else:
            assert a == b, msg


class TestFloatConstructors(FloatTestCast):

    @given(st.integers())
    def test_from_int(self, x):
        y = fp.Float.from_int(x)
        assert isinstance(y, fp.Float)
        assert x == int(y)

    @given(st.floats(allow_nan=True, allow_infinity=True, allow_subnormal=True))
    def test_from_float(self, x):
        y = fp.Float.from_float(x)
        assert isinstance(y, fp.Float)
        z = float(y)
        assert math.isnan(z) if math.isnan(x) else z == x

    @given(st.fractions(min_value=-1, max_value=1, max_denominator=1000))
    def test_from_rational(self, x):
        if fp.utils.is_dyadic(x):
            y = fp.Float.from_rational(x)
            assert isinstance(y, fp.Float)
            assert x == y
        else:
            with pytest.raises(ValueError):
                fp.Float.from_rational(x)


class TestFloatReprMethods(FloatTestCast):

    @given(st.integers())
    def test_as_int(self, x):
        y = fp.Float.from_int(x)
        assert x == int(y)

    @given(st.floats(allow_nan=False, allow_infinity=False, allow_subnormal=True))
    def test_as_float(self, x):
        y = fp.Float.from_float(x)
        assert x == float(y)

    @given(st.fractions(min_value=-1, max_value=1, max_denominator=1000))
    def test_as_rational(self, x):
        if fp.utils.is_dyadic(x):
            y = fp.Float.from_rational(x)
            assert x == y.as_rational()
        else:
            with pytest.raises(ValueError):
                fp.Float.from_rational(x)

    @given(floats(prec_max=64, exp_max=512, exp_min=-512), st.integers(-512, 512))
    def test_split(self, x: fp.Float, n: int):
        hi, lo = x.split(n)
        assert isinstance(hi, fp.Float)
        assert isinstance(lo, fp.Float)
        if x.is_nar():
            self.assertEqualOrNan(x, hi, f'x={x}, n={n}, hi={hi}, lo={lo}')
            self.assertEqualOrNan(x, lo, f'x={x}, n={n}, hi={hi}, lo={lo}')
        else:
            assert x == hi + lo, f'x={x}, n={n}, hi={hi}, lo={lo}'
            assert hi.is_more_significant(n), f'x={x}, n={n}, hi={hi}, lo={lo}'
            assert lo.e <= n, f'x={x}, n={n}, hi={hi}, lo={lo}'

    @given(
        floats(prec_max=128, exp_min=-512, exp_max=512, ctx=fp.REAL),
        st.one_of(st.integers(0, 64), st.none()),
        st.one_of(st.integers(-512, 512), st.none())
    )
    def test_normalize(self, x: fp.Float, p: int | None, n: int | None):
        try:
            y = x.normalize(p=p, n=n)
            assert isinstance(y, fp.Float)
            self.assertEqualOrNan(x, y, f'x={x}, p={p}, n={n}, y={y}')
            if not x.is_nar():
                if p is not None:
                    assert y.p <= p, f'x={x}, p={p}, n={n}, y={y}'
                if n is not None:
                    assert y.exp > n, f'x={x}, p={p}, n={n}, y={y}'
        except ValueError:
            assert p is not None or n is not None, f'x={x}, p={p}, n={n}'

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
            assert lo != 0, f'x={x}, p={p}, n={n}'


class TestFloatArithmetic(FloatTestCast):

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
            with pytest.raises(ValueError):
                math.trunc(x)
        else:
            expect = math.trunc(_cvt_to_fraction(a))
            actual = math.trunc(x)
            assert actual == expect

    @given(st.floats())
    def test_floor(self, a: float):
        x = fp.Float.from_float(a)
        if x.is_nar():
            with pytest.raises(ValueError):
                math.floor(x)
        else:
            expect = math.floor(_cvt_to_fraction(a))
            actual = math.floor(x)
            assert actual == expect

    @given(st.floats())
    def test_ceil(self, a: float):
        x = fp.Float.from_float(a)
        if x.is_nar():
            with pytest.raises(ValueError):
                math.ceil(x)
        else:
            expect = math.ceil(_cvt_to_fraction(a))
            actual = math.ceil(x)
            assert actual == expect

    @given(st.floats())
    def test_round(self, a: float):
        x = fp.Float.from_float(a)
        if x.is_nar():
            with pytest.raises(ValueError):
                round(x)
        else:
            expect = round(_cvt_to_fraction(a))
            actual = round(x)
            assert actual == expect

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
