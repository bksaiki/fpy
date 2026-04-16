import fpy2 as fp
import math

from fractions import Fraction
from hypothesis import assume, given, strategies as st

from ...generators import real_floats, rounding_modes


class TestRealFloatConstructors():
    """Testing `RealFloat` constructors."""

    @given(st.integers())
    def test_from_int(self, a: int):
        x = fp.RealFloat.from_int(a)
        assert isinstance(x, fp.RealFloat)
        assert x == a

    @given(st.floats(allow_infinity=False, allow_nan=False))
    def test_from_float(self, a: float):
        x = fp.RealFloat.from_float(a)
        assert isinstance(x, fp.RealFloat)
        assert x == a

    @given(st.fractions(min_value=-1e6, max_value=1e6, max_denominator=1_000_000).filter(
        lambda x: fp.utils.is_dyadic(x)
    ))
    def test_from_rational(self, a: Fraction):
        x = fp.RealFloat.from_rational(a)
        assert isinstance(x, fp.RealFloat)
        assert x == a


class TestRealFloatReprMethods():
    """Testing `RealFloat` representation methods"""

    def test_is_more_significant(self):
        x = fp.RealFloat(c=7, exp=0)
        assert x.is_more_significant(-2)
        assert x.is_more_significant(-1)
        assert not x.is_more_significant(0)
        assert not x.is_more_significant(1)

        y = fp.RealFloat(c=3, exp=-2)
        assert y.is_more_significant(-4)
        assert y.is_more_significant(-3)
        assert not y.is_more_significant(-2)
        assert not y.is_more_significant(-1)
        assert not y.is_more_significant(0)


    @given(real_floats(prec_max=64, exp_max=512, exp_min=-512), st.integers(-512, 512))
    def test_split(self, x: fp.RealFloat, n: int):
        hi, lo = x.split(n)
        assert isinstance(hi, fp.RealFloat)
        assert isinstance(lo, fp.RealFloat)
        assert x == hi + lo, f'x={x}, n={n}, hi={hi}, lo={lo}'
        assert hi.is_more_significant(n), f'x={x}, n={n}, hi={hi}, lo={lo}'
        assert lo.e <= n, f'x={x}, n={n}, hi={hi}, lo={lo}'

    @given(
        real_floats(prec_max=128, exp_min=-512, exp_max=512),
        st.one_of(st.integers(0, 64), st.none()),
        st.one_of(st.integers(-512, 512), st.none())
    )
    def test_normalize(self, x: fp.RealFloat, p: int | None, n: int | None):
        try:
            y = x.normalize(p=p, n=n)
            assert isinstance(y, fp.RealFloat)
            assert x == y, f'x={x}, p={p}, n={n}, y={y}'
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

    @given(
        real_floats(prec_max=128, exp_min=-512, exp_max=512),
        st.one_of(st.integers(1, 64), st.none()),
        st.one_of(st.integers(-512, 512), st.none())
    )
    def test_next_away_zero(self, x: fp.RealFloat, p: int | None, n: int | None):
        assume(n is None or x.exp > n)
        assume(p is None or x.p <= p)
        y = x.next_away_zero(p=p, n=n)
        assert isinstance(y, fp.RealFloat)
        assert y != 0, f'x={x}, p={p}, n={n}, y={y}'
        assert abs(y) > abs(x), f'x={x}, p={p}, n={n}, y={y}'
        if p is not None:
            assert y.p <= p, f'x={x}, p={p}, n={n}, y={y}'
        if n is not None:
            assert y.exp > n, f'x={x}, p={p}, n={n}, y={y}'

    @given(
        real_floats(prec_max=128, exp_min=-512, exp_max=512),
        st.one_of(st.integers(1, 64), st.none()),
        st.one_of(st.integers(-512, 512), st.none())
    )
    def test_next_towards_zero(self, x: fp.RealFloat, p: int | None, n: int | None):
        assume(n is None or x.exp > n)
        assume(p is None or x.p <= p)
        assume(x != 0)
        y = x.next_towards_zero(p=p, n=n)
        assert isinstance(y, fp.RealFloat)
        assert abs(y) < abs(x), f'x={x}, p={p}, n={n}, y={y}'
        if p is not None:
            assert y.p <= p, f'x={x}, p={p}, n={n}, y={y}'
        if n is not None:
            assert y.exp > n, f'x={x}, p={p}, n={n}, y={y}'

    @given(
        real_floats(prec_max=128, exp_min=-512, exp_max=512),
        st.one_of(st.integers(1, 64), st.none()),
        st.one_of(st.integers(-512, 512), st.none())
    )
    def test_next_up(self, x: fp.RealFloat, p: int | None, n: int | None):
        assume(n is None or x.exp > n)
        assume(p is None or x.p <= p)
        y = x.next_up(p=p, n=n)
        assert isinstance(y, fp.RealFloat)
        assert y > x, f'x={x}, p={p}, n={n}, y={y}'
        if p is not None:
            assert y.p <= p, f'x={x}, p={p}, n={n}, y={y}'
        if n is not None:
            assert y.exp > n, f'x={x}, p={p}, n={n}, y={y}'

    @given(
        real_floats(prec_max=128, exp_min=-512, exp_max=512),
        st.one_of(st.integers(1, 64), st.none()),
        st.one_of(st.integers(-512, 512), st.none())
    )
    def test_next_down(self, x: fp.RealFloat, p: int | None, n: int | None):
        assume(n is None or x.exp > n)
        assume(p is None or x.p <= p)
        y = x.next_down(p=p, n=n)
        assert isinstance(y, fp.RealFloat)
        assert y < x, f'x={x}, p={p}, n={n}, y={y}'
        if p is not None:
            assert y.p <= p, f'x={x}, p={p}, n={n}, y={y}'
        if n is not None:
            assert y.exp > n, f'x={x}, p={p}, n={n}, y={y}'


class TestRealFloatArithmetic():
    """Testing `RealFloat` arithmetic operations."""

    @given(st.floats(allow_infinity=False, allow_nan=False))
    def test_abs(self, a: float):
        expect = abs(Fraction(a))
        actual = abs(fp.RealFloat.from_float(a))
        assert actual == expect

    @given(st.floats(allow_infinity=False, allow_nan=False))
    def test_neg(self, a: float):
        expect = -Fraction(a)
        actual = -fp.RealFloat.from_float(a)
        assert actual == expect

    @given(st.floats(allow_infinity=False, allow_nan=False))
    def test_trunc(self, a: float):
        expect = math.trunc(Fraction(a))
        actual = math.trunc(fp.RealFloat.from_float(a))
        assert actual == expect

    @given(st.floats(allow_infinity=False, allow_nan=False))
    def test_floor(self, a: float):
        expect = math.floor(Fraction(a))
        actual = math.floor(fp.RealFloat.from_float(a))
        assert actual == expect

    @given(st.floats(allow_infinity=False, allow_nan=False))
    def test_ceil(self, a: float):
        expect = math.ceil(Fraction(a))
        actual = math.ceil(fp.RealFloat.from_float(a))
        assert actual == expect

    @given(st.floats(allow_infinity=False, allow_nan=False))
    def test_round(self, a: float):
        expect = round(Fraction(a))
        actual = round(fp.RealFloat.from_float(a))
        assert actual == expect

    @given(
        st.floats(allow_infinity=False, allow_nan=False),
        st.floats(allow_infinity=False, allow_nan=False)
    )
    def test_add(self, a: float, b: float):
        expect = Fraction(a) + Fraction(b)
        actual = fp.RealFloat.from_float(a) + fp.RealFloat.from_float(b)
        assert actual == expect

    @given(
        st.floats(allow_infinity=False, allow_nan=False),
        st.integers()
        | st.floats(allow_infinity=False, allow_nan=False)
        | st.fractions().filter(lambda x: fp.utils.is_dyadic(x))
    )
    def test_add_mixed(self, a: float, b: int | float | Fraction):
        actual = fp.RealFloat.from_float(a) + b
        assert isinstance(actual, fp.RealFloat)

    @given(
        st.floats(allow_infinity=False, allow_nan=False),
        st.floats(allow_infinity=False, allow_nan=False)
    )
    def test_sub(self, a: float, b: float):
        expect = Fraction(a) - Fraction(b)
        actual = fp.RealFloat.from_float(a) - fp.RealFloat.from_float(b)
        assert actual == expect

    @given(
        st.floats(allow_infinity=False, allow_nan=False),
        st.integers()
        | st.floats(allow_infinity=False, allow_nan=False)
        | st.fractions().filter(lambda x: fp.utils.is_dyadic(x))
    )
    def test_sub_mixed(self, a: float, b: int | float | Fraction):
        actual = fp.RealFloat.from_float(a) - b
        assert isinstance(actual, fp.RealFloat)

    @given(
        st.floats(allow_infinity=False, allow_nan=False),
        st.floats(allow_infinity=False, allow_nan=False)
    )
    def test_mul(self, a: float, b: float):
        expect = Fraction(a) * Fraction(b)
        actual = fp.RealFloat.from_float(a) * fp.RealFloat.from_float(b)
        assert actual == expect

    @given(
        st.floats(allow_infinity=False, allow_nan=False),
        st.integers()
        | st.floats(allow_infinity=False, allow_nan=False)
        | st.fractions().filter(lambda x: fp.utils.is_dyadic(x))
    )
    def test_mul_mixed(self, a: float, b: int | float | Fraction):
        actual = fp.RealFloat.from_float(a) * b
        assert isinstance(actual, fp.RealFloat)

    @given(
        st.floats(allow_infinity=False, allow_nan=False).filter(lambda x: x != 0.0),
        st.integers(min_value=0, max_value=1000)
    )
    def test_pow(self, a: float, b: int):
        expect = Fraction(a) ** b
        actual = fp.RealFloat.from_float(a) ** b
        assert actual == expect
