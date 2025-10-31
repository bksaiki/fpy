"""
Tests for `fpy2/ops.py` with `RealContext`.
"""

import fpy2 as fp
import math
import unittest

from fractions import Fraction
from hypothesis import given, strategies as st

from ..generators import floats

def _cvt_to_frac(x: int | float | Fraction | fp.RealFloat | fp.Float) -> float | Fraction:
    match x:
        case int():
            return Fraction(x)
        case float():
            return x
        case Fraction():
            return x
        case fp.Float():
            if x.isnan:
                return float('nan')
            elif x.isinf:
                return float('inf') if x.is_positive() else float('-inf')
            else:
                return x.as_rational()
        case fp.RealFloat():
            return x.as_rational()
        case _:
            raise TypeError(f'cannot convert {type(x)} to Fraction or float')

def _ceil(x: float | Fraction):
    if math.isnan(x):
        return float('nan')
    elif math.isinf(x):
        return x
    else:
        return math.ceil(x)

def _floor(x: float | Fraction):
    if math.isnan(x):
        return float('nan')
    elif math.isinf(x):
        return x
    else:
        return math.floor(x)

def _trunc(x: float | Fraction):
    if math.isnan(x):
        return float('nan')
    elif math.isinf(x):
        return x
    else:
        return math.trunc(x)

@st.composite
def number(
    draw,
    prec: int = 8,
    exp_min: int = -10,
    exp_max: int = 10
):
    """
    Returns a strategy for generating either a
    - `int`,
    - `Fraction`,
    - `Float`,
    - or `RealFloat`.
    """
    choice = draw(st.integers(min_value=0, max_value=2))
    if choice == 0:
        return draw(st.integers())
    elif choice == 2:
        return draw(st.fractions())
    else:
        return draw(floats(prec_max=prec, exp_min=exp_min, exp_max=exp_max, allow_nan=True, allow_infinity=True))


class TestOps(unittest.TestCase):

    def assertEqualOrNan(self, a: Fraction | float, b: Fraction | float, msg: str = ''):
        if math.isnan(a) or math.isnan(b):
            self.assertTrue(math.isnan(a) and math.isnan(b), msg)
        else:
            self.assertEqual(a, b, msg)

    @given(number())
    def test_neg(self, a):
        af = _cvt_to_frac(a)
        op = _cvt_to_frac(fp.neg(a))
        self.assertEqualOrNan(op, -af, f'Failed negation: {a} ({af})')

    @given(number())
    def test_abs(self, a):
        af = _cvt_to_frac(a)
        op = _cvt_to_frac(fp.fabs(a))
        self.assertEqualOrNan(op, abs(af), f'Failed absolute value: {a} ({af})')

    @given(number(), number())
    def test_add(self, a, b):
        af = _cvt_to_frac(a)
        bf = _cvt_to_frac(b)
        op = _cvt_to_frac(fp.add(a, b))
        self.assertEqualOrNan(op, af + bf, f'Failed addition: {a} + {b} ({af} + {bf})')

    @given(number(), number())
    def test_sub(self, a, b):
        af = _cvt_to_frac(a)
        bf = _cvt_to_frac(b)
        op = _cvt_to_frac(fp.sub(a, b))
        self.assertEqualOrNan(op, af - bf, f'Failed subtraction: {a} - {b} ({af} - {bf})')

    @given(number(), number())
    def test_mul(self, a, b):
        af = _cvt_to_frac(a)
        bf = _cvt_to_frac(b)
        op = _cvt_to_frac(fp.mul(a, b))
        self.assertEqualOrNan(op, af * bf, f'Failed multiplication: {a} * {b} ({af} * {bf})')

    @given(number(), number(), number())
    def test_fma(self, a, b, c):
        af = _cvt_to_frac(a)
        bf = _cvt_to_frac(b)
        cf = _cvt_to_frac(c)
        op = _cvt_to_frac(fp.fma(a, b, c))
        self.assertEqualOrNan(op, af * bf + cf, f'Failed FMA: {a} * {b} + {c} ({af} * {bf} + {cf})')

    @given(number())
    def test_ceil(self, a):
        af = _cvt_to_frac(a)
        op = _cvt_to_frac(fp.ceil(a))
        self.assertEqualOrNan(op, _ceil(af), f'Failed ceiling: {a} ({af})')

    @given(number())
    def test_floor(self, a):
        af = _cvt_to_frac(a)
        op = _cvt_to_frac(fp.floor(a))
        self.assertEqualOrNan(op, _floor(af), f'Failed floor: {a} ({af})')

    @given(number())
    def test_trunc(self, a):
        af = _cvt_to_frac(a)
        op = _cvt_to_frac(fp.trunc(a))
        self.assertEqualOrNan(op, _trunc(af), f'Failed truncation: {a} ({af})')


class TestModOp(unittest.TestCase):
    """Tests FPy's modulus operator."""

    def assertEqualOrNan(self, a: Fraction | float, b: Fraction | float, msg: str = ''):
        if math.isnan(a) or math.isnan(b):
            self.assertTrue(math.isnan(a) and math.isnan(b), msg)
        else:
            self.assertEqual(a, b, msg)

    def test_mod_fixed(self):
        CTX = fp.MPFixedContext(nmin=-1, enable_inf=True, enable_nan=True)
        XS = [5.0, -5.0, 3.0, -3.0, 0.0, float('inf'), float('-inf'), float('nan')]

        for x in XS:
            for y in XS:
                try:
                    expect = x % y
                except ZeroDivisionError:
                    expect = float('nan')
                actual = fp.mod(x, y, CTX)
                self.assertEqualOrNan(expect, actual, f'Failed modulus: {x} % {y}')

    @given(st.integers(), st.integers())
    def test_mod(self, x: int, y: int):
        CTX = fp.MPFixedContext(nmin=-1, enable_inf=True, enable_nan=True)

        try:
            expect: int | float = x % y
        except ZeroDivisionError:
            expect = float('nan')

        actual = fp.mod(x, y, CTX)
        self.assertEqualOrNan(_cvt_to_frac(expect), _cvt_to_frac(actual), f'Failed modulus: {x} % {y}')
