"""
Testing `Float` and `REalFloat` comparison methods.
"""

import fpy2 as fp
import unittest

from fractions import Fraction
from hypothesis import given, strategies as st

from ..generators import real_floats, floats

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
    - `float`,
    - `Fraction`,
    - `Float`,
    - or `RealFloat`.
    """
    choice = draw(st.integers(min_value=0, max_value=4))
    if choice == 0:
        return draw(st.integers())
    elif choice == 1:
        return draw(st.floats(allow_nan=True, allow_infinity=True, allow_subnormal=True))
    elif choice == 2:
        return draw(st.fractions())
    elif choice == 3:
        return draw(floats(prec=prec, exp_min=exp_min, exp_max=exp_max, allow_nan=True, allow_infinity=True))
    else:
        return draw(real_floats(signed=True, prec=prec, exp_min=exp_min, exp_max=exp_max))


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


class TestCompareMethods(unittest.TestCase):

    @given(number(), number())
    def test_compare_eq(self, a, b):
        af = _cvt_to_frac(a)
        bf = _cvt_to_frac(b)
        self.assertEqual(a == b, af == bf, f'Failed comparison: {a} == {b} ({af} == {bf})')

    @given(number(), number())
    def test_compare_ne(self, a, b):
        af = _cvt_to_frac(a)
        bf = _cvt_to_frac(b)
        self.assertEqual(a != b, af != bf, f'Failed comparison: {a} != {b} ({af} != {bf})')

    @given(number(), number())
    def test_compare_lt(self, a, b):
        af = _cvt_to_frac(a)
        bf = _cvt_to_frac(b)
        self.assertEqual(a < b, af < bf, f'Failed comparison: {a} < {b} ({af} < {bf})')

    @given(number(), number())
    def test_compare_le(self, a, b):
        af = _cvt_to_frac(a)
        bf = _cvt_to_frac(b)
        self.assertEqual(a <= b, af <= bf, f'Failed comparison: {a} <= {b} ({af} <= {bf})')

    @given(number(), number())
    def test_compare_gt(self, a, b):
        af = _cvt_to_frac(a)
        bf = _cvt_to_frac(b)
        self.assertEqual(a > b, af > bf, f'Failed comparison: {a} > {b} ({af} > {bf})')

    @given(number(), number())
    def test_compare_ge(self, a, b):
        af = _cvt_to_frac(a)
        bf = _cvt_to_frac(b)
        self.assertEqual(a >= b, af >= bf, f'Failed comparison: {a} >= {b}  ({af} >= {bf})')
