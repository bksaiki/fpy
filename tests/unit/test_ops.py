"""
Tests for `fpy2/ops.py`.
"""

import fpy2 as fp
import unittest

from hypothesis import given, strategies as st

from .generators import floats, common_contexts


class TestRoundedOps(unittest.TestCase):

    def assertEqualOrNan(self, a: fp.Float, b: fp.Float, msg = None):
        if a.isnan or b.isnan:
            self.assertTrue(a.isnan and b.isnan, msg)
        else:
            self.assertEqual(a, b, msg)

    @given(
        common_contexts(),
        floats(prec_max=32, exp_min=-100, exp_max=100, allow_infinity=False, allow_nan=False),
        floats(prec_max=32, exp_min=-100, exp_max=100, allow_infinity=False, allow_nan=False),
        st.integers(min_value=0, max_value=16),
        st.integers(min_value=0, max_value=16)
    )
    def test_add(self, ctx: fp.Context, x: fp.Float, y: fp.Float, shiftx: int, shifty: int) -> None:
        x2 = x.normalize(x.p + shiftx)
        y2 = y.normalize(y.p + shifty)
        r1 = fp.add(x, y, ctx)
        r2 = fp.add(x2, y2, ctx)
        self.assertEqualOrNan(r1, r2, f'{x} + {y}: {r1} != {r2} with shifts {shiftx}, {shifty}')

    @given(
        common_contexts(),
        floats(prec_max=32, exp_min=-100, exp_max=100, allow_infinity=False, allow_nan=False),
        floats(prec_max=32, exp_min=-100, exp_max=100, allow_infinity=False, allow_nan=False),
        st.integers(min_value=0, max_value=16),
        st.integers(min_value=0, max_value=16)
    )
    def test_mul(self, ctx: fp.Context, x: fp.Float, y: fp.Float, shiftx: int, shifty: int) -> None:
        x2 = x.normalize(x.p + shiftx)
        y2 = y.normalize(y.p + shifty)
        r1 = fp.mul(x, y, ctx)
        r2 = fp.mul(x2, y2, ctx)
        self.assertEqualOrNan(r1, r2, f'{x} * {y}: {r1} != {r2} with shifts {shiftx}, {shifty}')

    @given(
        common_contexts().filter(lambda ctx: ctx is not fp.REAL),
        floats(prec_max=32, exp_min=-100, exp_max=100, allow_infinity=False, allow_nan=False),
        st.integers(min_value=0, max_value=16),
        st.integers(min_value=0, max_value=16)
    )
    def test_sin(self, ctx: fp.Context, x: fp.Float, shiftx: int, shifty: int) -> None:
        x2 = x.normalize(x.p + shiftx)
        r1 = fp.sin(x, ctx)
        r2 = fp.sin(x2, ctx)
        self.assertEqualOrNan(r1, r2, f'sin({x}): {r1} != {r2} with shifts {shiftx}, {shifty}')

    @given(
        common_contexts().filter(lambda ctx: ctx is not fp.REAL),
        floats(prec_max=32, exp_min=-100, exp_max=100, allow_infinity=False, allow_nan=False),
        st.integers(min_value=0, max_value=16),
        st.integers(min_value=0, max_value=16)
    )
    def test_cos(self, ctx: fp.Context, x: fp.Float, shiftx: int, shifty: int) -> None:
        x2 = x.normalize(x.p + shiftx)
        r1 = fp.cos(x, ctx)
        r2 = fp.cos(x2, ctx)
        self.assertEqualOrNan(r1, r2, f'cos({x}): {r1} != {r2} with shifts {shiftx}, {shifty}')

