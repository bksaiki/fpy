import fpy2 as fp
import unittest

from fractions import Fraction
from hypothesis import given, strategies as st

from ..generators import real_floats, floats

_MAX_PRECISION = 100
_EXP_LIMIT = 10_000

def _all_contexts():
    common = [
        o for o in vars(fp.number).values()
        if isinstance(o, fp.Context)
    ]

    inst = [
        fp.MPFloatContext(24, fp.RM.RNE),
        fp.MPSFloatContext(24, -127, fp.RM.RNE),
        fp.MPBFloatContext(24, -127, fp.RealFloat.from_int(2 ** 100).normalize(24), fp.RM.RNE),
        fp.MPFixedContext(-8, fp.RM.RNE),
        fp.MPBFixedContext(-8, fp.RealFloat.from_int(2 ** 30), fp.RM.RNE, fp.OV.SATURATE),
        fp.RealContext()
    ]

    return common + inst


class ContextRoundTestCase(unittest.TestCase):
    """Testing `Context.round()`"""

    @given(st.integers())
    def test_round_int(self, x: int):
        for ctx in _all_contexts():
            y = ctx.round(x)
            self.assertIsInstance(y, fp.Float)
            self.assertIs(y.ctx, ctx)

    @given(st.fractions())
    def test_round_fraction(self, x: Fraction):
        for ctx in _all_contexts():
            if not isinstance(ctx, fp.RealContext):
                y = ctx.round(x)
                self.assertIsInstance(y, fp.Float)
                self.assertIs(y.ctx, ctx)

    @given(st.floats(allow_nan=True, allow_infinity=True, allow_subnormal=True))
    def test_round_python_float(self, x: float):
        for ctx in _all_contexts():
            if not isinstance(ctx, fp.MPFixedContext | fp.MPBFixedContext | fp.FixedContext):
                y = ctx.round(x)
                self.assertIsInstance(y, fp.Float)
                self.assertIs(y.ctx, ctx)

    @given(real_floats(_MAX_PRECISION, -_EXP_LIMIT, _EXP_LIMIT))
    def test_round_real_float(self, x: fp.RealFloat):
        for ctx in _all_contexts():
            y = ctx.round(x)
            self.assertIsInstance(y, fp.Float)
            self.assertIs(y.ctx, ctx)

    @given(floats(53, allow_nan=False, allow_infinity=False, ctx=fp.FP64))
    def test_round_float(self, x: fp.Float):
        for ctx in _all_contexts():
            y = ctx.round(x)
            self.assertIsInstance(y, fp.Float)
            self.assertIs(y.ctx, ctx)

class ContextRoundAtTestCase(unittest.TestCase):
    """Testing `Context.round_at()`"""

    @given(st.integers(), st.integers())
    def test_round_int(self, x: int, n: int):
        for ctx in _all_contexts():
            if not isinstance(ctx, fp.RealContext):
                y = ctx.round_at(x, n)
                self.assertIsInstance(y, fp.Float)
                self.assertIs(y.ctx, ctx)

    @given(st.fractions(), st.integers())
    def test_round_fraction(self, x: Fraction,n: int):
        for ctx in _all_contexts():
            if not isinstance(ctx, fp.RealContext):
                y = ctx.round_at(x, n)
                self.assertIsInstance(y, fp.Float)
                self.assertIs(y.ctx, ctx)

    @given(st.floats(allow_nan=True, allow_infinity=True, allow_subnormal=True), st.integers())
    def test_round_python_float(self, x: float, n: int):
        for ctx in _all_contexts():
            if not isinstance(ctx, fp.MPFixedContext | fp.MPBFixedContext | fp.FixedContext | fp.RealContext):
                y = ctx.round_at(x, n)
                self.assertIsInstance(y, fp.Float)
                self.assertIs(y.ctx, ctx)

    @given(real_floats(_MAX_PRECISION, -_EXP_LIMIT, _EXP_LIMIT), st.integers())
    def test_round_real_float(self, x: fp.RealFloat, n: int):
        for ctx in _all_contexts():
            if not isinstance(ctx, fp.RealContext):
                y = ctx.round_at(x, n)
                self.assertIsInstance(y, fp.Float)
                self.assertIs(y.ctx, ctx)

    @given(floats(53, allow_nan=False, allow_infinity=False, ctx=fp.FP64), st.integers())
    def test_round_float(self, x: fp.Float, n: int):
        for ctx in _all_contexts():
            if not isinstance(ctx, fp.RealContext):
                y = ctx.round_at(x, n)
                self.assertIsInstance(y, fp.Float)
                self.assertIs(y.ctx, ctx)
