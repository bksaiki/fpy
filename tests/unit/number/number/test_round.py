import fpy2 as fp
import unittest

from fractions import Fraction
from hypothesis import given, strategies as st

from ...generators import real_floats, floats, rounding_modes


class TestRealFloatRoundMethods(unittest.TestCase):
    """Testing `RealFloat` rounding methods"""

    def test_round_params(self):
        inputs = [
            # float-style rounding
            (0, 0b101010, 3, None, 3, 2), # 0b101010 * 2 ** 0, max_p=3, min_n=None => p=3, n=2
            (0, 0b10, 3, None, 3, -2),    # 0b10 * 2 ** 0, max_p=3, min_n=None => p=3, n=-2
            # fixed-style rounding
            (0, 0b101010, None, 2, None, 2), # 0b101010 * 2 ** 0, max_p=None, min_n=2 => p=None, n=2
            (0, 0b101010, None, 7, None, 7), # 0b101010 * 2 ** 0, max_p=None, min_n=7 => p=None, n=7
            # float-style (with subnormals) rounding
            (0, 0b101010, 3, 2, 3, 2), # 0b101010 * 2 ** 0, max_p=3, min_n=2 => p=3, n=2
            (0, 0b101010, 3, 5, 3, 5), # 0b101010 * 2 ** 0, max_p=3, min_n=5 => p=3, n=5
            (0, 0b10, 3, 3, 3, 3), # 0b10 * 2 ** 0, max_p=3, min_n=3 => p=3, n=3
        ]

        for exp, c, max_p, min_n, expect_p, expect_n in inputs:
            x = fp.RealFloat(exp=exp, c=c)
            p, n = x._round_params(max_p=max_p, min_n=min_n)
            self.assertEqual(p, expect_p, f'x={x}, max_p={max_p}, min_n={min_n}, p={p}, expect_p={expect_p}')
            self.assertEqual(n, expect_n, f'x={x}, max_p={max_p}, min_n={min_n}, n={n}, expect_n={expect_n}')

    def test_round_examples(self):
        inputs = [
            # 8 * 2 ** -3 (representable)
            (-3, 8, -1, 2, fp.RM.RNE), # 8 * 2 ** -3 => 1 * 2 ** -1
            (-3, 8, -1, 2, fp.RM.RNA), # 8 * 2 ** -3 => 1 * 2 ** -1
            (-3, 8, -1, 2, fp.RM.RTP), # 8 * 2 ** -3 => 1 * 2 ** -1
            (-3, 8, -1, 2, fp.RM.RTN), # 8 * 2 ** -3 => 1 * 2 ** -1
            (-3, 8, -1, 2, fp.RM.RTZ), # 8 * 2 ** -3 => 1 * 2 ** -1
            (-3, 8, -1, 2, fp.RM.RAZ), # 8 * 2 ** -3 => 1 * 2 ** -1
            # 9 * 2 ** -3 (below halfway)
            (-3, 9, -1, 2, fp.RM.RNE), # 9 * 2 ** -3 => 1 * 2 ** -1 (down)
            (-3, 9, -1, 2, fp.RM.RNA), # 9 * 2 ** -3 => 1 * 2 ** -1 (down)
            (-3, 9, -1, 3, fp.RM.RTP), # 9 * 2 ** -3 => 1 * 3 ** -1 (up)
            (-3, 9, -1, 2, fp.RM.RTN), # 9 * 2 ** -3 => 1 * 2 ** -1 (down)
            (-3, 9, -1, 2, fp.RM.RTZ), # 9 * 2 ** -3 => 1 * 2 ** -1 (down)
            (-3, 9, -1, 3, fp.RM.RAZ), # 9 * 2 ** -3 => 1 * 3 ** -1 (up)
            # 10 * 2 ** -3 (exactly halfway)
            (-3, 10, -1, 2, fp.RM.RNE), # 10 * 2 ** -3 => 1 * 2 ** -1 (down)
            (-3, 10, -1, 3, fp.RM.RNA), # 10 * 2 ** -3 => 1 * 3 ** -1 (up)
            (-3, 10, -1, 3, fp.RM.RTP), # 10 * 2 ** -3 => 1 * 3 ** -1 (up)
            (-3, 10, -1, 2, fp.RM.RTN), # 10 * 2 ** -3 => 1 * 2 ** -1 (down)
            (-3, 10, -1, 2, fp.RM.RTZ), # 10 * 2 ** -3 => 1 * 2 ** -1 (down)
            (-3, 10, -1, 3, fp.RM.RAZ), # 10 * 2 ** -3 => 1 * 3 ** -1 (up)
            # 11 * 2 ** -3 (above halfway)
            (-3, 11, -1, 3, fp.RM.RNE), # 11 * 2 ** -3 => 1 * 3 ** -1 (up)
            (-3, 11, -1, 3, fp.RM.RNA), # 11 * 2 ** -3 => 1 * 3 ** -1 (up)
            (-3, 11, -1, 3, fp.RM.RTP), # 11 * 2 ** -3 => 1 * 3 ** -1 (up)
            (-3, 11, -1, 2, fp.RM.RTN), # 11 * 2 ** -3 => 1 * 2 ** -1 (down)
            (-3, 11, -1, 2, fp.RM.RTZ), # 11 * 2 ** -3 => 1 * 2 ** -1 (down
            (-3, 11, -1, 3, fp.RM.RAZ), # 11 * 2 ** -3 => 1 * 3 ** -1 (up)
            # 12 * 2 ** -3 (representable)
            (-3, 12, -1, 3, fp.RM.RNE), # 12 * 2 ** -3 => 1 * 3 ** -1
            (-3, 12, -1, 3, fp.RM.RNA), # 12 * 2 ** -3 => 1 * 3 ** -1
            (-3, 12, -1, 3, fp.RM.RTP), # 12 * 2 ** -3 => 1 * 3 ** -1
            (-3, 12, -1, 3, fp.RM.RTN), # 12 * 2 ** -3 => 1 * 3 ** -1
            (-3, 12, -1, 3, fp.RM.RTZ), # 12 * 2 ** -3 => 1 * 3 ** -1
            (-3, 12, -1, 3, fp.RM.RAZ), # 12 * 2 ** -3 => 1 * 3 ** -1
        ]

        for exp, c, exp_rounded, c_rounded, rm in inputs:
            x = fp.RealFloat(exp=exp, c=c)
            x_rounded = x.round(max_p=2, rm=rm)
            expect = fp.RealFloat(exp=exp_rounded, c=c_rounded)
            self.assertEqual(x_rounded, expect, f'x={x}, rm={rm!r}, x_rounded={x_rounded}, expect={expect}')
            self.assertLessEqual(x_rounded.p, 2, f'x={x}, rm={rm!r}, x_rounded.p={x_rounded.p}')

    @given(
        real_floats(prec_max=100),
        st.one_of(st.none(), st.integers(min_value=0)),
        st.one_of(st.none(), st.integers()),
        rounding_modes()
    )
    def test_round(self, x: fp.RealFloat, p: int | None, n: int | None, rm: fp.RoundingMode):
        if p is not None or n is not None:
            rounded = x.round(max_p=p, min_n=n, rm=rm)
            self.assertIsInstance(rounded, fp.RealFloat)
            if p is not None:
                self.assertLessEqual(rounded.p, p, f'x={x}, p={p}, rm={rm!r}, rounded.p={rounded.p}')
            if n is not None:
                self.assertGreater(rounded.exp, n, f'x={x}, n={n}, rm={rm!r}, rounded.n={rounded.n}')

    def test_round_stochastic_examples(self):
        inputs = [
            # 8 * 2 ** -3 (representable)
            (-3, 8, -1, 2, 0), # 8 * 2 ** -3 => 1 * 2 ** -1 (0: down)
            (-3, 8, -1, 2, 1), # 8 * 2 ** -3 => 1 * 2 ** -1 (1: down)
            (-3, 8, -1, 2, 2), # 8 * 2 ** -3 => 1 * 2 ** -1 (2: down)
            (-3, 8, -1, 2, 3), # 8 * 2 ** -3 => 1 * 2 ** -1 (3: down)
            # 9 * 2 ** -3 (below halfway)
            (-3, 9, -1, 3, 0), # 9 * 2 ** -3 => 1 * 2 ** -1 (0: up)
            (-3, 9, -1, 2, 1), # 9 * 2 ** -3 => 1 * 2 ** -1 (1: down)
            (-3, 9, -1, 2, 2), # 9 * 2 ** -3 => 1 * 3 ** -1 (2: down)
            (-3, 9, -1, 2, 3), # 9 * 2 ** -3 => 1 * 2 ** -1 (3: down)
            # 10 * 2 ** -3 (exactly halfway)
            (-3, 10, -1, 3, 0), # 10 * 2 ** -3 => 1 * 2 ** -1 (0: up)
            (-3, 10, -1, 3, 1), # 10 * 2 ** -3 => 1 * 3 ** -1 (1: up)
            (-3, 10, -1, 2, 2), # 10 * 2 ** -3 => 1 * 3 ** -1 (2: down)
            (-3, 10, -1, 2, 3), # 10 * 2 ** -3 => 1 * 2 ** -1 (3: down)
            # 11 * 2 ** -3 (above halfway)
            (-3, 11, -1, 3, 0), # 11 * 2 ** -3 => 1 * 3 ** -1 (0: up)
            (-3, 11, -1, 3, 1), # 11 * 2 ** -3 => 1 * 3 ** -1 (1: up)
            (-3, 11, -1, 3, 2), # 11 * 2 ** -3 => 1 * 3 ** -1 (2: up)
            (-3, 11, -1, 2, 3), # 11 * 2 ** -3 => 1 * 2 ** -1 (3: down)
            # 12 * 2 ** -3 (representable)
            (-3, 12, -1, 3, 0), # 12 * 2 ** -3 => 1 * 3 ** -1 (0: down)
            (-3, 12, -1, 3, 1), # 12 * 2 ** -3 => 1 * 3 ** -1 (1: down)
            (-3, 12, -1, 3, 2), # 12 * 2 ** -3 => 1 * 3 ** -1 (2: down)
            (-3, 12, -1, 3, 3), # 12 * 2 ** -3 => 1 * 3 ** -1 (3: down)
        ]

        for exp, c, exp_rounded, c_rounded, randbits in inputs:
            x = fp.RealFloat(exp=exp, c=c)
            x_rounded = x.round(max_p=2, num_randbits=2, randbits=randbits)
            expect = fp.RealFloat(exp=exp_rounded, c=c_rounded)
            self.assertEqual(x_rounded, expect, f'x={x}, randbits={randbits}, x_rounded={x_rounded}, expect={expect}')
            self.assertLessEqual(x_rounded.p, 2, f'x={x}, randbits={randbits}, x_rounded.p={x_rounded.p}')

    @given(
        real_floats(prec_max=100),
        st.one_of(st.none(), st.integers(min_value=0)),
        st.one_of(st.none(), st.integers()),
        st.one_of(st.none(), st.integers(min_value=1, max_value=100)),
        rounding_modes()
    )
    def test_round_stochastic(
        self,
        x: fp.RealFloat,
        p: int | None,
        n: int | None,
        num_randbits: int,
        rm: fp.RoundingMode
    ):
        if p is not None or n is not None:
            rounded = x.round(max_p=p, min_n=n, rm=rm, num_randbits=num_randbits)
            self.assertIsInstance(rounded, fp.RealFloat)
            if p is not None:
                self.assertLessEqual(rounded.p, p, f'x={x}, p={p}, rm={rm!r}, rounded.p={rounded.p}')
            if n is not None:
                self.assertGreater(rounded.exp, n, f'x={x}, n={n}, rm={rm!r}, rounded.n={rounded.n}')

    def test_round_at_examples(self):
        inputs = [
            # 8 * 2 ** -3 (representable)
            (-3, 8, -1, 2, fp.RM.RNE), # 8 * 2 ** -3 => 1 * 2 ** -1
            (-3, 8, -1, 2, fp.RM.RNA), # 8 * 2 ** -3 => 1 * 2 ** -1
            (-3, 8, -1, 2, fp.RM.RTP), # 8 * 2 ** -3 => 1 * 2 ** -1
            (-3, 8, -1, 2, fp.RM.RTN), # 8 * 2 ** -3 => 1 * 2 ** -1
            (-3, 8, -1, 2, fp.RM.RTZ), # 8 * 2 ** -3 => 1 * 2 ** -1
            (-3, 8, -1, 2, fp.RM.RAZ), # 8 * 2 ** -3 => 1 * 2 ** -1
            # 9 * 2 ** -3 (below halfway)
            (-3, 9, -1, 2, fp.RM.RNE), # 9 * 2 ** -3 => 1 * 2 ** -1 (down)
            (-3, 9, -1, 2, fp.RM.RNA), # 9 * 2 ** -3 => 1 * 2 ** -1 (down)
            (-3, 9, -1, 3, fp.RM.RTP), # 9 * 2 ** -3 => 1 * 3 ** -1 (up)
            (-3, 9, -1, 2, fp.RM.RTN), # 9 * 2 ** -3 => 1 * 2 ** -1 (down)
            (-3, 9, -1, 2, fp.RM.RTZ), # 9 * 2 ** -3 => 1 * 2 ** -1 (down)
            (-3, 9, -1, 3, fp.RM.RAZ), # 9 * 2 ** -3 => 1 * 3 ** -1 (up)
            # 10 * 2 ** -3 (exactly halfway)
            (-3, 10, -1, 2, fp.RM.RNE), # 10 * 2 ** -3 => 1 * 2 ** -1 (down)
            (-3, 10, -1, 3, fp.RM.RNA), # 10 * 2 ** -3 => 1 * 3 ** -1 (up)
            (-3, 10, -1, 3, fp.RM.RTP), # 10 * 2 ** -3 => 1 * 3 ** -1 (up)
            (-3, 10, -1, 2, fp.RM.RTN), # 10 * 2 ** -3 => 1 * 2 ** -1 (down)
            (-3, 10, -1, 2, fp.RM.RTZ), # 10 * 2 ** -3 => 1 * 2 ** -1 (down)
            (-3, 10, -1, 3, fp.RM.RAZ), # 10 * 2 ** -3 => 1 * 3 ** -1 (up)
            # 11 * 2 ** -3 (above halfway)
            (-3, 11, -1, 3, fp.RM.RNE), # 11 * 2 ** -3 => 1 * 3 ** -1 (up)
            (-3, 11, -1, 3, fp.RM.RNA), # 11 * 2 ** -3 => 1 * 3 ** -1 (up)
            (-3, 11, -1, 3, fp.RM.RTP), # 11 * 2 ** -3 => 1 * 3 ** -1 (up)
            (-3, 11, -1, 2, fp.RM.RTN), # 11 * 2 ** -3 => 1 * 2 ** -1 (down)
            (-3, 11, -1, 2, fp.RM.RTZ), # 11 * 2 ** -3 => 1 * 2 ** -1 (down
            (-3, 11, -1, 3, fp.RM.RAZ), # 11 * 2 ** -3 => 1 * 3 ** -1 (up)
            # 12 * 2 ** -3 (representable)
            (-3, 12, -1, 3, fp.RM.RNE), # 12 * 2 ** -3 => 1 * 3 ** -1
            (-3, 12, -1, 3, fp.RM.RNA), # 12 * 2 ** -3 => 1 * 3 ** -1
            (-3, 12, -1, 3, fp.RM.RTP), # 12 * 2 ** -3 => 1 * 3 ** -1
            (-3, 12, -1, 3, fp.RM.RTN), # 12 * 2 ** -3 => 1 * 3 ** -1
            (-3, 12, -1, 3, fp.RM.RTZ), # 12 * 2 ** -3 => 1 * 3 ** -1
            (-3, 12, -1, 3, fp.RM.RAZ), # 12 * 2 ** -3 => 1 * 3 ** -1
        ]

        for exp, c, exp_rounded, c_rounded, rm in inputs:
            x = fp.RealFloat(exp=exp, c=c)
            x_rounded = x.round_at(n=-2, rm=rm)
            expect = fp.RealFloat(exp=exp_rounded, c=c_rounded)
            self.assertEqual(x_rounded, expect, f'x={x}, rm={rm!r}, x_rounded={x_rounded}, expect={expect}')
            self.assertGreater(x_rounded.exp, -2, f'x={x}, rm={rm!r}, x_rounded.exp={x_rounded.exp}')

    @given(
        real_floats(prec_max=100),
        st.integers(),
        st.one_of(st.none(), st.integers()),
        rounding_modes()
    )
    def test_round_at(self, x: fp.RealFloat, n: int, p: int | None, rm: fp.RoundingMode):
        rounded = x.round_at(n=n, rm=rm)
        self.assertIsInstance(rounded, fp.RealFloat)

    def test_round_at_stochastic_examples(self):
        inputs = [
            # 8 * 2 ** -3 (representable)
            (-3, 8, -1, 2, 0), # 8 * 2 ** -3 => 1 * 2 ** -1 (0: down)
            (-3, 8, -1, 2, 1), # 8 * 2 ** -3 => 1 * 2 ** -1 (1: down)
            (-3, 8, -1, 2, 2), # 8 * 2 ** -3 => 1 * 2 ** -1 (2: down)
            (-3, 8, -1, 2, 3), # 8 * 2 ** -3 => 1 * 2 ** -1 (3: down)
            # 9 * 2 ** -3 (below halfway)
            (-3, 9, -1, 3, 0), # 9 * 2 ** -3 => 1 * 2 ** -1 (0: up)
            (-3, 9, -1, 2, 1), # 9 * 2 ** -3 => 1 * 2 ** -1 (1: down)
            (-3, 9, -1, 2, 2), # 9 * 2 ** -3 => 1 * 3 ** -1 (2: down)
            (-3, 9, -1, 2, 3), # 9 * 2 ** -3 => 1 * 2 ** -1 (3: down)
            # 10 * 2 ** -3 (exactly halfway)
            (-3, 10, -1, 3, 0), # 10 * 2 ** -3 => 1 * 2 ** -1 (0: up)
            (-3, 10, -1, 3, 1), # 10 * 2 ** -3 => 1 * 3 ** -1 (1: up)
            (-3, 10, -1, 2, 2), # 10 * 2 ** -3 => 1 * 3 ** -1 (2: down)
            (-3, 10, -1, 2, 3), # 10 * 2 ** -3 => 1 * 2 ** -1 (3: down)
            # 11 * 2 ** -3 (above halfway)
            (-3, 11, -1, 3, 0), # 11 * 2 ** -3 => 1 * 3 ** -1 (0: up)
            (-3, 11, -1, 3, 1), # 11 * 2 ** -3 => 1 * 3 ** -1 (1: up)
            (-3, 11, -1, 3, 2), # 11 * 2 ** -3 => 1 * 3 ** -1 (2: up)
            (-3, 11, -1, 2, 3), # 11 * 2 ** -3 => 1 * 2 ** -1 (3: down)
            # 12 * 2 ** -3 (representable)
            (-3, 12, -1, 3, 0), # 12 * 2 ** -3 => 1 * 3 ** -1 (0: down)
            (-3, 12, -1, 3, 1), # 12 * 2 ** -3 => 1 * 3 ** -1 (1: down)
            (-3, 12, -1, 3, 2), # 12 * 2 ** -3 => 1 * 3 ** -1 (2: down)
            (-3, 12, -1, 3, 3), # 12 * 2 ** -3 => 1 * 3 ** -1 (3: down)
        ]

        for exp, c, exp_rounded, c_rounded, randbits in inputs:
            x = fp.RealFloat(exp=exp, c=c)
            x_rounded = x.round_at(n=-2, num_randbits=2, randbits=randbits)
            expect = fp.RealFloat(exp=exp_rounded, c=c_rounded)
            self.assertEqual(x_rounded, expect, f'x={x}, randbits={randbits}, x_rounded={x_rounded}, expect={expect}')
            self.assertGreater(x_rounded.exp, -2, f'x={x}, randbits={randbits}, x_rounded.exp={x_rounded.exp}')

    @given(
        real_floats(prec_max=100),
        st.integers(),
        st.one_of(st.none(), st.integers(min_value=1, max_value=100)),
        rounding_modes()
    )
    def test_round_at_stochastic(
        self,
        x: fp.RealFloat,
        n: int,
        num_randbits: int | None,
        rm: fp.RoundingMode
    ):
        rounded = x.round_at(n=n, rm=rm, num_randbits=num_randbits)
        self.assertIsInstance(rounded, fp.RealFloat)
        self.assertGreater(rounded.exp, n, f'x={x}, randbits={num_randbits}, rounded.exp={rounded.exp}')


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
        fp.ExpContext(8),
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

    @given(real_floats(prec_max=_MAX_PRECISION, exp_min=-_EXP_LIMIT, exp_max=_EXP_LIMIT))
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

    @given(real_floats(prec_max=_MAX_PRECISION, exp_min=-_EXP_LIMIT, exp_max=_EXP_LIMIT), st.integers())
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
