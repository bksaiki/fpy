import fpy2 as fp
import unittest

from hypothesis import assume, given, strategies as st
from ...generators import real_floats

class ReferenceFlags:
    """Reference implementation of status flags for `RealFloat.round()`."""

    @staticmethod
    def inexact(x: fp.RealFloat, p: int | None, n: int | None, rm: fp.RM) -> bool:
        rounded = x.round(max_p=p, min_n=n, rm=rm)
        return rounded != x

    @staticmethod
    def carry(x: fp.RealFloat, p: int | None, n: int | None, rm: fp.RM) -> bool:
        rounded = x.round(max_p=p, min_n=n, rm=rm)
        if p is None:
            emin = float('inf')
        elif n is None:
            emin = float('-inf')
        else:
            emin = p + n
        return x != 0 and rounded != 0 and rounded.e > x.e and x.e >= emin

    @staticmethod
    def tiny_pre(x: fp.RealFloat, p: int | None, n: int | None, rm: fp.RM) -> bool:
        if p is None or n is None:
            return False
        emin = p + n
        return abs(x) < fp.RealFloat.power_of_2(emin)

    @staticmethod
    def tiny_post(x: fp.RealFloat, p: int | None, n: int | None, rm: fp.RM) -> bool:
        if p is None or n is None:
            return False
        emin = p + n
        y = x.round(max_p=p, rm=rm)
        return abs(y) < fp.RealFloat.power_of_2(emin)

    @staticmethod
    def underflow_pre(x: fp.RealFloat, p: int | None, n: int | None, rm: fp.RM) -> bool:
        tiny_pre = ReferenceFlags.tiny_pre(x, p, n, rm)
        inexact = ReferenceFlags.inexact(x, p, n, rm)
        return tiny_pre and inexact

    @staticmethod
    def underflow_post(x: fp.RealFloat, p: int | None, n: int | None, rm: fp.RM) -> bool:
        tiny_post = ReferenceFlags.tiny_post(x, p, n, rm)
        inexact = ReferenceFlags.inexact(x, p, n, rm)
        return tiny_post and inexact


class TestRoundFlags(unittest.TestCase):
    """Testing status flags set by `RealFloat.round()`."""

    @given(
        real_floats(prec_max=16, exp_min=-16, exp_max=16),
        st.one_of(st.none(), st.integers(min_value=1, max_value=8)),
        st.one_of(st.none(), st.integers(min_value=-16, max_value=16)),
        st.sampled_from(fp.RM)
    )
    def test_inexact(self, x: fp.RealFloat, p: int | None, n: int | None, rm: fp.RM):
        assume(p is not None or n is not None)
        rounded = x.round(max_p=p, min_n=n, rm=rm)
        inexact = ReferenceFlags.inexact(x, p, n, rm)
        self.assertEqual(rounded.inexact, inexact, f'x={x}, p={p}, n={n}, rm={rm!r}, rounded={rounded!r}, inexact={inexact}')

    @given(
        real_floats(prec_max=16, exp_min=-16, exp_max=16),
        st.one_of(st.none(), st.integers(min_value=1, max_value=8)),
        st.one_of(st.none(), st.integers(min_value=-16, max_value=16)),
        st.sampled_from(fp.RM)
    )
    def test_carry(self, x: fp.RealFloat, p: int | None, n: int | None, rm: fp.RM):
        assume(p is not None or n is not None)
        rounded = x.round(max_p=p, min_n=n, rm=rm)
        carry = ReferenceFlags.carry(x, p, n, rm)
        self.assertEqual(rounded.carry, carry, f'x={x}, p={p}, n={n}, rm={rm!r}, rounded={rounded!r}, carry={carry}')

    @given(
        real_floats(prec_max=16, exp_min=-16, exp_max=16),
        st.one_of(st.none(), st.integers(min_value=1, max_value=8)),
        st.one_of(st.none(), st.integers(min_value=-16, max_value=16)),
        st.sampled_from(fp.RM)
    )
    def test_tiny_pre(self, x: fp.RealFloat, p: int | None, n: int | None, rm: fp.RM):
        assume(p is not None or n is not None)
        rounded = x.round(max_p=p, min_n=n, rm=rm)
        tiny_pre = ReferenceFlags.tiny_pre(x, p, n, rm)
        self.assertEqual(rounded.tiny_pre, tiny_pre, f'x={x}, p={p}, n={n}, rm={rm!r}, rounded={rounded!r}, tiny_pre={tiny_pre}')

    @given(
        real_floats(prec_max=16, exp_min=-16, exp_max=16),
        st.one_of(st.none(), st.integers(min_value=1, max_value=8)),
        st.one_of(st.none(), st.integers(min_value=-16, max_value=16)),
        st.sampled_from(fp.RM)
    )
    def test_tiny_post(self, x: fp.RealFloat, p: int | None, n: int | None, rm: fp.RM):
        assume(p is not None or n is not None)
        rounded = x.round(max_p=p, min_n=n, rm=rm)
        tiny_post = ReferenceFlags.tiny_post(x, p, n, rm)
        self.assertEqual(rounded.tiny_post, tiny_post, f'x={x}, p={p}, n={n}, rm={rm!r}, rounded={rounded!r}, tiny_post={tiny_post}')

    @given(
        real_floats(prec_max=16, exp_min=-16, exp_max=16),
        st.one_of(st.none(), st.integers(min_value=1, max_value=8)),
        st.one_of(st.none(), st.integers(min_value=-16, max_value=16)),
        st.sampled_from(fp.RM)
    )
    def test_underflow_pre(self, x: fp.RealFloat, p: int | None, n: int | None, rm: fp.RM):
        assume(p is not None or n is not None)
        rounded = x.round(max_p=p, min_n=n, rm=rm)
        underflow_pre = ReferenceFlags.underflow_pre(x, p, n, rm)
        self.assertEqual(rounded.underflow_pre, underflow_pre, f'x={x}, p={p}, n={n}, rm={rm!r}, rounded={rounded!r}, underflow_pre={underflow_pre}')

    @given(
        real_floats(prec_max=16, exp_min=-16, exp_max=16),
        st.one_of(st.none(), st.integers(min_value=1, max_value=8)),
        st.one_of(st.none(), st.integers(min_value=-16, max_value=16)),
        st.sampled_from(fp.RM)
    )
    def test_underflow_post(self, x: fp.RealFloat, p: int | None, n: int | None, rm: fp.RM):
        assume(p is not None or n is not None)
        rounded = x.round(max_p=p, min_n=n, rm=rm)
        underflow_post = ReferenceFlags.underflow_post(x, p, n, rm)
        self.assertEqual(rounded.underflow_post, underflow_post, f'x={x}, p={p}, n={n}, rm={rm!r}, rounded={rounded!r}, underflow_post={underflow_post}')


class TestArithmeticFlags(unittest.TestCase):
    """Testing `invalid` and `divzero` flags set by mathematical operations."""

    _CTX = fp.FP64

    # ------------------------------------------------------------------
    # add

    def test_add(self):
        # no invalid: NaN result propagated from a NaN input
        nan = fp.Float.nan()
        r = fp.add(nan, fp.Float.from_int(1), self._CTX)
        self.assertTrue(r.isnan, f'expected NaN, got {r!r}')
        self.assertFalse(r.invalid, f'unexpected invalid flag when NaN propagated from input')
        # no divzero: Inf result propagated from an Inf input
        inf = fp.Float.inf(s=False)
        r = fp.add(inf, fp.Float.from_int(1), self._CTX)
        self.assertTrue(r.isinf, f'expected Inf, got {r!r}')
        self.assertFalse(r.divzero, f'unexpected divzero flag when Inf propagated from input')

    # ------------------------------------------------------------------
    # sub

    def test_sub(self):
        # invalid: inf - inf
        inf = fp.Float.inf(s=False)
        r = fp.sub(inf, inf, self._CTX)
        self.assertTrue(r.isnan, f'expected NaN, got {r!r}')
        self.assertTrue(r.invalid, f'expected invalid flag for inf - inf')

    # ------------------------------------------------------------------
    # mul

    def test_mul(self):
        # invalid: 0 * inf
        zero = fp.Float.from_int(0)
        inf = fp.Float.inf(s=False)
        r = fp.mul(zero, inf, self._CTX)
        self.assertTrue(r.isnan, f'expected NaN, got {r!r}')
        self.assertTrue(r.invalid, f'expected invalid flag for 0 * inf')

    # ------------------------------------------------------------------
    # div

    def test_div(self):
        # invalid: 0/0 and ∞/∞
        zero = fp.Float.from_int(0)
        r = fp.div(zero, zero, self._CTX)
        self.assertTrue(r.isnan, f'expected NaN, got {r!r}')
        self.assertTrue(r.invalid, f'expected invalid flag for 0/0')
        inf = fp.Float.inf(s=False)
        r = fp.div(inf, inf, self._CTX)
        self.assertTrue(r.isnan, f'expected NaN, got {r!r}')
        self.assertTrue(r.invalid, f'expected invalid flag for inf/inf')
        # divzero: non-zero finite / 0
        one = fp.Float.from_int(1)
        r = fp.div(one, zero, self._CTX)
        self.assertTrue(r.isinf, f'expected Inf, got {r!r}')
        self.assertTrue(r.divzero, f'expected divzero flag for 1/0')

    # ------------------------------------------------------------------
    # sqrt

    def test_sqrt(self):
        # invalid: sqrt of a negative finite number
        x = fp.Float.from_int(-1)
        r = fp.sqrt(x, self._CTX)
        self.assertTrue(r.isnan, f'expected NaN, got {r!r}')
        self.assertTrue(r.invalid, f'expected invalid flag for sqrt(-1)')

    # ------------------------------------------------------------------
    # fma

    def test_fma(self):
        # invalid: fma(0, ∞, finite)
        zero = fp.Float.from_int(0)
        inf = fp.Float.inf(s=False)
        one = fp.Float.from_int(1)
        r = fp.fma(zero, inf, one, self._CTX)
        self.assertTrue(r.isnan, f'expected NaN, got {r!r}')
        self.assertTrue(r.invalid, f'expected invalid flag for fma(0, inf, 1)')

    # ------------------------------------------------------------------
    # remainder

    def test_remainder(self):
        # invalid: remainder(finite, 0) and remainder(∞, finite)
        x = fp.Float.from_int(3)
        zero = fp.Float.from_int(0)
        r = fp.remainder(x, zero, self._CTX)
        self.assertTrue(r.isnan, f'expected NaN, got {r!r}')
        self.assertTrue(r.invalid, f'expected invalid flag for remainder(3, 0)')
        inf = fp.Float.inf(s=False)
        one = fp.Float.from_int(1)
        r = fp.remainder(inf, one, self._CTX)
        self.assertTrue(r.isnan, f'expected NaN, got {r!r}')
        self.assertTrue(r.invalid, f'expected invalid flag for remainder(inf, 1)')

    # ------------------------------------------------------------------
    # log / log2 / log10 / log1p

    def test_log(self):
        # divzero: log(0)
        zero = fp.Float.from_int(0)
        r = fp.log(zero, self._CTX)
        self.assertTrue(r.isinf, f'expected Inf, got {r!r}')
        self.assertTrue(r.divzero, f'expected divzero flag for log(0)')
        # invalid: log(x) for x < 0
        x = fp.Float.from_int(-1)
        r = fp.log(x, self._CTX)
        self.assertTrue(r.isnan, f'expected NaN, got {r!r}')
        self.assertTrue(r.invalid, f'expected invalid flag for log(-1)')

    def test_log2(self):
        # divzero: log2(0)
        zero = fp.Float.from_int(0)
        r = fp.log2(zero, self._CTX)
        self.assertTrue(r.isinf, f'expected Inf, got {r!r}')
        self.assertTrue(r.divzero, f'expected divzero flag for log2(0)')

    def test_log10(self):
        # divzero: log10(0)
        zero = fp.Float.from_int(0)
        r = fp.log10(zero, self._CTX)
        self.assertTrue(r.isinf, f'expected Inf, got {r!r}')
        self.assertTrue(r.divzero, f'expected divzero flag for log10(0)')

    def test_log1p(self):
        # divzero: log1p(-1) = log(0)
        x = fp.Float.from_int(-1)
        r = fp.log1p(x, self._CTX)
        self.assertTrue(r.isinf, f'expected Inf, got {r!r}')
        self.assertTrue(r.divzero, f'expected divzero flag for log1p(-1)')

    # ------------------------------------------------------------------
    # sin / cos / tan

    def test_sin(self):
        # invalid: sin(∞)
        inf = fp.Float.inf(s=False)
        r = fp.sin(inf, self._CTX)
        self.assertTrue(r.isnan, f'expected NaN, got {r!r}')
        self.assertTrue(r.invalid, f'expected invalid flag for sin(inf)')

    def test_cos(self):
        # invalid: cos(∞)
        inf = fp.Float.inf(s=False)
        r = fp.cos(inf, self._CTX)
        self.assertTrue(r.isnan, f'expected NaN, got {r!r}')
        self.assertTrue(r.invalid, f'expected invalid flag for cos(inf)')

    def test_tan(self):
        # invalid: tan(∞)
        inf = fp.Float.inf(s=False)
        r = fp.tan(inf, self._CTX)
        self.assertTrue(r.isnan, f'expected NaN, got {r!r}')
        self.assertTrue(r.invalid, f'expected invalid flag for tan(inf)')

    # ------------------------------------------------------------------
    # asin / acos

    def test_asin(self):
        # invalid: asin(x) for |x| > 1
        x = fp.Float.from_int(2)
        r = fp.asin(x, self._CTX)
        self.assertTrue(r.isnan, f'expected NaN, got {r!r}')
        self.assertTrue(r.invalid, f'expected invalid flag for asin(2)')

    def test_acos(self):
        # invalid: acos(x) for |x| > 1
        x = fp.Float.from_int(-2)
        r = fp.acos(x, self._CTX)
        self.assertTrue(r.isnan, f'expected NaN, got {r!r}')
        self.assertTrue(r.invalid, f'expected invalid flag for acos(-2)')

    # ------------------------------------------------------------------
    # atanh / acosh

    def test_atanh(self):
        # divzero: atanh(1)
        one = fp.Float.from_int(1)
        r = fp.atanh(one, self._CTX)
        self.assertTrue(r.isinf, f'expected Inf, got {r!r}')
        self.assertTrue(r.divzero, f'expected divzero flag for atanh(1)')
        # invalid: atanh(x) for |x| > 1
        x = fp.Float.from_int(2)
        r = fp.atanh(x, self._CTX)
        self.assertTrue(r.isnan, f'expected NaN, got {r!r}')
        self.assertTrue(r.invalid, f'expected invalid flag for atanh(2)')

    def test_acosh(self):
        # invalid: acosh(x) for x < 1
        x = fp.Float.from_int(0)
        r = fp.acosh(x, self._CTX)
        self.assertTrue(r.isnan, f'expected NaN, got {r!r}')
        self.assertTrue(r.invalid, f'expected invalid flag for acosh(0)')

    # ------------------------------------------------------------------
    # pow

    def test_pow(self):
        # invalid: pow(x<0, non-integer y)
        x = fp.Float.from_int(-2)
        y = fp.Float.from_float(0.5)
        r = fp.pow(x, y, self._CTX)
        self.assertTrue(r.isnan, f'expected NaN, got {r!r}')
        self.assertTrue(r.invalid, f'expected invalid flag for pow(-2, 0.5)')
