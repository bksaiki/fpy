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
