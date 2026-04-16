import fpy2 as fp
import numpy as np
import random

from hypothesis import given, strategies as st

from ...generators import floats


class RoundTestCase():
    """Testing rounding methods of `MPFixedContext`."""

    @given(
        floats(prec_max=16, exp_min=-32, exp_max=32, allow_nan=False, allow_infinity=False),
        st.integers(min_value=-32, max_value=32),
        st.sampled_from(fp.RM)
    )
    def test_round_stochastic(self, x: fp.Float, n: int, rm: fp.RM):
        ctx = fp.MPFixedContext(n, rm, num_randbits=0, rng=random.Random())
        ctx_rtz = fp.MPFixedContext(n, fp.RM.RTZ)
        ctx_raz = fp.MPFixedContext(n, fp.RM.RAZ)
        rounded = ctx.round(x)
        rtz = ctx_rtz.round(x)
        raz = ctx_raz.round(x)
        assert isinstance(rounded, fp.Float)
        assert rounded == rtz or rounded == raz
        assert rtz == raz == not rounded.inexact

    @given(
        floats(prec_max=16, exp_min=-32, exp_max=32, allow_nan=False, allow_infinity=False),
        st.integers(min_value=-32, max_value=32),
        st.sampled_from(fp.RM)
    )
    def test_round_stochastic_numpy(self, x: fp.Float, n: int, rm: fp.RM):
        rng = np.random.default_rng()
        ctx = fp.MPFixedContext(n, rm, num_randbits=0, rng=rng)
        ctx_rtz = fp.MPFixedContext(n, fp.RM.RTZ)
        ctx_raz = fp.MPFixedContext(n, fp.RM.RAZ)
        rounded = ctx.round(x)
        rtz = ctx_rtz.round(x)
        raz = ctx_raz.round(x)
        assert isinstance(rounded, fp.Float)
        assert rounded == rtz or rounded == raz
        assert rtz == raz == not rounded.inexact
