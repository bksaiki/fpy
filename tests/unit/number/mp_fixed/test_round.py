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
        st.sampled_from(fp.RM),
        # `num_randbits=0` is the deterministic path
        st.sampled_from([1, 8, 32]),
        st.sampled_from([random.Random, np.random.default_rng]),
        st.integers(min_value=0),
    )
    def test_round_stochastic(self, x: fp.Float, n: int, rm: fp.RM, num_randbits: int, make_rng, seed: int):
        ctx = fp.MPFixedContext(n, rm, num_randbits=num_randbits, rng=make_rng(seed))
        ctx_rtz = fp.MPFixedContext(n, fp.RM.RTZ)
        ctx_raz = fp.MPFixedContext(n, fp.RM.RAZ)
        rounded = ctx.round(x)
        rtz = ctx_rtz.round(x)
        raz = ctx_raz.round(x)
        assert isinstance(rounded, fp.Float)
        assert rounded == rtz or rounded == raz
        assert (rtz == raz) == (not rounded.inexact)
