import random

from fpy2 import MPBFixedContext, RealFloat, RM, OV

class RoundTestCase():
    """Testing `MPBFixedContext.round()`"""

    def test_fuzz(self, num_values: int = 10_000, expmin: int = -24, expmax: int = 24, pmax: int = 12):
        limit = RealFloat.from_int(2 ** 12)
        seed = 1

        # saturation
        sat_ctx = MPBFixedContext(-1, limit, RM.RTZ, OV.SATURATE)
        random.seed(seed)
        for _ in range(num_values):
            s = random.choice([False, True])
            exp = random.randint(expmin, expmax)
            c = random.randint(0, 2**pmax - 1)
            x = RealFloat(s=s, exp=exp, c=c)
            sat_ctx.round(x)

        # wrap
        wrap_ctx = MPBFixedContext(-1, limit, RM.RTZ, OV.WRAP)
        random.seed(seed)
        for _ in range(num_values):
            s = random.choice([False, True])
            exp = random.randint(expmin, expmax)
            c = random.randint(0, 2**pmax - 1)
            x = RealFloat(s=s, exp=exp, c=c)
            wrap_ctx.round(x)


    def test_overflow(self):
        limit = RealFloat.from_int(2 ** 8)
        x = RealFloat.from_int(2 ** 8 + 1)
        neg_x = RealFloat.from_int(-(2 ** 8 + 1))

        # saturation
        sat_ctx = MPBFixedContext(-1, limit, RM.RTZ, OV.SATURATE)
        assert sat_ctx.round(x) == limit
        assert sat_ctx.round(neg_x) == -limit

        # wrap
        wrap_ctx = MPBFixedContext(-1, limit, RM.RTZ, OV.WRAP)
        assert wrap_ctx.round(x) == -limit
        assert wrap_ctx.round(neg_x) == limit

