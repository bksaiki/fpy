import random

from fpy2 import Float, MPFixedContext


class NormalizeTestCase():
    """Testing `MPFixedContext.normalize()`"""

    def test_normalize(self, num_values: int = 10_000):
        random.seed(1)
        for _ in range(num_values):
            nmin = random.randint(-64, 64)
            ctx = MPFixedContext(nmin)

            # a representable value: its digits at or above `nmin + 1`
            s = random.choice([False, True])
            c = random.randint(0, 2**64 - 1)
            exp = random.randint(nmin + 1, nmin + 64)
            x = Float(s=s, exp=exp, c=c, ctx=ctx)

            y = ctx.normalize(x)
            assert y.as_rational() == x.as_rational() and y.s == x.s, f'x={x}, y={y}'
            assert y.exp == nmin + 1, f'x={x}, y={y}'
