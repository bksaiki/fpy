import random
import unittest

from fpy2 import Float, MPContext, RM

class NormalizeTestCase(unittest.TestCase):
    """Testing `IEEEContext.normalize()`"""

    def test_normalize(self, num_values: int = 10_000, p_max: int = 128):
        random.seed(1)
        for _ in range(num_values):
            # sample maximum starting precision
            p_init = random.randint(2, p_max)
            ctx = MPContext(p_max, RM.RNE)

            # sample random value
            s = random.choice([False, True])
            c = random.randint(0, 2**p_init - 1)
            e = random.randint(-1024, 1024)
            x = Float(s=s, e=e, c=c, ctx=ctx)

            # normalize value
            y = ctx.normalize(x)

            # check expected
            if x.is_zero():
                self.assertTrue(y.is_zero(), f'x={x}, y={y}')
            else:
                self.assertEqual(x, y, f'x={x}, y={y}')
                self.assertEqual(y.p, p_max, f'x={x}, y={y} (p={y.p})')
