import fpy2 as fp
import unittest


class TestRealFloatMethods(unittest.TestCase):
    """Testing `RealFloat.is_more_significant()`"""

    def test_is_more_significant(self):
        x = fp.RealFloat(c=7, exp=0)
        self.assertTrue(x.is_more_significant(-2))
        self.assertTrue(x.is_more_significant(-1))
        self.assertFalse(x.is_more_significant(0))
        self.assertFalse(x.is_more_significant(1))

        y = fp.RealFloat(c=3, exp=-2)
        self.assertTrue(y.is_more_significant(-4))
        self.assertTrue(y.is_more_significant(-3))
        self.assertFalse(y.is_more_significant(-2))
        self.assertFalse(y.is_more_significant(-1))
        self.assertFalse(y.is_more_significant(0))


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
