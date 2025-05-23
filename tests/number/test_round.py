import unittest

from fpy2 import RealFloat, RM


class RoundTestCase(unittest.TestCase):
    """Testing `RealFloat.round()`"""

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
            x = RealFloat(exp=exp, c=c)
            p, n = x._round_params(max_p=max_p, min_n=min_n)
            self.assertEqual(p, expect_p, f'x={x}, max_p={max_p}, min_n={min_n}, p={p}, expect_p={expect_p}')
            self.assertEqual(n, expect_n, f'x={x}, max_p={max_p}, min_n={min_n}, n={n}, expect_n={expect_n}')


    def test_round_float_p2(self):
        inputs = [
            # 8 * 2 ** -3 (representable)
            (-3, 8, -1, 2, RM.RNE), # 8 * 2 ** -3 => 1 * 2 ** -1
            (-3, 8, -1, 2, RM.RNA), # 8 * 2 ** -3 => 1 * 2 ** -1
            (-3, 8, -1, 2, RM.RTP), # 8 * 2 ** -3 => 1 * 2 ** -1
            (-3, 8, -1, 2, RM.RTN), # 8 * 2 ** -3 => 1 * 2 ** -1
            (-3, 8, -1, 2, RM.RTZ), # 8 * 2 ** -3 => 1 * 2 ** -1
            (-3, 8, -1, 2, RM.RAZ), # 8 * 2 ** -3 => 1 * 2 ** -1
            # 9 * 2 ** -3 (below halfway)
            (-3, 9, -1, 2, RM.RNE), # 9 * 2 ** -3 => 1 * 2 ** -1 (down)
            (-3, 9, -1, 2, RM.RNA), # 9 * 2 ** -3 => 1 * 2 ** -1 (down)
            (-3, 9, -1, 3, RM.RTP), # 9 * 2 ** -3 => 1 * 3 ** -1 (up)
            (-3, 9, -1, 2, RM.RTN), # 9 * 2 ** -3 => 1 * 2 ** -1 (down)
            (-3, 9, -1, 2, RM.RTZ), # 9 * 2 ** -3 => 1 * 2 ** -1 (down)
            (-3, 9, -1, 3, RM.RAZ), # 9 * 2 ** -3 => 1 * 3 ** -1 (up)
            # 10 * 2 ** -3 (exactly halfway)
            (-3, 10, -1, 2, RM.RNE), # 10 * 2 ** -3 => 1 * 2 ** -1 (down)
            (-3, 10, -1, 3, RM.RNA), # 10 * 2 ** -3 => 1 * 3 ** -1 (up)
            (-3, 10, -1, 3, RM.RTP), # 10 * 2 ** -3 => 1 * 3 ** -1 (up)
            (-3, 10, -1, 2, RM.RTN), # 10 * 2 ** -3 => 1 * 2 ** -1 (down)
            (-3, 10, -1, 2, RM.RTZ), # 10 * 2 ** -3 => 1 * 2 ** -1 (down)
            (-3, 10, -1, 3, RM.RAZ), # 10 * 2 ** -3 => 1 * 3 ** -1 (up)
            # 11 * 2 ** -3 (above halfway)
            (-3, 11, -1, 3, RM.RNE), # 11 * 2 ** -3 => 1 * 3 ** -1 (up)
            (-3, 11, -1, 3, RM.RNA), # 11 * 2 ** -3 => 1 * 3 ** -1 (up)
            (-3, 11, -1, 3, RM.RTP), # 11 * 2 ** -3 => 1 * 3 ** -1 (up)
            (-3, 11, -1, 2, RM.RTN), # 11 * 2 ** -3 => 1 * 2 ** -1 (down)
            (-3, 11, -1, 2, RM.RTZ), # 11 * 2 ** -3 => 1 * 2 ** -1 (down
            (-3, 11, -1, 3, RM.RAZ), # 11 * 2 ** -3 => 1 * 3 ** -1 (up)
            # 12 * 2 ** -3 (representable)
            (-3, 12, -1, 3, RM.RNE), # 12 * 2 ** -3 => 1 * 3 ** -1
            (-3, 12, -1, 3, RM.RNA), # 12 * 2 ** -3 => 1 * 3 ** -1
            (-3, 12, -1, 3, RM.RTP), # 12 * 2 ** -3 => 1 * 3 ** -1
            (-3, 12, -1, 3, RM.RTN), # 12 * 2 ** -3 => 1 * 3 ** -1
            (-3, 12, -1, 3, RM.RTZ), # 12 * 2 ** -3 => 1 * 3 ** -1
            (-3, 12, -1, 3, RM.RAZ), # 12 * 2 ** -3 => 1 * 3 ** -1
        ]

        for exp, c, exp_rounded, c_rounded, rm in inputs:
            x = RealFloat(exp=exp, c=c)
            x_rounded = x.round(max_p=2, rm=rm)
            expect = RealFloat(exp=exp_rounded, c=c_rounded)
            self.assertEqual(x_rounded, expect, f'x={x}, rm={rm!r}, x_rounded={x_rounded}, expect={expect}')
