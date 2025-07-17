import fpy2 as fp
import unittest


class RoundTestCase(unittest.TestCase):
    """Testing `RealFloat.round()`"""

    def test_round(self):
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

    def test_round_stochastic(self):
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
