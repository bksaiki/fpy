import fpy2 as fp
import unittest


class RoundTestCase(unittest.TestCase):
    """Testing `RealFloat.round()`"""

    def test_round_float_p2(self):
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
