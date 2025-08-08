import fpy2 as fp
import unittest

class TestParams(unittest.TestCase):
    """Test `ExpContext` parameters."""

    def test_common(self):
        e8m0 = fp.ExpContext(8, 0)
        self.assertEqual(e8m0.nbits, 8)
        self.assertEqual(e8m0.eoffset, 0)
        self.assertEqual(e8m0.emin, -127)
        self.assertEqual(e8m0.emax, 127)

        e8m0_p1 = fp.ExpContext(8, 1)
        self.assertEqual(e8m0_p1.nbits, 8)
        self.assertEqual(e8m0_p1.eoffset, 1)
        self.assertEqual(e8m0_p1.emin, -126)
        self.assertEqual(e8m0_p1.emax, 128)

        e8m0_m1 = fp.ExpContext(8, -1)
        self.assertEqual(e8m0_m1.nbits, 8)
        self.assertEqual(e8m0_m1.eoffset, -1)
        self.assertEqual(e8m0_m1.emin, -128)
        self.assertEqual(e8m0_m1.emax, 126)
