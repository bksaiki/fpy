import unittest

import unittest

from fpy2 import (
    S1E5M2, S1E4M3,
    MX_E5M2, MX_E4M3, MX_E3M2, MX_E2M3, MX_E2M1,
    FP8P1, FP8P2, FP8P3, FP8P4, FP8P5, FP8P6, FP8P7
)

class TestGraphcoreParams(unittest.TestCase):
    """Test Graphcore floating-point parameters."""

    def test_s1e5m2_params(self):
        self.assertEqual(S1E5M2.nbits, 8)
        self.assertEqual(S1E5M2.pmax, 3)
        self.assertEqual(S1E5M2.emin, -15)
        self.assertEqual(S1E5M2.emax, 15)
        self.assertEqual(S1E5M2.expmin, -17)
        self.assertEqual(S1E5M2.expmax, 13)

    def test_s1e4m3_params(self):
        self.assertEqual(S1E4M3.nbits, 8)
        self.assertEqual(S1E4M3.pmax, 4)
        self.assertEqual(S1E4M3.emin, -7)
        self.assertEqual(S1E4M3.emax, 7)
        self.assertEqual(S1E4M3.expmin, -10)
        self.assertEqual(S1E4M3.expmax, 4)


class TestOCPParams(unittest.TestCase):
    """Test OCP floating-point parameters."""

    def test_mx_e5m2_params(self):
        self.assertEqual(MX_E5M2.nbits, 8)
        self.assertEqual(MX_E5M2.pmax, 3)
        self.assertEqual(MX_E5M2.emin, -14)
        self.assertEqual(MX_E5M2.emax, 15)
        self.assertEqual(MX_E5M2.expmin, -16)
        self.assertEqual(MX_E5M2.expmax, 13)

    def test_mx_e4m3_params(self):
        self.assertEqual(MX_E4M3.nbits, 8)
        self.assertEqual(MX_E4M3.pmax, 4)
        self.assertEqual(MX_E4M3.emin, -6)
        self.assertEqual(MX_E4M3.emax, 8)
        self.assertEqual(MX_E4M3.expmin, -9)
        self.assertEqual(MX_E4M3.expmax, 5)

    def test_mx_e3m2_params(self):
        self.assertEqual(MX_E3M2.nbits, 6)
        self.assertEqual(MX_E3M2.pmax, 3)
        self.assertEqual(MX_E3M2.emin, -2)
        self.assertEqual(MX_E3M2.emax, 4)
        self.assertEqual(MX_E3M2.expmin, -4)
        self.assertEqual(MX_E3M2.expmax, 2)

    def test_mx_e2m3_params(self):
        self.assertEqual(MX_E2M3.nbits, 6)
        self.assertEqual(MX_E2M3.pmax, 4)
        self.assertEqual(MX_E2M3.emin, 0)
        self.assertEqual(MX_E2M3.emax, 2)
        self.assertEqual(MX_E2M3.expmin, -3)
        self.assertEqual(MX_E2M3.expmax, -1)

    def test_mx_e2m1_params(self):
        self.assertEqual(MX_E2M1.nbits, 4)
        self.assertEqual(MX_E2M1.pmax, 2)
        self.assertEqual(MX_E2M1.emin, 0)
        self.assertEqual(MX_E2M1.emax, 2)
        self.assertEqual(MX_E2M1.expmin, -1)
        self.assertEqual(MX_E2M1.expmax, 1)


class TestIEEEP3109Params(unittest.TestCase):
    """Test IEEE P3109 floating-point parameters."""

    def test_fp8p1_params(self):
        self.assertEqual(FP8P1.nbits, 8)
        self.assertEqual(FP8P1.pmax, 1)
        self.assertEqual(FP8P1.emin, -62)
        self.assertEqual(FP8P1.emax, 63)
        self.assertEqual(FP8P1.expmin, -62)
        self.assertEqual(FP8P1.expmax, 63)

    def test_fp8p2_params(self):
        self.assertEqual(FP8P2.nbits, 8)
        self.assertEqual(FP8P2.pmax, 2)
        self.assertEqual(FP8P2.emin, -31)
        self.assertEqual(FP8P2.emax, 31)
        self.assertEqual(FP8P2.expmin, -32)
        self.assertEqual(FP8P2.expmax, 30)

    def test_fp8p3_params(self):
        self.assertEqual(FP8P3.nbits, 8)
        self.assertEqual(FP8P3.pmax, 3)
        self.assertEqual(FP8P3.emin, -15)
        self.assertEqual(FP8P3.emax, 15)
        self.assertEqual(FP8P3.expmin, -17)
        self.assertEqual(FP8P3.expmax, 13)

    def test_fp8p4_params(self):
        self.assertEqual(FP8P4.nbits, 8)
        self.assertEqual(FP8P4.pmax, 4)
        self.assertEqual(FP8P4.emin, -7)
        self.assertEqual(FP8P4.emax, 7)
        self.assertEqual(FP8P4.expmin, -10)
        self.assertEqual(FP8P4.expmax, 4)

    def test_fp8p5_params(self):
        self.assertEqual(FP8P5.nbits, 8)
        self.assertEqual(FP8P5.pmax, 5)
        self.assertEqual(FP8P5.emin, -3)
        self.assertEqual(FP8P5.emax, 3)
        self.assertEqual(FP8P5.expmin, -7)
        self.assertEqual(FP8P5.expmax, -1)

    def test_fp8p6_params(self):
        self.assertEqual(FP8P6.nbits, 8)
        self.assertEqual(FP8P6.pmax, 6)
        self.assertEqual(FP8P6.emin, -1)
        self.assertEqual(FP8P6.emax, 1)
        self.assertEqual(FP8P6.expmin, -6)
        self.assertEqual(FP8P6.expmax, -4)

    def test_fp8p7_params(self):
        self.assertEqual(FP8P7.nbits, 8)
        self.assertEqual(FP8P7.pmax, 7)
        self.assertEqual(FP8P7.emin, 0)
        self.assertEqual(FP8P7.emax, 0)
        self.assertEqual(FP8P7.expmin, -6)
        self.assertEqual(FP8P7.expmax, -6)
