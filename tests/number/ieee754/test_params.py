import unittest

from fpy2 import FP128, FP32, FP64, FP16


class TestIEEE754Params(unittest.TestCase):
    """Test IEEE-754 floating-point parameters."""

    def test_fp128_params(self):
        self.assertEqual(FP128.nbits, 128)
        self.assertEqual(FP128.pmax, 113)
        self.assertEqual(FP128.emin, -16382)
        self.assertEqual(FP128.emax, 16383)
        self.assertEqual(FP128.expmin, -16494)
        self.assertEqual(FP128.expmax, 16271)

    def test_fp64_params(self):
        self.assertEqual(FP64.nbits, 64)
        self.assertEqual(FP64.pmax, 53)
        self.assertEqual(FP64.emin, -1022)
        self.assertEqual(FP64.emax, 1023)
        self.assertEqual(FP64.expmin, -1074)
        self.assertEqual(FP64.expmax, 971)

    def test_fp32_params(self):
        self.assertEqual(FP32.nbits, 32)
        self.assertEqual(FP32.pmax, 24)
        self.assertEqual(FP32.emin, -126)
        self.assertEqual(FP32.emax, 127)
        self.assertEqual(FP32.expmin, -149)
        self.assertEqual(FP32.expmax, 104)

    def test_fp16_params(self):
        self.assertEqual(FP16.nbits, 16)
        self.assertEqual(FP16.pmax, 11)
        self.assertEqual(FP16.emin, -14)
        self.assertEqual(FP16.emax, 15)
        self.assertEqual(FP16.expmin, -24)
        self.assertEqual(FP16.expmax, 5)
