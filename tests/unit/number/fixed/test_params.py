import unittest

from fpy2 import (
    UINT8, UINT16, UINT32, UINT64,
    SINT8, SINT16, SINT32, SINT64,
    RealFloat
)

class TestIntegerParams(unittest.TestCase):
    """Test integer parameters."""

    def test_sint8_params(self):
        self.assertEqual(SINT8.nbits, 8)
        self.assertEqual(SINT8.maxval(s=True), RealFloat.from_int(-128))
        self.assertEqual(SINT8.maxval(), RealFloat.from_int(127))

    def test_sint16_params(self):
        self.assertEqual(SINT16.nbits, 16)
        self.assertEqual(SINT16.maxval(s=True), RealFloat.from_int(-32768))
        self.assertEqual(SINT16.maxval(), RealFloat.from_int(32767))

    def test_sint32_params(self):
        self.assertEqual(SINT32.nbits, 32)
        self.assertEqual(SINT32.maxval(s=True), RealFloat.from_int(-2147483648))
        self.assertEqual(SINT32.maxval(), RealFloat.from_int(2147483647))

    def test_sint64_params(self):
        self.assertEqual(SINT64.nbits, 64)
        self.assertEqual(SINT64.maxval(s=True), RealFloat.from_int(-9223372036854775808))
        self.assertEqual(SINT64.maxval(), RealFloat.from_int(9223372036854775807))

    def test_uint8_params(self):
        self.assertEqual(UINT8.nbits, 8)
        self.assertEqual(UINT8.maxval(), RealFloat.from_int(255))

    def test_uint16_params(self):
        self.assertEqual(UINT16.nbits, 16)
        self.assertEqual(UINT16.maxval(), RealFloat.from_int(65535))

    def test_uint32_params(self):
        self.assertEqual(UINT32.nbits, 32)
        self.assertEqual(UINT32.maxval(), RealFloat.from_int(4294967295))

    def test_uint64_params(self):
        self.assertEqual(UINT64.nbits, 64)
        self.assertEqual(UINT64.maxval(), RealFloat.from_int(18446744073709551615))
