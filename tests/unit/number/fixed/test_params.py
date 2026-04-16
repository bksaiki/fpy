
from fpy2 import (
    UINT8, UINT16, UINT32, UINT64,
    SINT8, SINT16, SINT32, SINT64,
    RealFloat
)

class TestIntegerParams():
    """Test integer parameters."""

    def test_sint8_params(self):
        assert SINT8.nbits == 8
        assert SINT8.maxval(s=True) == RealFloat.from_int(-128)
        assert SINT8.maxval() == RealFloat.from_int(127)
        assert SINT8.largest() == RealFloat.from_int(127)
        assert SINT8.smallest() == RealFloat.from_int(-128)

    def test_sint16_params(self):
        assert SINT16.nbits == 16
        assert SINT16.maxval(s=True) == RealFloat.from_int(-32768)
        assert SINT16.maxval() == RealFloat.from_int(32767)
        assert SINT16.largest() == RealFloat.from_int(32767)
        assert SINT16.smallest() == RealFloat.from_int(-32768)

    def test_sint32_params(self):
        assert SINT32.nbits == 32
        assert SINT32.maxval(s=True) == RealFloat.from_int(-2147483648)
        assert SINT32.maxval() == RealFloat.from_int(2147483647)
        assert SINT32.largest() == RealFloat.from_int(2147483647)

    def test_sint64_params(self):
        assert SINT64.nbits == 64
        assert SINT64.maxval(s=True) == RealFloat.from_int(-9223372036854775808)
        assert SINT64.maxval() == RealFloat.from_int(9223372036854775807)
        assert SINT64.largest() == RealFloat.from_int(9223372036854775807)
        assert SINT64.smallest() == RealFloat.from_int(-9223372036854775808)

    def test_uint8_params(self):
        assert UINT8.nbits == 8
        assert UINT8.maxval() == RealFloat.from_int(255)
        assert UINT8.largest() == RealFloat.from_int(255)
        assert UINT8.smallest() == RealFloat.from_int(0)

    def test_uint16_params(self):
        assert UINT16.nbits == 16
        assert UINT16.maxval() == RealFloat.from_int(65535)
        assert UINT16.largest() == RealFloat.from_int(65535)
        assert UINT16.smallest() == RealFloat.from_int(0)

    def test_uint32_params(self):
        assert UINT32.nbits == 32
        assert UINT32.maxval() == RealFloat.from_int(4294967295)
        assert UINT32.largest() == RealFloat.from_int(4294967295)
        assert UINT32.smallest() == RealFloat.from_int(0)

    def test_uint64_params(self):
        assert UINT64.nbits == 64
        assert UINT64.maxval() == RealFloat.from_int(18446744073709551615)
        assert UINT64.largest() == RealFloat.from_int(18446744073709551615)
        assert UINT64.smallest() == RealFloat.from_int(0)
