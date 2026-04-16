
from fpy2 import FP128, FP32, FP64, FP16


class TestIEEE754Params():
    """Test IEEE-754 floating-point parameters."""

    def test_fp128_params(self):
        assert FP128.nbits == 128
        assert FP128.pmax == 113
        assert FP128.emin == -16382
        assert FP128.emax == 16383
        assert FP128.expmin == -16494
        assert FP128.expmax == 16271

    def test_fp64_params(self):
        assert FP64.nbits == 64
        assert FP64.pmax == 53
        assert FP64.emin == -1022
        assert FP64.emax == 1023
        assert FP64.expmin == -1074
        assert FP64.expmax == 971

    def test_fp32_params(self):
        assert FP32.nbits == 32
        assert FP32.pmax == 24
        assert FP32.emin == -126
        assert FP32.emax == 127
        assert FP32.expmin == -149
        assert FP32.expmax == 104

    def test_fp16_params(self):
        assert FP16.nbits == 16
        assert FP16.pmax == 11
        assert FP16.emin == -14
        assert FP16.emax == 15
        assert FP16.expmin == -24
        assert FP16.expmax == 5
