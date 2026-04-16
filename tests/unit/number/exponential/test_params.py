import fpy2 as fp

class TestParams():
    """Test `ExpContext` parameters."""

    def test_common(self):
        e8m0 = fp.ExpContext(8, 0)
        assert e8m0.nbits == 8
        assert e8m0.eoffset == 0
        assert e8m0.emin == -127
        assert e8m0.emax == 127
        assert e8m0.ebias == 127

        e8m0_p1 = fp.ExpContext(8, 1)
        assert e8m0_p1.nbits == 8
        assert e8m0_p1.eoffset == 1
        assert e8m0_p1.emin == -126
        assert e8m0_p1.emax == 128
        assert e8m0_p1.ebias == 126

        e8m0_m1 = fp.ExpContext(8, -1)
        assert e8m0_m1.nbits == 8
        assert e8m0_m1.eoffset == -1
        assert e8m0_m1.emin == -128
        assert e8m0_m1.emax == 126
        assert e8m0_m1.ebias == 128
