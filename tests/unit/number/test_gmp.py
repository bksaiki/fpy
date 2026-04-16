import fpy2 as fp

from fpy2.number.gmputils import float_to_mpfr, mpfr_to_float
from hypothesis import given

from ..generators import floats


class TestGMPConversion():
    """Testing conversion between `fpy2` number and `gmpy2` numbers."""

    def assertEqualOrNan(self, a: fp.Float, b: fp.Float, msg = None):
        if a.isnan or b.isnan:
            assert a.isnan and b.isnan, msg
        else:
            assert a == b, msg

    @given(floats(prec_max=32, exp_min=-100, exp_max=100))
    def test_to_gmp(self, x: fp.Float):
        y = float_to_mpfr(x)
        x2 = mpfr_to_float(y)
        assert isinstance(x2, fp.Float)
        self.assertEqualOrNan(x, x2)
