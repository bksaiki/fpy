
import fpy2 as fp

from fractions import Fraction
from hypothesis import given, strategies as st

from ...generators import floats

PREC = 8
EXPMAX = 32
EXPMIN = -32

EXT_PREC = 4 * PREC
EXT_EXPMAX = 2 * EXPMAX
EXT_EXPMIN = 2 * EXPMIN

CTX = fp.MPFixedContext(EXPMIN - 1)


class OrdinalTestCase():
    """Testing ordinal methods of `MPSFloatContext`."""

    @given(floats(prec_max=PREC, exp_min=CTX.expmin, exp_max=EXPMAX, allow_nan=False, allow_infinity=False))
    def test_to_ordinal(self, x: fp.Float):
        ord = CTX.to_ordinal(x)
        assert isinstance(ord, int)

    @given(st.integers())
    def test_from_ordinal(self, ord: int):
        x = CTX.from_ordinal(ord)
        assert isinstance(x, fp.Float)

    @given(st.integers())
    def test_round_trip(self, ord: int):
        x = CTX.from_ordinal(ord)
        ord2 = CTX.to_ordinal(x)
        assert ord == ord2

    @given(floats(prec_max=EXT_PREC, exp_min=EXT_EXPMIN, exp_max=EXT_EXPMAX, allow_nan=False, allow_infinity=False))
    def test_to_fractional_ordinal(self, x: fp.Float):
        ord = CTX.to_fractional_ordinal(x)
        ord_above = CTX.to_ordinal(CTX.with_params(rm=fp.RM.RTP).round(x))
        ord_below = CTX.to_ordinal(CTX.with_params(rm=fp.RM.RTN).round(x))

        assert isinstance(ord, Fraction)
        assert ord >= ord_below
        assert ord <= ord_above
