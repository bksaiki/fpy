import math
import struct

import fpy2 as fp

from hypothesis import given

from ..generators import floats


_PREC_MAX=24
_EXP_MIN=-100
_EXP_MAX=100


def _same(x: fp.Float, y: float | fp.Float) -> bool:
    return math.isnan(float(y)) if x.isnan else x == y


def _scaled(a: fp.Float, b: fp.Float, s: fp.Float) -> float | fp.Float:
    """`|a - b| / |s|` rounded once under FP64, for finite inputs."""
    n = abs(a.as_rational() - b.as_rational())
    if s == 0:
        return math.nan if n == 0 else math.inf
    return fp.FP64.round(n / abs(s.as_rational()))


def _ord(x: fp.Float) -> int:
    """Ordinal of *x* in FP64: its bit pattern, read as sign-magnitude."""
    i, = struct.unpack('<q', struct.pack('<d', float(x)))
    return -(i & ((1 << 63) - 1)) if i < 0 else i


class TestMetrics():
    """Testing `fpy2.libraries.metrics` functionality."""

    @given(
        floats(prec_max=_PREC_MAX, exp_min=_EXP_MIN, exp_max=_EXP_MAX),
        floats(prec_max=_PREC_MAX, exp_min=_EXP_MIN, exp_max=_EXP_MAX)
    )
    def test_absolute_error(self, a: fp.Float, b: fp.Float):
        """Testing `absolute_error` function"""
        err = fp.libraries.metrics.absolute_error(a, b)
        assert isinstance(err, fp.Float)
        assert _same(err, abs(float(a) - float(b)))

    @given(
        floats(prec_max=_PREC_MAX, exp_min=_EXP_MIN, exp_max=_EXP_MAX),
        floats(prec_max=_PREC_MAX, exp_min=_EXP_MIN, exp_max=_EXP_MAX),
        floats(prec_max=_PREC_MAX, exp_min=_EXP_MIN, exp_max=_EXP_MAX)
    )
    def test_scaled_error(self, a: fp.Float, b: fp.Float, scale: fp.Float):
        """Testing `scaled_error` function"""
        err = fp.libraries.metrics.scaled_error(a, b, scale)
        assert isinstance(err, fp.Float)
        if any(v.is_nar() for v in (a, b, scale)):
            assert err.isnan or err >= 0
        else:
            assert _same(err, _scaled(a, b, scale))

    @given(
        floats(prec_max=_PREC_MAX, exp_min=_EXP_MIN, exp_max=_EXP_MAX),
        floats(prec_max=_PREC_MAX, exp_min=_EXP_MIN, exp_max=_EXP_MAX)
    )
    def test_relative_error(self, a: fp.Float, b: fp.Float):
        """Testing `relative_error` function"""
        err = fp.libraries.metrics.relative_error(a, b)
        assert isinstance(err, fp.Float)
        assert _same(err, fp.libraries.metrics.scaled_error(a, b, b))

    @given(
        floats(prec_max=_PREC_MAX, exp_min=_EXP_MIN, exp_max=_EXP_MAX, allow_nan=False, allow_infinity=False),
        floats(prec_max=_PREC_MAX, exp_min=_EXP_MIN, exp_max=_EXP_MAX, allow_nan=False, allow_infinity=False)
    )
    def test_ordinal_error(self, a: fp.Float, b: fp.Float):
        """Testing `ordinal_error` function"""
        err = fp.libraries.metrics.ordinal_error(a, b)
        assert isinstance(err, fp.Float)
        assert err == abs(_ord(a) - _ord(b))
