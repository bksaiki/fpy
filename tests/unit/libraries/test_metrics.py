import fpy2 as fp
import unittest

from fractions import Fraction
from hypothesis import given, strategies as st

from ..generators import floats


_PREC_MAX=24
_EXP_MIN=-100
_EXP_MAX=100


class TestMetrics(unittest.TestCase):
    """Testing `fpy2.libraries.metrics` functionality."""

    @given(
        floats(prec_max=_PREC_MAX, exp_min=_EXP_MIN, exp_max=_EXP_MAX),
        floats(prec_max=_PREC_MAX, exp_min=_EXP_MIN, exp_max=_EXP_MAX)
    )
    def test_absolute_error(self, a: fp.Float, b: fp.Float):
        """Testing `absolute_error` function"""
        err = fp.libraries.metrics.absolute_error(a, b)
        self.assertIsInstance(err, fp.Float)
        self.assertTrue(err.isnan or err >= 0)

    @given(
        floats(prec_max=_PREC_MAX, exp_min=_EXP_MIN, exp_max=_EXP_MAX),
        floats(prec_max=_PREC_MAX, exp_min=_EXP_MIN, exp_max=_EXP_MAX),
        floats(prec_max=_PREC_MAX, exp_min=_EXP_MIN, exp_max=_EXP_MAX)
    )
    def test_scaled_error(self, a: fp.Float, b: fp.Float, scale: fp.Float):
        """Testing `scaled_error` function"""
        err = fp.libraries.metrics.scaled_error(a, b, scale)
        self.assertIsInstance(err, fp.Float)
        self.assertTrue(err.isnan or err >= 0)

    @given(
        floats(prec_max=_PREC_MAX, exp_min=_EXP_MIN, exp_max=_EXP_MAX),
        floats(prec_max=_PREC_MAX, exp_min=_EXP_MIN, exp_max=_EXP_MAX)
    )
    def test_relative_error(self, a: fp.Float, b: fp.Float):
        """Testing `relative_error` function"""
        err = fp.libraries.metrics.relative_error(a, b)
        self.assertIsInstance(err, fp.Float)
        self.assertTrue(err.isnan or err >= 0)

    @given(
        floats(prec_max=_PREC_MAX, exp_min=_EXP_MIN, exp_max=_EXP_MAX, allow_nan=False, allow_infinity=False),
        floats(prec_max=_PREC_MAX, exp_min=_EXP_MIN, exp_max=_EXP_MAX, allow_nan=False, allow_infinity=False)
    )
    def test_ordinal_error(self, a: fp.Float, b: fp.Float):
        """Testing `ordinal_error` function"""
        err = fp.libraries.metrics.ordinal_error(a, b)
        self.assertIsInstance(err, fp.Float)
        self.assertTrue(err >= 0)
