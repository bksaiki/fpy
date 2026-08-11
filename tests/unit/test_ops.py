"""
Tests for `fpy2/ops.py`.
"""

from fractions import Fraction

import fpy2 as fp
import pytest

from hypothesis import given, strategies as st

from .generators import floats, common_contexts


class TestRoundedOps():

    def assertEqualOrNan(self, a: fp.Float, b: fp.Float, msg = None):
        if a.isnan or b.isnan:
            assert a.isnan and b.isnan, msg
        else:
            assert a == b, msg

    @given(
        common_contexts(),
        floats(prec_max=32, exp_min=-100, exp_max=100, allow_infinity=False, allow_nan=False),
        floats(prec_max=32, exp_min=-100, exp_max=100, allow_infinity=False, allow_nan=False),
        st.integers(min_value=0, max_value=16),
        st.integers(min_value=0, max_value=16)
    )
    def test_add(self, ctx: fp.Context, x: fp.Float, y: fp.Float, shiftx: int, shifty: int) -> None:
        x2 = x.normalize(x.p + shiftx)
        y2 = y.normalize(y.p + shifty)
        r1 = fp.add(x, y, ctx)
        r2 = fp.add(x2, y2, ctx)
        self.assertEqualOrNan(r1, r2, f'{x} + {y}: {r1} != {r2} with shifts {shiftx}, {shifty}')

    @given(
        common_contexts(),
        floats(prec_max=32, exp_min=-100, exp_max=100, allow_infinity=False, allow_nan=False),
        floats(prec_max=32, exp_min=-100, exp_max=100, allow_infinity=False, allow_nan=False),
        st.integers(min_value=0, max_value=16),
        st.integers(min_value=0, max_value=16)
    )
    def test_mul(self, ctx: fp.Context, x: fp.Float, y: fp.Float, shiftx: int, shifty: int) -> None:
        x2 = x.normalize(x.p + shiftx)
        y2 = y.normalize(y.p + shifty)
        r1 = fp.mul(x, y, ctx)
        r2 = fp.mul(x2, y2, ctx)
        self.assertEqualOrNan(r1, r2, f'{x} * {y}: {r1} != {r2} with shifts {shiftx}, {shifty}')

    @given(
        common_contexts().filter(lambda ctx: ctx is not fp.REAL),
        floats(prec_max=32, exp_min=-100, exp_max=100, allow_infinity=False, allow_nan=False),
        st.integers(min_value=0, max_value=16),
        st.integers(min_value=0, max_value=16)
    )
    def test_sin(self, ctx: fp.Context, x: fp.Float, shiftx: int, shifty: int) -> None:
        x2 = x.normalize(x.p + shiftx)
        r1 = fp.sin(x, ctx)
        r2 = fp.sin(x2, ctx)
        self.assertEqualOrNan(r1, r2, f'sin({x}): {r1} != {r2} with shifts {shiftx}, {shifty}')

    @given(
        common_contexts().filter(lambda ctx: ctx is not fp.REAL),
        floats(prec_max=32, exp_min=-100, exp_max=100, allow_infinity=False, allow_nan=False),
        st.integers(min_value=0, max_value=16),
        st.integers(min_value=0, max_value=16)
    )
    def test_cos(self, ctx: fp.Context, x: fp.Float, shiftx: int, shifty: int) -> None:
        x2 = x.normalize(x.p + shiftx)
        r1 = fp.cos(x, ctx)
        r2 = fp.cos(x2, ctx)
        self.assertEqualOrNan(r1, r2, f'cos({x}): {r1} != {r2} with shifts {shiftx}, {shifty}')


class TestTupleAccessors:
    """Eager semantics of the ``fst`` / ``snd`` tuple accessors."""

    def test_fst(self):
        assert fp.fst((1, 2)) == 1

    def test_snd(self):
        assert fp.snd((1, 2)) == 2

    def test_fst_requires_two_elements(self):
        with pytest.raises(ValueError):
            fp.fst((9,))
        with pytest.raises(ValueError):
            fp.fst((1, 2, 3))

    def test_snd_requires_two_elements(self):
        with pytest.raises(ValueError):
            fp.snd((9,))
        with pytest.raises(ValueError):
            fp.snd((1, 2, 3))


class TestPowReal:
    """``pow`` under ``REAL``, which is exact only for a non-negative
    integer exponent (the case ``**`` used to desugar to)."""

    @given(
        floats(prec_max=16, exp_min=-20, exp_max=20, allow_infinity=False, allow_nan=False),
        st.integers(min_value=0, max_value=8),
    )
    def test_matches_repeated_mul(self, x: fp.Float, n: int) -> None:
        acc = fp.round(1, fp.REAL)
        for _ in range(n):
            acc = fp.mul(acc, x, fp.REAL)
        assert fp.pow(x, n, fp.REAL) == acc, f'{x} ** {n}'

    @pytest.mark.parametrize('x, n, expect', [
        (2, 10, 1024),
        (-2, 3, -8),
        (-2, 2, 4),
        (Fraction(2, 3), 3, Fraction(8, 27)),
        # `** 0` is 1 for every base, so no operand is inspected
        (2, 0, 1),
        (float('inf'), 0, 1),
        (float('nan'), 0, 1),
    ])
    def test_exact(self, x, n, expect) -> None:
        assert fp.pow(x, n, fp.REAL) == expect

    def test_signed_zero(self) -> None:
        # the sign survives only an odd exponent
        assert fp.pow(-0.0, 3, fp.REAL).s
        assert not fp.pow(-0.0, 2, fp.REAL).s

    @pytest.mark.parametrize('x, n, isinf, s', [
        (float('inf'), 2, True, False),
        (float('-inf'), 2, True, False),
        (float('-inf'), 3, True, True),
    ])
    def test_infinite_base(self, x, n, isinf, s) -> None:
        r = fp.pow(x, n, fp.REAL)
        assert r.isinf == isinf and r.s == s

    def test_nan_base(self) -> None:
        assert fp.pow(float('nan'), 2, fp.REAL).isnan

    @pytest.mark.parametrize('n', [-1, -2, 0.5, 2.5, float('inf'), float('nan')])
    def test_inexact_exponent_declines(self, n) -> None:
        # a negative exponent needs division and a non-integral one is
        # irrational in general; neither is exact, so `REAL` refuses
        with pytest.raises(NotImplementedError):
            fp.pow(2.0, n, fp.REAL)
