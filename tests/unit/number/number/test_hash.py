"""
Testing `RealFloat.__hash__` and `Float.__hash__`.

The key invariant is Python's hash/equality contract: whenever `a == b`,
`hash(a) == hash(b)`. Since `RealFloat` and `Float` compare equal to `int`,
`float`, and `Fraction`, their hashes must agree with the hashes those types
produce, i.e., they must follow the numeric hash specification.
"""

from fractions import Fraction

import fpy2 as fp
import pytest
from hypothesis import given, strategies as st

from ...generators import floats, real_floats

_PREC = 8
_EXP_MIN = -10
_EXP_MAX = 10


@st.composite
def number(draw):
    """
    Returns a strategy for generating an `int`, `float`, `Fraction`,
    `Float`, or `RealFloat`, all drawn from a narrow range of magnitudes
    so that distinct types collide on equal values often.
    """
    choice = draw(st.integers(min_value=0, max_value=4))
    if choice == 0:
        return draw(st.integers(min_value=-64, max_value=64))
    elif choice == 1:
        return draw(st.sampled_from([
            0.0, -0.0, 0.5, -0.5, 1.0, -1.0, 2.0, 0.25, -0.75, 3.5, 64.0,
            float('inf'), float('-inf'), float('nan'),
        ]))
    elif choice == 2:
        return draw(st.fractions(min_value=-64, max_value=64, max_denominator=16))
    elif choice == 3:
        return draw(floats(
            prec_max=_PREC, exp_min=_EXP_MIN, exp_max=_EXP_MAX,
            allow_nan=True, allow_infinity=True,
        ))
    else:
        return draw(real_floats(
            prec_max=_PREC, signed=True, exp_min=_EXP_MIN, exp_max=_EXP_MAX,
        ))


class TestHashEqContract:
    """`a == b` implies `hash(a) == hash(b)`."""

    @given(number(), number())
    def test_eq_implies_same_hash(self, a, b):
        if a == b:
            assert hash(a) == hash(b), f'{a!r} == {b!r} but hashes differ'

    @given(real_floats(prec_max=_PREC, exp_min=_EXP_MIN, exp_max=_EXP_MAX))
    def test_real_float_hash_is_stable(self, x):
        assert hash(x) == hash(x)

    @given(floats(prec_max=_PREC, exp_min=_EXP_MIN, exp_max=_EXP_MAX,
                  allow_nan=False))
    def test_float_hash_is_stable(self, x):
        h1 = hash(x)
        keepalive = [float('nan')]  # noqa: F841
        h2 = hash(x)
        assert h1 == h2


class TestNumericHashSpec:
    """`RealFloat`/`Float` hashes agree with the numeric tower."""

    @given(real_floats(prec_max=_PREC, exp_min=_EXP_MIN, exp_max=_EXP_MAX))
    def test_real_float_matches_rational(self, x):
        assert hash(x) == hash(x.as_rational())

    @given(floats(prec_max=_PREC, exp_min=_EXP_MIN, exp_max=_EXP_MAX,
                  allow_nan=False, allow_infinity=False))
    def test_finite_float_matches_rational(self, x):
        assert hash(x) == hash(x.as_rational())

    @given(st.integers())
    def test_integer_valued_matches_int(self, i):
        assert hash(fp.RealFloat.from_int(i)) == hash(i)
        assert hash(fp.Float.from_int(i)) == hash(i)

    @given(st.floats(allow_nan=False, allow_infinity=False))
    def test_float_valued_matches_float(self, f):
        assert hash(fp.RealFloat.from_float(f)) == hash(f)
        assert hash(fp.Float.from_float(f)) == hash(f)

    @given(st.integers(min_value=-64, max_value=64), st.integers(min_value=0, max_value=32))
    def test_dyadic_matches_fraction(self, m, k):
        # m / 2**k, exactly representable
        x = fp.RealFloat(m=m, exp=-k)
        assert hash(x) == hash(Fraction(m, 2 ** k))

    def test_scaled_by_power_of_two(self):
        # exercises both exp >= 0 and exp < 0 over a wide range
        for exp in range(-64, 65):
            x = fp.RealFloat(s=False, exp=exp, c=3)
            assert hash(x) == hash(Fraction(3) * Fraction(2) ** exp), f'exp={exp}'
            assert hash(-x) == hash(Fraction(-3) * Fraction(2) ** exp), f'exp={exp}'


class TestNumericEquivalence:
    """Numerically equal values hash equally regardless of encoding."""

    @given(
        real_floats(prec_max=_PREC, exp_min=_EXP_MIN, exp_max=_EXP_MAX),
        st.integers(min_value=0, max_value=32),
    )
    def test_shift_invariant(self, x, shift):
        # (s, exp, c) and (s, exp - shift, c << shift) denote the same value
        y = fp.RealFloat(s=x.s, exp=x.exp - shift, c=x.c << shift)
        assert x == y
        assert hash(x) == hash(y)

    @given(st.booleans(), st.integers(min_value=-32, max_value=32))
    def test_all_zeros_hash_as_zero(self, s, exp):
        # every zero, of either sign and any exponent, is numerically 0
        x = fp.RealFloat(s=s, exp=exp, c=0)
        assert x == 0
        assert hash(x) == hash(0)
        assert hash(fp.Float(x=x)) == hash(0)

    def test_negative_zero_hashes_as_zero(self):
        assert hash(fp.RealFloat(s=True, exp=0, c=0)) == hash(-0.0) == hash(0)
        assert hash(fp.Float.zero(s=True)) == hash(0)

    def test_flags_do_not_affect_hash(self):
        # flags are not part of equality, so they must not be part of the hash
        x = fp.RealFloat(s=False, exp=-2, c=5)
        y = fp.RealFloat(x=x, inexact=True, overflow=True, carry=True)
        assert x == y
        assert hash(x) == hash(y)

    def test_context_does_not_affect_hash(self):
        # the rounding context is not part of equality either
        x = fp.RealFloat(s=False, exp=-2, c=5)
        a = fp.Float.from_real(x)
        b = fp.Float.from_real(x, fp.FP64)
        assert a == b
        assert hash(a) == hash(b)

    @given(real_floats(prec_max=_PREC, exp_min=_EXP_MIN, exp_max=_EXP_MAX))
    def test_float_agrees_with_real_float(self, x):
        # a finite `Float` and its `RealFloat` payload are equal
        assert fp.Float.from_real(x) == x
        assert hash(fp.Float.from_real(x)) == hash(x)


class TestSpecialValues:

    def test_infinities(self):
        assert hash(fp.Float.inf()) == hash(float('inf'))
        assert hash(fp.Float.inf(s=True)) == hash(float('-inf'))
        assert hash(fp.Float.inf()) != hash(fp.Float.inf(s=True))
        # and the hashes are consistent with equality
        assert fp.Float.inf() == float('inf')
        assert fp.Float.inf(s=True) == float('-inf')

    @pytest.mark.xfail(reason=(
        '`Float.__hash__` returns `hash(float(\'nan\'))` for NaN, which is '
        'identity-based since Python 3.10 (bpo-43475) and so is not a '
        'constant: it varies per call, per NaN value, and per process'
    ))
    def test_nan_hash_is_constant(self):
        # NaN is not equal to itself, so no *particular* hash is required, but
        # the hash must at least be a constant: otherwise a NaN key cannot be
        # found in a dict, even by identity.
        nans = [fp.Float.nan(), fp.Float.nan(s=True), fp.Float(isnan=True)]
        hashes = []
        keepalive = []
        for n in nans:
            hashes.append(hash(n))
            # claim the address the previous temporary `float('nan')` used, so
            # the next one is not handed the same id
            keepalive.append(float('nan'))
        assert len(set(hashes)) == 1, f'NaN hashes differ: {hashes}'

    @pytest.mark.xfail(reason='same cause as `test_nan_hash_is_constant`')
    def test_nan_survives_dict_roundtrip(self):
        nan = fp.Float.nan()
        d = {nan: 'value'}
        keepalive = [float('nan') for _ in range(4)]  # noqa: F841
        assert nan in d
        assert d[nan] == 'value'


class TestContainers:
    """Hash-based containers behave as the equality relation implies."""

    def test_lookup_across_types(self):
        d = {fp.RealFloat(s=False, exp=-2, c=3): 'three quarters'}
        assert d[0.75] == 'three quarters'
        assert d[Fraction(3, 4)] == 'three quarters'
        assert d[fp.Float(s=False, exp=-4, c=12)] == 'three quarters'

        d2 = {7: 'seven'}
        assert d2[fp.RealFloat.from_int(7)] == 'seven'
        assert d2[fp.Float.from_int(7)] == 'seven'

    def test_set_dedups_equal_values(self):
        # all four denote 1
        vals = [
            1,
            1.0,
            Fraction(1),
            fp.RealFloat(s=False, exp=0, c=1),
            fp.RealFloat(s=False, exp=-3, c=8),
            fp.Float(s=False, exp=-1, c=2),
        ]
        assert len(set(vals)) == 1

    def test_set_keeps_distinct_values(self):
        vals = [
            fp.RealFloat(s=False, exp=0, c=1),
            fp.RealFloat(s=True, exp=0, c=1),
            fp.RealFloat(s=False, exp=-1, c=1),
            fp.Float.inf(),
            fp.Float.inf(s=True),
            fp.Float.zero(),
        ]
        assert len(set(vals)) == len(vals)
