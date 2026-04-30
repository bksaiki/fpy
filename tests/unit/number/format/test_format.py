"""
Tests for concrete format types, format() and from_format().

For each context class, we verify:
1. format() returns the correct format class with the correct parameters.
2. from_format() reconstructs a context equivalent to the original.
3. Format-specific methods (representable_under, to_ordinal, ...) work correctly.
4. Format equality and hashing work correctly.
"""

import pytest
import fpy2 as fp
import fpy2.number.format
from fractions import Fraction
from hypothesis import given, assume, strategies as st

from tests.unit.generators import (
    mp_float_contexts,
    mps_float_contexts,
    mp_fixed_contexts,
    efloat_contexts,
    fixed_contexts,
    sm_fixed_contexts,
    floats,
    common_contexts,
)


###########################################################
# Helper: generators for concrete format types

@st.composite
def mp_float_formats(draw, max_p: int = 64):
    p = draw(st.integers(1, max_p))
    return fp.number.format.MPFloatFormat(p)

@st.composite
def mp_fixed_formats(draw, min_n: int = -32, max_n: int = 32):
    n = draw(st.integers(min_n, max_n))
    enable_nan = draw(st.booleans())
    enable_inf = draw(st.booleans())
    return fp.number.format.MPFixedFormat(n, enable_nan, enable_inf)

@st.composite
def mps_float_formats(draw, max_p: int = 64, min_emin: int = -128, max_emin: int = 128):
    p = draw(st.integers(1, max_p))
    emin = draw(st.integers(min_emin, max_emin))
    return fp.number.format.MPSFloatFormat(p, emin)

@st.composite
def efloat_formats(draw, max_es: int = 4, max_nbits: int = 8,
                   min_eoffset: int = -16, max_eoffset: int = 16):
    # reuse efloat_contexts and extract format
    ctx = draw(efloat_contexts(max_es=max_es, max_nbits=max_nbits,
                               min_eoffset=min_eoffset, max_eoffset=max_eoffset))
    return ctx.format()

@st.composite
def ieee_formats(draw, max_es: int = 5, max_nbits: int = 16):
    es = draw(st.integers(2, max_es))
    nbits = draw(st.integers(es + 2, max_nbits))
    return fp.number.format.IEEEFormat(es, nbits)

@st.composite
def fixed_formats(draw, min_scale: int = -8, max_scale: int = 8, max_nbits: int = 8):
    ctx = draw(fixed_contexts(min_scale=min_scale, max_scale=max_scale, max_nbits=max_nbits))
    return ctx.format()

@st.composite
def sm_fixed_formats(draw, min_scale: int = -8, max_scale: int = 8, max_nbits: int = 8):
    ctx = draw(sm_fixed_contexts(min_scale=min_scale, max_scale=max_scale, max_nbits=max_nbits))
    return ctx.format()

@st.composite
def exp_formats(draw, max_nbits: int = 8, min_eoffset: int = -16, max_eoffset: int = 16):
    nbits = draw(st.integers(1, max_nbits))
    eoffset = draw(st.integers(min_eoffset, max_eoffset))
    return fp.number.format.ExpFormat(nbits, eoffset)


###########################################################
# Tests: abstract hierarchy membership

class TestFormatHierarchy:
    """Verify that format types implement the correct abstract interfaces."""

    def test_mp_float_format_is_format(self):
        fmt = fp.number.format.MPFloatFormat(53)
        assert isinstance(fmt, fp.number.format.Format)
        assert not isinstance(fmt, fp.number.format.OrdinalFormat)
        assert not isinstance(fmt, fp.number.format.SizedFormat)
        assert not isinstance(fmt, fp.number.format.EncodableFormat)

    def test_mp_fixed_format_is_ordinal(self):
        fmt = fp.number.format.MPFixedFormat(-10)
        assert isinstance(fmt, fp.number.format.Format)
        assert isinstance(fmt, fp.number.format.OrdinalFormat)
        assert not isinstance(fmt, fp.number.format.SizedFormat)

    def test_mps_float_format_is_ordinal(self):
        fmt = fp.number.format.MPSFloatFormat(24, -126)
        assert isinstance(fmt, fp.number.format.OrdinalFormat)
        assert not isinstance(fmt, fp.number.format.SizedFormat)

    def test_mpb_fixed_format_is_sized(self):
        fmt = fp.number.format.MPBFixedFormat(-1, fp.RealFloat(exp=0, c=127))
        assert isinstance(fmt, fp.number.format.OrdinalFormat)
        assert isinstance(fmt, fp.number.format.SizedFormat)
        assert not isinstance(fmt, fp.number.format.EncodableFormat)

    def test_mpb_float_format_is_sized(self):
        fmt = fp.number.format.MPBFloatFormat(24, -126, fp.RealFloat(exp=128, c=0xffffff))
        assert isinstance(fmt, fp.number.format.OrdinalFormat)
        assert isinstance(fmt, fp.number.format.SizedFormat)
        assert not isinstance(fmt, fp.number.format.EncodableFormat)

    def test_efloat_format_is_encodable(self):
        fmt = fp.MX_E4M3.format()
        assert isinstance(fmt, fp.number.format.OrdinalFormat)
        assert isinstance(fmt, fp.number.format.SizedFormat)
        assert isinstance(fmt, fp.number.format.EncodableFormat)

    def test_ieee_format_is_encodable(self):
        fmt = fp.number.format.IEEEFormat(11, 64)
        assert isinstance(fmt, fp.number.format.EFloatFormat)
        assert isinstance(fmt, fp.number.format.EncodableFormat)

    def test_fixed_format_is_encodable(self):
        fmt = fp.SINT8.format()
        assert isinstance(fmt, fp.number.format.MPBFixedFormat)
        assert isinstance(fmt, fp.number.format.EncodableFormat)

    def test_sm_fixed_format_is_encodable(self):
        fmt = fp.SMFixedContext(0, 8).format()
        assert isinstance(fmt, fp.number.format.MPBFixedFormat)
        assert isinstance(fmt, fp.number.format.EncodableFormat)

    def test_exp_format_is_encodable(self):
        fmt = fp.MX_E8M0.format()
        assert isinstance(fmt, fp.number.format.EncodableFormat)


###########################################################
# Tests: format() and from_format() round-trip

class TestFormatRoundTrip:
    """Test that format() and from_format() round-trip correctly."""

    @given(mp_float_contexts(max_p=64))
    def test_mp_float_round_trip(self, ctx: fp.MPFloatContext):
        fmt = ctx.format()
        assert isinstance(fmt, fp.number.format.MPFloatFormat)
        assert fmt.pmax == ctx.pmax
        # Reconstruct context from format
        ctx2 = fp.MPFloatContext.from_format(fmt)
        assert ctx2.pmax == ctx.pmax

    @given(mp_fixed_contexts(min_n=-32, max_n=32))
    def test_mp_fixed_round_trip(self, ctx: fp.MPFixedContext):
        fmt = ctx.format()
        assert isinstance(fmt, fp.number.format.MPFixedFormat)
        assert fmt.nmin == ctx.nmin
        assert fmt.enable_nan == ctx.enable_nan
        assert fmt.enable_inf == ctx.enable_inf
        ctx2 = fp.MPFixedContext.from_format(fmt)
        assert ctx2.nmin == ctx.nmin
        assert ctx2.enable_nan == ctx.enable_nan
        assert ctx2.enable_inf == ctx.enable_inf

    @given(mps_float_contexts(max_p=64, min_emin=-128, max_emin=128))
    def test_mps_float_round_trip(self, ctx: fp.MPSFloatContext):
        fmt = ctx.format()
        assert isinstance(fmt, fp.number.format.MPSFloatFormat)
        assert fmt.pmax == ctx.pmax
        assert fmt.emin == ctx.emin
        ctx2 = fp.MPSFloatContext.from_format(fmt)
        assert ctx2.pmax == ctx.pmax
        assert ctx2.emin == ctx.emin

    @given(efloat_contexts(max_es=4, max_nbits=8, min_eoffset=-8, max_eoffset=8))
    def test_efloat_round_trip(self, ctx: fp.EFloatContext):
        fmt = ctx.format()
        assert isinstance(fmt, fp.number.format.EFloatFormat)
        assert fmt.es == ctx.es
        assert fmt.nbits == ctx.nbits
        assert fmt.enable_inf == ctx.enable_inf
        assert fmt.nan_kind == ctx.nan_kind
        assert fmt.eoffset == ctx.eoffset
        ctx2 = fp.EFloatContext.from_format(fmt)
        assert ctx2.es == ctx.es
        assert ctx2.nbits == ctx.nbits
        assert ctx2.enable_inf == ctx.enable_inf

    @given(ieee_formats(max_es=5, max_nbits=16))
    def test_ieee_round_trip(self, fmt: fp.number.format.IEEEFormat):
        ctx = fp.IEEEContext.from_format(fmt)
        assert isinstance(ctx, fp.IEEEContext)
        fmt2 = ctx.format()
        assert isinstance(fmt2, fp.number.format.IEEEFormat)
        assert fmt == fmt2

    @given(fixed_contexts(min_scale=-8, max_scale=8, max_nbits=8))
    def test_fixed_round_trip(self, ctx: fp.FixedContext):
        fmt = ctx.format()
        assert isinstance(fmt, fp.number.format.FixedFormat)
        assert fmt.signed == ctx.signed
        assert fmt.scale == ctx.scale
        assert fmt.nbits == ctx.nbits
        ctx2 = fp.FixedContext.from_format(fmt)
        assert ctx2.signed == ctx.signed
        assert ctx2.scale == ctx.scale
        assert ctx2.nbits == ctx.nbits

    @given(sm_fixed_contexts(min_scale=-8, max_scale=8, max_nbits=8))
    def test_sm_fixed_round_trip(self, ctx: fp.SMFixedContext):
        fmt = ctx.format()
        assert isinstance(fmt, fp.number.format.SMFixedFormat)
        assert fmt.scale == ctx.scale
        assert fmt.nbits == ctx.nbits
        ctx2 = fp.SMFixedContext.from_format(fmt)
        assert ctx2.scale == ctx.scale
        assert ctx2.nbits == ctx.nbits

    @given(exp_formats(max_nbits=8, min_eoffset=-8, max_eoffset=8))
    def test_exp_round_trip(self, fmt: fp.number.format.ExpFormat):
        ctx = fp.ExpContext.from_format(fmt)
        assert isinstance(ctx, fp.ExpContext)
        assert ctx.nbits == fmt.nbits
        assert ctx.eoffset == fmt.eoffset
        fmt2 = ctx.format()
        assert fmt2 == fmt


###########################################################
# Tests: format equality and hashing

class TestFormatEquality:
    """Test that format __eq__ and __hash__ work correctly."""

    def test_mp_float_eq(self):
        fmt1 = fp.number.format.MPFloatFormat(53)
        fmt2 = fp.number.format.MPFloatFormat(53)
        assert fmt1 == fmt2
        assert hash(fmt1) == hash(fmt2)

    def test_mp_float_neq(self):
        fmt1 = fp.number.format.MPFloatFormat(24)
        fmt2 = fp.number.format.MPFloatFormat(53)
        assert fmt1 != fmt2

    def test_mp_fixed_eq(self):
        fmt1 = fp.number.format.MPFixedFormat(-4, True, False)
        fmt2 = fp.number.format.MPFixedFormat(-4, True, False)
        assert fmt1 == fmt2
        assert hash(fmt1) == hash(fmt2)

    def test_mp_fixed_neq_nmin(self):
        assert fp.number.format.MPFixedFormat(-4) != fp.number.format.MPFixedFormat(-5)

    def test_mp_fixed_neq_enable_nan(self):
        assert fp.number.format.MPFixedFormat(-4, True) != fp.number.format.MPFixedFormat(-4, False)

    def test_ieee_eq(self):
        fmt1 = fp.number.format.IEEEFormat(11, 64)
        fmt2 = fp.number.format.IEEEFormat(11, 64)
        assert fmt1 == fmt2
        assert hash(fmt1) == hash(fmt2)

    def test_ieee_neq(self):
        assert fp.number.format.IEEEFormat(8, 32) != fp.number.format.IEEEFormat(11, 64)

    def test_fixed_format_eq(self):
        fmt1 = fp.number.format.FixedFormat(True, 0, 8)
        fmt2 = fp.number.format.FixedFormat(True, 0, 8)
        assert fmt1 == fmt2
        assert hash(fmt1) == hash(fmt2)

    def test_sm_fixed_format_eq(self):
        fmt1 = fp.number.format.SMFixedFormat(0, 8)
        fmt2 = fp.number.format.SMFixedFormat(0, 8)
        assert fmt1 == fmt2
        assert hash(fmt1) == hash(fmt2)


###########################################################
# Tests: is_equiv

class TestFormatIsEquiv:
    """Test format equality (formerly is_equiv, now __eq__)."""

    def test_mp_float_is_equiv_self(self):
        fmt = fp.number.format.MPFloatFormat(53)
        assert fmt == fmt

    def test_mp_float_is_equiv_other(self):
        fmt1 = fp.number.format.MPFloatFormat(53)
        fmt2 = fp.number.format.MPFloatFormat(53)
        assert fmt1 == fmt2

    def test_mp_float_not_equiv_different_p(self):
        assert fp.number.format.MPFloatFormat(53) != fp.number.format.MPFloatFormat(24)

    def test_mp_fixed_is_equiv(self):
        fmt1 = fp.number.format.MPFixedFormat(-4, True, False)
        fmt2 = fp.number.format.MPFixedFormat(-4, True, False)
        assert fmt1 == fmt2

    def test_ieee_is_equiv(self):
        fmt1 = fp.number.format.IEEEFormat(11, 64)
        fmt2 = fp.number.format.IEEEFormat(11, 64)
        assert fmt1 == fmt2

    def test_ieee_not_equiv_different_nbits(self):
        assert fp.number.format.IEEEFormat(8, 32) != fp.number.format.IEEEFormat(11, 64)


###########################################################
# Tests: representable_under / canonical_under / normal_under

class TestFormatRepresentable:
    """Test representable_under for format types."""

    @given(
        mps_float_formats(max_p=8, min_emin=-64, max_emin=64).flatmap(
            lambda fmt: st.tuples(
                st.just(fmt),
                floats(prec_max=8, exp_min=-100, exp_max=100, allow_infinity=False, allow_nan=False, ctx=fp.MPSFloatContext.from_format(fmt))
            )
        )
    )
    def test_mps_float_representable(self, fmt_x):
        fmt, x = fmt_x
        assert fmt.representable_in(x)

    @given(efloat_formats(max_es=4, max_nbits=8, min_eoffset=-8, max_eoffset=8).flatmap(
        lambda fmt: st.tuples(
            st.just(fmt),
            floats(prec_max=8, exp_min=-100, exp_max=100, allow_infinity=False, allow_nan=False, ctx=fp.EFloatContext.from_format(fmt))
        )
    ))
    def test_efloat_representable(self, fmt_x):
        fmt, x = fmt_x
        assert fmt.representable_in(x)

    @given(ieee_formats(max_es=5, max_nbits=16).flatmap(
        lambda fmt: st.tuples(
            st.just(fmt),
            floats(prec_max=8, exp_min=-100, exp_max=100, allow_infinity=False, allow_nan=False, ctx=fp.IEEEContext.from_format(fmt))
        )
    ))
    def test_ieee_representable(self, fmt_x):
        fmt, x = fmt_x
        assert fmt.representable_in(x)


###########################################################
# Tests: OrdinalFormat methods

class TestOrdinalFormat:
    """Test OrdinalFormat methods on format types."""

    @given(
        mps_float_formats(max_p=8, min_emin=-32, max_emin=32).flatmap(
            lambda fmt: st.tuples(
                st.just(fmt),
                floats(prec_max=8, exp_min=-50, exp_max=50, allow_infinity=False, allow_nan=False, ctx=fp.MPSFloatContext.from_format(fmt))
            )
        )
    )
    def test_mps_to_ordinal_from_ordinal_roundtrip(self, fmt_x):
        fmt, x = fmt_x
        ord_val = fmt.to_ordinal(x)
        x2 = fmt.from_ordinal(ord_val)
        assert fmt.to_ordinal(x2) == ord_val

    @given(
        mp_fixed_formats(min_n=-16, max_n=16).flatmap(
            lambda fmt: st.tuples(
                st.just(fmt),
                floats(prec_max=8, exp_min=-50, exp_max=50, allow_infinity=False, allow_nan=False, ctx=fp.MPFixedContext.from_format(fmt))
            )
        )
    )
    def test_mp_fixed_to_fractional_ordinal(self, fmt_x):
        fmt, x = fmt_x
        frac_ord = fmt.to_fractional_ordinal(x)
        assert isinstance(frac_ord, Fraction)

    @given(
        mps_float_formats(max_p=4, min_emin=-8, max_emin=8).flatmap(
            lambda fmt: st.tuples(
                st.just(fmt),
                floats(prec_max=4, exp_min=-20, exp_max=20, allow_infinity=False, allow_nan=False, ctx=fp.MPSFloatContext.from_format(fmt))
            )
        )
    )
    def test_mps_next_up(self, fmt_x):
        fmt, x = fmt_x
        assume(x < fmt.from_ordinal(1000000))
        y = fmt.next_up(x)
        assert isinstance(y, fp.Float)
        # ordinal of y should be one more than ordinal of x
        assert fmt.to_ordinal(y) == fmt.to_ordinal(x) + 1

    @given(
        mps_float_formats(max_p=4, min_emin=-8, max_emin=8).flatmap(
            lambda fmt: st.tuples(
                st.just(fmt),
                floats(prec_max=4, exp_min=-20, exp_max=20, allow_infinity=False, allow_nan=False, ctx=fp.MPSFloatContext.from_format(fmt))
            )
        )
    )
    def test_mps_next_down(self, fmt_x):
        fmt, x = fmt_x
        assume(x > fmt.from_ordinal(-1000000))
        y = fmt.next_down(x)
        assert isinstance(y, fp.Float)
        assert fmt.to_ordinal(y) == fmt.to_ordinal(x) - 1


###########################################################
# Tests: SizedFormat methods

class TestSizedFormat:
    """Test SizedFormat methods on format types."""

    @given(efloat_formats(max_es=4, max_nbits=8, min_eoffset=-8, max_eoffset=8).filter(
        lambda fmt: fp.EFloatContext.from_format(fmt).has_nonzero()
    ))
    def test_efloat_maxval(self, fmt: fp.number.format.EFloatFormat):
        maxval = fmt.maxval()
        assert isinstance(maxval, fp.Float)
        assert not maxval.is_negative()

    @given(ieee_formats(max_es=5, max_nbits=16))
    def test_ieee_largest_smallest(self, fmt: fp.number.format.IEEEFormat):
        largest = fmt.largest()
        smallest = fmt.smallest()
        assert isinstance(largest, fp.Float)
        assert isinstance(smallest, fp.Float)
        assert not largest.is_negative()
        assert not smallest.is_positive()

    @given(fixed_formats(min_scale=-8, max_scale=8, max_nbits=8))
    def test_fixed_maxval(self, fmt: fp.number.format.FixedFormat):
        maxval = fmt.maxval()
        assert isinstance(maxval, fp.Float)
        assert not maxval.is_negative()


###########################################################
# Tests: EncodableFormat methods

class TestEncodableFormat:
    """Test EncodableFormat methods on format types."""

    @given(
        ieee_formats(max_es=5, max_nbits=16).flatmap(
            lambda fmt: st.tuples(
                st.just(fmt),
                floats(prec_max=8, exp_min=-100, exp_max=100, allow_infinity=False, allow_nan=False, ctx=fp.IEEEContext.from_format(fmt))
            )
        )
    )
    def test_ieee_encode_decode_roundtrip(self, fmt_x):
        fmt, x = fmt_x
        encoded = fmt.encode(x)
        assert isinstance(encoded, int)
        assert encoded >= 0
        decoded = fmt.decode(encoded)
        assert x == decoded

    @given(
        fixed_formats(min_scale=-8, max_scale=8, max_nbits=8).flatmap(
            lambda fmt: st.tuples(
                st.just(fmt),
                floats(prec_max=8, exp_min=-50, exp_max=50, allow_infinity=False, allow_nan=False, ctx=fp.FixedContext.from_format(fmt))
            )
        )
    )
    def test_fixed_encode_decode_roundtrip(self, fmt_x):
        fmt, x = fmt_x
        encoded = fmt.encode(x)
        assert isinstance(encoded, int)
        decoded = fmt.decode(encoded)
        assert x == decoded

    @given(exp_formats(max_nbits=8, min_eoffset=-8, max_eoffset=8).flatmap(
        lambda fmt: st.tuples(
            st.just(fmt),
            st.integers(0, (1 << fmt.nbits) - 2)  # representable ordinals (exclude NaN = all ones)
        )
    ))
    def test_exp_encode_decode_roundtrip(self, fmt_i):
        fmt, i = fmt_i
        x = fmt.decode(i)
        assert isinstance(x, fp.Float)
        if not x.isnan:
            encoded = fmt.encode(x)
            assert isinstance(encoded, int)
            decoded = fmt.decode(encoded)
            assert x == decoded

    @given(ieee_formats(max_es=5, max_nbits=16))
    def test_ieee_total_bits(self, fmt: fp.number.format.IEEEFormat):
        assert fmt.total_bits() == fmt.nbits

    @given(fixed_formats(min_scale=-8, max_scale=8, max_nbits=8))
    def test_fixed_total_bits(self, fmt: fp.number.format.FixedFormat):
        assert fmt.total_bits() == fmt.nbits

    @given(exp_formats(max_nbits=8, min_eoffset=-8, max_eoffset=8))
    def test_exp_total_bits(self, fmt: fp.number.format.ExpFormat):
        assert fmt.total_bits() == fmt.nbits


###########################################################
# Tests: specific known-good values

class TestFormatKnownValues:
    """Test format types with known-good values."""

    def test_fp64_format_from_format(self):
        ctx = fp.FP64
        fmt = ctx.format()
        assert isinstance(fmt, fp.number.format.IEEEFormat)
        assert fmt.es == 11
        assert fmt.nbits == 64
        ctx2 = fp.IEEEContext.from_format(fmt)
        assert ctx2.is_equiv(ctx)

    def test_fp32_format_from_format(self):
        ctx = fp.FP32
        fmt = ctx.format()
        assert isinstance(fmt, fp.number.format.IEEEFormat)
        assert fmt.es == 8
        assert fmt.nbits == 32
        ctx2 = fp.IEEEContext.from_format(fmt)
        assert ctx2.is_equiv(ctx)

    def test_sint8_format(self):
        ctx = fp.SINT8
        fmt = ctx.format()
        assert isinstance(fmt, fp.number.format.FixedFormat)
        assert fmt.signed
        assert fmt.nbits == 8

    def test_uint8_format(self):
        ctx = fp.UINT8
        fmt = ctx.format()
        assert isinstance(fmt, fp.number.format.FixedFormat)
        assert not fmt.signed
        assert fmt.nbits == 8

    def test_mx_e8m0_format(self):
        ctx = fp.MX_E8M0
        fmt = ctx.format()
        assert isinstance(fmt, fp.number.format.ExpFormat)
        assert fmt.nbits == 8

    def test_mp_float_format_format_method(self):
        """format() on MPFloatContext returns MPFloatFormat."""
        ctx = fp.MPFloatContext(24)
        fmt = ctx.format()
        assert isinstance(fmt, fp.number.format.MPFloatFormat)
        assert fmt.pmax == 24

    def test_mpfloat_format_is_equiv_context(self):
        """MPFloatFormat equality mirrors MPFloatContext.is_equiv."""
        fmt1 = fp.number.format.MPFloatFormat(53)
        fmt2 = fp.number.format.MPFloatFormat(53)
        assert fmt1 == fmt2
        assert fmt1 != fp.number.format.MPFloatFormat(24)

    def test_mps_format_minval(self):
        """MPSFloatFormat.minval() works correctly."""
        fmt = fp.number.format.MPSFloatFormat(24, -126)
        minv = fmt.minval()
        assert isinstance(minv, fp.Float)
        assert minv.is_positive()
        neg_minv = fmt.minval(s=True)
        assert isinstance(neg_minv, fp.Float)
        assert neg_minv.is_negative()

    def test_mpb_float_format(self):
        """MPBFloatFormat works correctly."""
        ctx = fp.MPBFloatContext(24, -126, fp.RealFloat(exp=128, c=0xffffff))
        fmt = ctx.format()
        assert isinstance(fmt, fp.number.format.MPBFloatFormat)
        assert fmt.pmax == ctx.pmax
        assert fmt.emin == ctx.emin

    def test_efloat_format_encode_decode(self):
        """EFloatFormat encode/decode works correctly."""
        fmt = fp.MX_E4M3.format()
        ctx = fp.MX_E4M3
        v = ctx.round(1.5)
        enc = fmt.encode(v)
        dec = fmt.decode(enc)
        assert v == dec
