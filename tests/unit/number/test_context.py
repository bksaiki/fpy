"""
Testing `Context` methods.
"""

import fpy2 as fp

from fractions import Fraction
from hypothesis import assume, given, strategies as st

from ..generators import *

def _ordinal_contexts():
    return (
        fixed_contexts(min_scale=-8, max_scale=8, max_nbits=8)
        | sm_fixed_contexts(min_scale=-8, max_scale=8, max_nbits=8)
        | mp_fixed_contexts(min_n=-16, max_n=16)
        | mps_float_contexts(max_p=8, min_emin=-64, max_emin=64)
        | efloat_contexts(max_es=4, max_nbits=8, min_eoffset=-16, max_eoffset=16)
    )

def _sized_contexts():
    return (
        fixed_contexts(min_scale=-8, max_scale=8, max_nbits=8)
        | sm_fixed_contexts(min_scale=-8, max_scale=8, max_nbits=8)
        | efloat_contexts(max_es=4, max_nbits=8, min_eoffset=-16, max_eoffset=16)
    )

def _encodable_contexts():
    return _sized_contexts()


class TestOrdinalContext():
    """Testing `OrdinalContext` methods."""

    @given(common_contexts().filter(lambda ctx: isinstance(ctx, fp.OrdinalContext)))
    def test_minval_common(self, ctx: fp.OrdinalContext):
        pos_min = ctx.minval()
        assert isinstance(pos_min, fp.Float)
        assert pos_min.is_positive()

        try:
            neg_min = ctx.minval(s=True)
            assert isinstance(neg_min, fp.Float)
            assert neg_min.is_negative()
        except ValueError:
            pass

    @given(_ordinal_contexts().filter(lambda ctx: not isinstance(ctx, fp.EFloatContext) or ctx.has_nonzero()))
    def test_minval(self, ctx: fp.EncodableContext):
        pos_min = ctx.minval()
        assert isinstance(pos_min, fp.Float)
        assert pos_min.is_positive()

        try:
            neg_min = ctx.minval(s=True)
            assert isinstance(neg_min, fp.Float)
            assert neg_min.is_negative()
        except ValueError:
            pass

    @given(
        common_contexts().filter(
            lambda ctx: isinstance(ctx, fp.OrdinalContext)).flatmap(
            lambda ctx: st.tuples(
                st.just(ctx),
                floats(prec_max=16, exp_min=-100, exp_max=100, allow_infinity=False, allow_nan=False, ctx=ctx)
            )))
    def test_to_ordinal_common(self, ctx_x: tuple[fp.OrdinalContext, fp.Float]):
        ctx, x = ctx_x
        ord = ctx.to_ordinal(x)
        assert isinstance(ord, int)

    @given(
        _ordinal_contexts().flatmap(
            lambda ctx: st.tuples(
                st.just(ctx),
                floats(prec_max=16, exp_min=-100, exp_max=100, allow_infinity=False, allow_nan=False, ctx=ctx)
            )))
    def test_to_ordinal(self, ctx_x: tuple[fp.EncodableContext, fp.Float]):
        ctx, x = ctx_x
        ord = ctx.to_ordinal(x)
        assert isinstance(ord, int)


    @given(
        common_contexts().filter(
            lambda ctx: (
                isinstance(ctx, fp.OrdinalContext)
                and not isinstance(ctx, fp.SizedContext)
            )),
        st.integers(-1000, 1000)
    )
    def test_from_ordinal_common(self, ctx: fp.OrdinalContext, ord: int):
        x = ctx.from_ordinal(ord)
        assert isinstance(x, fp.Float)

    @given(
        _ordinal_contexts().filter(lambda ctx: not isinstance(ctx, fp.SizedContext)),
        st.integers(-1000, 1000)
    )
    def test_from_ordinal(self, ctx: fp.EncodableContext, ord: int):
        x = ctx.from_ordinal(ord)
        assert isinstance(x, fp.Float)

    @given(
        common_contexts().filter(
            lambda ctx: isinstance(ctx, fp.OrdinalContext)),
        floats(prec_max=16, exp_min=-100, exp_max=100, allow_infinity=False, allow_nan=False)
    )
    def test_to_fractional_ordinal_common(self, ctx: fp.OrdinalContext, x: fp.Float):
        frac_ord = ctx.to_fractional_ordinal(x)
        assert isinstance(frac_ord, Fraction)

    @given(
        common_contexts().filter(
            lambda ctx: isinstance(ctx, fp.OrdinalContext)).flatmap(
            lambda ctx: st.tuples(
                st.just(ctx),
                floats(prec_max=16, exp_min=-100, exp_max=100, allow_infinity=False, allow_nan=False, ctx=ctx),
                floats(prec_max=16, exp_min=-100, exp_max=100, allow_infinity=False, allow_nan=False, ctx=ctx)
            )))
    def test_next_towards_common(self, ctx_x_y: tuple[fp.OrdinalContext, fp.Float, fp.Float]):
        ctx, x, y = ctx_x_y
        if x != y:
            next_towards = ctx.next_towards(x, y)
            assert isinstance(next_towards, fp.Float)
            assert abs(ctx.to_ordinal(next_towards) - ctx.to_ordinal(x)) == 1

    @given(
        _ordinal_contexts().flatmap(
            lambda ctx: st.tuples(
                st.just(ctx),
                floats(prec_max=16, exp_min=-100, exp_max=100, allow_infinity=False, allow_nan=False, ctx=ctx),
                floats(prec_max=16, exp_min=-100, exp_max=100, allow_infinity=False, allow_nan=False, ctx=ctx)
            )))
    def test_next_towards(self, ctx_x_y: tuple[fp.EncodableContext, fp.Float, fp.Float]):
        ctx, x, y = ctx_x_y
        if x != y:
            next_towards = ctx.next_towards(x, y)
            assert isinstance(next_towards, fp.Float)
            assert abs(ctx.to_ordinal(next_towards) - ctx.to_ordinal(x)) == 1

    @given(
        common_contexts().filter(
            lambda ctx: isinstance(ctx, fp.OrdinalContext)).flatmap(
            lambda ctx: st.tuples(
                st.just(ctx),
                floats(prec_max=16, exp_min=-100, exp_max=100, allow_infinity=False, allow_nan=False, ctx=ctx)
            )))
    def test_next_towards_zero_common(self, ctx_x: tuple[fp.OrdinalContext, fp.Float]):
        ctx, x = ctx_x
        if x != 0:
            y = ctx.next_towards_zero(x)
            assert isinstance(y, fp.Float)
            assert abs(y) < abs(x)
            assert abs(ctx.to_ordinal(y) - ctx.to_ordinal(x)) == 1

    @given(
        _ordinal_contexts().flatmap(
            lambda ctx: st.tuples(
                st.just(ctx),
                floats(prec_max=16, exp_min=-100, exp_max=100, allow_infinity=False, allow_nan=False, ctx=ctx)
            )))
    def test_next_towards_zero(self, ctx_x: tuple[fp.EncodableContext, fp.Float]):
        ctx, x = ctx_x
        if x != 0:
            y = ctx.next_towards_zero(x)
            assert isinstance(y, fp.Float)
            assert abs(y) < abs(x)
            assert abs(ctx.to_ordinal(y) - ctx.to_ordinal(x)) == 1

    @given(
        common_contexts().filter(
            lambda ctx: isinstance(ctx, fp.OrdinalContext)).flatmap(
            lambda ctx: st.tuples(
                st.just(ctx),
                floats(prec_max=16, exp_min=-100, exp_max=100, allow_infinity=False, allow_nan=False, ctx=ctx)
            )))
    def test_next_away_zero_common(self, ctx_x: tuple[fp.OrdinalContext, fp.Float]):
        ctx, x = ctx_x
        assume(not isinstance(ctx, fp.SizedContext) or ctx.smallest() < x < ctx.largest())
        if x != 0:
            y = ctx.next_away_zero(x)
            assert isinstance(y, fp.Float)
            assert abs(y) > abs(x)
            assert abs(ctx.to_ordinal(y) - ctx.to_ordinal(x)) == 1

    @given(
        _ordinal_contexts().flatmap(
            lambda ctx: st.tuples(
                st.just(ctx),
                floats(prec_max=16, exp_min=-100, exp_max=100, allow_infinity=False, allow_nan=False, ctx=ctx)
            )))
    def test_next_away_zero(self, ctx_x: tuple[fp.EncodableContext, fp.Float]):
        ctx, x = ctx_x
        assume(not isinstance(ctx, fp.SizedContext) or ctx.smallest() < x < ctx.largest())
        if x != 0:
            y = ctx.next_away_zero(x)
            assert isinstance(y, fp.Float)
            assert abs(y) > abs(x)
            assert abs(ctx.to_ordinal(y) - ctx.to_ordinal(x)) == 1

    @given(
        common_contexts().filter(
            lambda ctx: isinstance(ctx, fp.OrdinalContext)).flatmap(
            lambda ctx: st.tuples(
                st.just(ctx),
                floats(prec_max=16, exp_min=-100, exp_max=100, allow_infinity=False, allow_nan=False, ctx=ctx)
            )))
    def test_next_up_common(self, ctx_x: tuple[fp.OrdinalContext, fp.Float]):
        ctx, x = ctx_x
        assume(not isinstance(ctx, fp.SizedContext) or x < ctx.largest())
        y = ctx.next_up(x)
        assert isinstance(y, fp.Float)
        assert y >= x
        assert abs(ctx.to_ordinal(y) - ctx.to_ordinal(x)) == 1

    @given(
        _ordinal_contexts().flatmap(
            lambda ctx: st.tuples(
                st.just(ctx),
                floats(prec_max=16, exp_min=-100, exp_max=100, allow_infinity=False, allow_nan=False, ctx=ctx)
            )))
    def test_next_up(self, ctx_x: tuple[fp.EncodableContext, fp.Float]):
        ctx, x = ctx_x
        assume(not isinstance(ctx, fp.SizedContext) or x < ctx.largest())
        y = ctx.next_up(x) 
        assert isinstance(y, fp.Float)
        assert y >= x
        assert abs(ctx.to_ordinal(y) - ctx.to_ordinal(x)) == 1

    @given(
        common_contexts().filter(
            lambda ctx: isinstance(ctx, fp.OrdinalContext)).flatmap(
            lambda ctx: st.tuples(
                st.just(ctx),
                floats(prec_max=16, exp_min=-100, exp_max=100, allow_infinity=False, allow_nan=False, ctx=ctx)
            )))
    def test_next_down_common(self, ctx_x: tuple[fp.OrdinalContext, fp.Float]):
        ctx, x = ctx_x
        assume(not isinstance(ctx, fp.SizedContext) or x > ctx.smallest())
        y = ctx.next_down(x)
        assert isinstance(y, fp.Float)
        assert y <= x
        assert abs(ctx.to_ordinal(y) - ctx.to_ordinal(x)) == 1

    @given(
        _ordinal_contexts().flatmap(
            lambda ctx: st.tuples(
                st.just(ctx),
                floats(prec_max=16, exp_min=-100, exp_max=100, allow_infinity=False, allow_nan=False, ctx=ctx)
            )))
    def test_next_down(self, ctx_x: tuple[fp.EncodableContext, fp.Float]):
        ctx, x = ctx_x
        assume(not isinstance(ctx, fp.SizedContext) or x > ctx.smallest())
        y = ctx.next_down(x)
        assert isinstance(y, fp.Float)
        assert y <= x
        assert abs(ctx.to_ordinal(y) - ctx.to_ordinal(x)) == 1


class TestSizedContext():
    """Testing `SizedContext` methods."""

    @given(
        common_contexts().filter(
            lambda ctx: isinstance(ctx, fp.SizedContext))
    )
    def test_maxval_common(self, ctx: fp.SizedContext):
        pos_max = ctx.maxval()
        assert isinstance(pos_max, fp.Float)
        assert not pos_max.is_negative()

        try:
            neg_max = ctx.maxval(s=True)
            assert isinstance(neg_max, fp.Float)
            assert not neg_max.is_positive()
        except ValueError: 
            pass

    @given(_sized_contexts())
    def test_maxval(self, ctx: fp.SizedContext):
        pos_max = ctx.maxval()
        assert isinstance(pos_max, fp.Float)
        assert not pos_max.is_negative()

        try:
            neg_max = ctx.maxval(s=True)
            assert isinstance(neg_max, fp.Float)
            assert not neg_max.is_positive()
        except ValueError: 
            pass


class TestEncodableContext():
    """Testing `EncodableContext` methods."""

    @given(
        common_contexts().filter(
            lambda ctx: isinstance(ctx, fp.EncodableContext)).flatmap(
            lambda ctx: st.tuples(
                st.just(ctx),
                floats(prec_max=16, exp_min=-100, exp_max=100, allow_infinity=False, allow_nan=False, ctx=ctx)
            )))
    def test_encode_common(self, ctx_x: tuple[fp.EncodableContext, fp.Float]):
        ctx, x = ctx_x
        encoded = ctx.encode(x)
        assert isinstance(encoded, int)
        assert 0 <= encoded

    @given(
        _encodable_contexts().flatmap(
            lambda ctx: st.tuples(
                st.just(ctx),
                floats(prec_max=16, exp_min=-100, exp_max=100, allow_infinity=False, allow_nan=False, ctx=ctx)
            )))
    def test_encode(self, ctx_x: tuple[fp.EncodableContext, fp.Float]):
        ctx, x = ctx_x
        encoded = ctx.encode(x)
        assert isinstance(encoded, int)
        assert 0 <= encoded


    @given(
        common_contexts().filter(
            lambda ctx: isinstance(ctx, fp.EncodableContext)).flatmap(
            lambda ctx: st.tuples(
                st.just(ctx),
                floats(prec_max=16, exp_min=-100, exp_max=100, allow_infinity=False, allow_nan=False, ctx=ctx)
            )))
    def test_decode_common(self, ctx_x: tuple[fp.EncodableContext, fp.Float]):
        ctx, x = ctx_x
        encoded = ctx.encode(x)
        decoded = ctx.decode(encoded)
        assert isinstance(decoded, fp.Float)
        assert x == decoded

    @given(
        _encodable_contexts().flatmap(
            lambda ctx: st.tuples(
                st.just(ctx),
                floats(prec_max=16, exp_min=-100, exp_max=100, allow_infinity=False, allow_nan=False, ctx=ctx)
            )))
    def test_decode(self, ctx_x: tuple[fp.EncodableContext, fp.Float]):
        ctx, x = ctx_x
        encoded = ctx.encode(x)
        decoded = ctx.decode(encoded)
        assert isinstance(decoded, fp.Float)
        assert x == decoded
