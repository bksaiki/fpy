"""
Testing `Context` methods.
"""

import fpy2 as fp
import unittest

from fractions import Fraction
from hypothesis import given, strategies as st

from ..generators import *

def _ordinal_contexts():
    return (
        fixed_contexts(min_scale=-8, max_scale=8, max_nbits=8)
        | sm_fixed_contexts(min_scale=-8, max_scale=8, max_nbits=8)
        | mp_fixed_contexts(min_n=-16, max_n=16)
        | mps_float_contexts(max_p=8, min_emin=-64, max_emin=64)
        | efloat_contexts(max_es=4, max_nbits=8)
    )

def _sized_contexts():
    return (
        fixed_contexts(min_scale=-8, max_scale=8, max_nbits=8)
        | sm_fixed_contexts(min_scale=-8, max_scale=8, max_nbits=8)
        | efloat_contexts(max_es=4, max_nbits=8)
    )

def _encodable_contexts():
    return _sized_contexts()


class TestOrdinalContext(unittest.TestCase):
    """Testing `OrdinalContext` methods."""

    @given(common_contexts().filter(lambda ctx: isinstance(ctx, fp.OrdinalContext)))
    def test_minval_common(self, ctx: fp.OrdinalContext):
        pos_min = ctx.minval()
        self.assertIsInstance(pos_min, fp.Float)
        self.assertTrue(pos_min.is_positive())

        try:
            neg_min = ctx.minval(s=True)
            self.assertIsInstance(neg_min, fp.Float)
            self.assertTrue(neg_min.is_negative())
        except ValueError:
            pass

    @given(_ordinal_contexts())
    def test_minval(self, ctx: fp.EncodableContext):
        pos_min = ctx.minval()
        self.assertIsInstance(pos_min, fp.Float)
        self.assertTrue(pos_min.is_positive())

        try:
            neg_min = ctx.minval(s=True)
            self.assertIsInstance(neg_min, fp.Float)
            self.assertTrue(neg_min.is_negative())
        except ValueError:
            pass

    @given(
        common_contexts().filter(
            lambda ctx: isinstance(ctx, fp.OrdinalContext)).flatmap(
            lambda ctx: st.tuples(
                st.just(ctx),
                floats(prec=16, exp_min=-100, exp_max=100, allow_infinity=False, allow_nan=False, ctx=ctx)
            )))
    def test_to_ordinal_common(self, ctx_x: tuple[fp.OrdinalContext, fp.Float]):
        ctx, x = ctx_x
        ord = ctx.to_ordinal(x)
        self.assertIsInstance(ord, int)

    @given(
        _ordinal_contexts().flatmap(
            lambda ctx: st.tuples(
                st.just(ctx),
                floats(prec=16, exp_min=-100, exp_max=100, allow_infinity=False, allow_nan=False, ctx=ctx)
            )))
    def test_to_ordinal(self, ctx_x: tuple[fp.EncodableContext, fp.Float]):
        ctx, x = ctx_x
        ord = ctx.to_ordinal(x)
        self.assertIsInstance(ord, int)


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
        self.assertIsInstance(x, fp.Float)

    @given(
        _ordinal_contexts().filter(lambda ctx: not isinstance(ctx, fp.SizedContext)),
        st.integers(-1000, 1000)
    )
    def test_from_ordinal(self, ctx: fp.EncodableContext, ord: int):
        x = ctx.from_ordinal(ord)
        self.assertIsInstance(x, fp.Float)

    @given(
        common_contexts().filter(
            lambda ctx: isinstance(ctx, fp.OrdinalContext)),
        floats(prec=16, exp_min=-100, exp_max=100, allow_infinity=False, allow_nan=False)
    )
    def test_to_fractional_ordinal_common(self, ctx: fp.OrdinalContext, x: fp.Float):
        frac_ord = ctx.to_fractional_ordinal(x)
        self.assertIsInstance(frac_ord, Fraction)


class TestSizedContext(unittest.TestCase):
    """Testing `SizedContext` methods."""

    @given(
        common_contexts().filter(
            lambda ctx: isinstance(ctx, fp.SizedContext))
    )
    def test_maxval_common(self, ctx: fp.SizedContext):
        pos_max = ctx.maxval()
        self.assertIsInstance(pos_max, fp.Float)
        self.assertFalse(pos_max.is_negative())

        try:
            neg_max = ctx.maxval(s=True)
            self.assertIsInstance(neg_max, fp.Float)
            self.assertFalse(neg_max.is_positive())
        except ValueError: 
            pass

    @given(_sized_contexts())
    def test_maxval(self, ctx: fp.SizedContext):
        pos_max = ctx.maxval()
        self.assertIsInstance(pos_max, fp.Float)
        self.assertFalse(pos_max.is_negative())

        try:
            neg_max = ctx.maxval(s=True)
            self.assertIsInstance(neg_max, fp.Float)
            self.assertFalse(neg_max.is_positive())
        except ValueError: 
            pass


class TestEncodableContext(unittest.TestCase):
    """Testing `EncodableContext` methods."""

    @given(
        common_contexts().filter(
            lambda ctx: isinstance(ctx, fp.EncodableContext)).flatmap(
            lambda ctx: st.tuples(
                st.just(ctx),
                floats(prec=16, exp_min=-100, exp_max=100, allow_infinity=False, allow_nan=False, ctx=ctx)
            )))
    def test_encode_common(self, ctx_x: tuple[fp.EncodableContext, fp.Float]):
        ctx, x = ctx_x
        encoded = ctx.encode(x)
        self.assertIsInstance(encoded, int)
        self.assertTrue(0 <= encoded)

    @given(
        _encodable_contexts().flatmap(
            lambda ctx: st.tuples(
                st.just(ctx),
                floats(prec=16, exp_min=-100, exp_max=100, allow_infinity=False, allow_nan=False, ctx=ctx)
            )))
    def test_encode(self, ctx_x: tuple[fp.EncodableContext, fp.Float]):
        ctx, x = ctx_x
        encoded = ctx.encode(x)
        self.assertIsInstance(encoded, int)
        self.assertTrue(0 <= encoded)


    @given(
        common_contexts().filter(
            lambda ctx: isinstance(ctx, fp.EncodableContext)).flatmap(
            lambda ctx: st.tuples(
                st.just(ctx),
                floats(prec=16, exp_min=-100, exp_max=100, allow_infinity=False, allow_nan=False, ctx=ctx)
            )))
    def test_decode_common(self, ctx_x: tuple[fp.EncodableContext, fp.Float]):
        ctx, x = ctx_x
        encoded = ctx.encode(x)
        decoded = ctx.decode(encoded)
        self.assertIsInstance(decoded, fp.Float)
        self.assertEqual(x, decoded)

    @given(
        _encodable_contexts().flatmap(
            lambda ctx: st.tuples(
                st.just(ctx),
                floats(prec=16, exp_min=-100, exp_max=100, allow_infinity=False, allow_nan=False, ctx=ctx)
            )))
    def test_decode(self, ctx_x: tuple[fp.EncodableContext, fp.Float]):
        ctx, x = ctx_x
        encoded = ctx.encode(x)
        decoded = ctx.decode(encoded)
        self.assertIsInstance(decoded, fp.Float)
        self.assertEqual(x, decoded)
