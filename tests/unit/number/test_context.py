"""
Testing `Context` methods.
"""

import fpy2 as fp
import unittest

from hypothesis import given, strategies as st

from ..generators import *

class TestOrdinalContext(unittest.TestCase):
    """Testing `OrdinalContext` methods."""

    @given(common_contexts())
    def test_minval_common(self, ctx: fp.Context):
        if isinstance(ctx, fp.OrdinalContext):
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
        fixed_contexts(min_scale=-8, max_scale=8, max_nbits=8)
        | mps_float_contexts(max_p=8, min_emin=-64, max_emin=64)
        | efloat_contexts(max_es=4, max_nbits=8)
    )
    def test_minval_sized(self, ctx: fp.EncodableContext):
        pos_min = ctx.minval()
        self.assertIsInstance(pos_min, fp.Float)
        self.assertTrue(pos_min.is_positive())

        neg_min = ctx.minval(s=True)
        self.assertIsInstance(neg_min, fp.Float)
        self.assertTrue(neg_min.is_negative())

