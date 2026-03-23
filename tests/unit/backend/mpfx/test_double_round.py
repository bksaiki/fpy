"""
Double rounding tests for `A(p, exp, bound)`.
"""

import fpy2 as fp
import unittest

from hypothesis import assume, given, strategies as st

from fpy2.backend.mpfx import AbstractFormat

from ...generators import real_floats

def _cvt_to_context(p, exp, b):
    if p == float('inf'):
        return fp.MPBFixedContext(nmin=exp-1, maxval=b, enable_inf=True, overflow=fp.OverflowMode.OVERFLOW)
    else:
        emin = exp + p - 1
        return fp.MPBFloatContext(pmax=p, emin=emin, maxval=b)

def _next_float(ctx, x: fp.RealFloat):
    """Returns the value just above maxval in ordinal space."""
    if isinstance(ctx, fp.MPBFixedContext):
        o = ctx._mp_ctx.to_ordinal(fp.Float.from_real(x))
        return ctx._mp_ctx.from_ordinal(o + 1).as_real()
    elif isinstance(ctx, fp.MPBFloatContext):
        o = ctx._mps_ctx.to_ordinal(fp.Float.from_real(x))
        return ctx._mps_ctx.from_ordinal(o + 1).as_real()
    else:
        raise TypeError(f'unexpected context type: {type(ctx)}')

def _make_contexts(p1, exp1, k1, dp, dexp, dk, rm1, rm2, ensure_odd=False):
    p2 = p1 + dp
    exp2 = exp1 - dexp

    b1_raw = fp.RealFloat(exp=exp1, c=k1)
    if p1 == float('inf'):
        b1 = b1_raw.round(min_n=exp1 - 1, rm=fp.RM.RTZ)
    else:
        b1 = b1_raw.round(p1, exp1 - 1, rm=fp.RM.RTZ)

    b2_raw = fp.RealFloat(exp=exp1, c=k1 + dk)
    if p2 == float('inf'):
        b2 = b2_raw.round(min_n=exp1 - 1, rm=fp.RM.RTZ)
    else:
        b2 = b2_raw.round(p2, exp2 - 1, rm=fp.RM.RTZ)

    ctx1 = _cvt_to_context(p1, exp1, b1).with_params(rm=rm1)
    ctx2 = _cvt_to_context(p2, exp2, b2).with_params(rm=rm2)

    if ensure_odd:
        # if ctx2's maxval is even in ordinal space, advance it to the next float
        # so that it is odd, making RTO tie-breaking consistent with RNE in ctx1
        pos_next = _next_float(ctx2, ctx2.pos_maxval)
        if ctx2.pos_maxval.c % 2 == 0:
            ctx2 = ctx2.with_params(
                maxval=pos_next,
                neg_maxval=-pos_next
            )

    return ctx1, ctx2

class TestDoubleRound(unittest.TestCase):
    """Testing double rounding with overflow."""

    @given(
        st.one_of(st.integers(1, 5), st.just(float('inf'))),  # p1: precision (inf = fixed-point)
        st.integers(-8, 0),                                   # exp1: minimum exponent of fmt1
        st.integers(1, 32),                                   # k1: bound of fmt1 is k1 * 2^exp1
        st.integers(0, 5),                                    # dp: p2 = p1 + dp
        st.integers(0, 5),                                    # dexp: exp2 = exp1 - dexp
        st.integers(0, 8),                                    # dk: b2 = (k1 + dk) * 2^exp1 (rounded to fit fmt2)
        real_floats(prec_max=16, exp_min=-16, exp_max=16),    # x: a real float to round
    )
    def test_rtz_rtz(self, p1, exp1, k1, dp, dexp, dk, x: fp.RealFloat):
        """ctx1[RTZ](ctx2[RTZ](x)) == ctx1[RTZ](x)"""
        ctx1, ctx2 = _make_contexts(p1, exp1, k1, dp, dexp, dk, fp.RM.RTZ, fp.RM.RTZ)
        A1 = AbstractFormat.from_context(ctx1)
        A2 = AbstractFormat.from_context(ctx2)
        y1 = ctx1.round(x)
        y2 = ctx2.round(x)
        y3 = ctx1.round(y2)
        self.assertLessEqual(A1, A2, f"Failed: A1={A1}, A2={A2}, x={float(x)}")
        self.assertEqual(y1, y3, f"Failed: y1={float(y1)}, y2={float(y2)}, y3={float(y3)}, x={float(x)}")

    @given(
        st.one_of(st.integers(1, 5), st.just(float('inf'))),  # p1: precision (inf = fixed-point)
        st.integers(-8, 0),                                   # exp1: minimum exponent of fmt1
        st.integers(1, 32),                                   # k1: bound of fmt1 is k1 * 2^exp1
        st.integers(0, 5),                                    # dp: p2 = p1 + dp
        st.integers(0, 5),                                    # dexp: exp2 = exp1 - dexp
        st.integers(0, 8),                                    # dk: b2 = (k1 + dk) * 2^exp1 (rounded to fit fmt2)
        real_floats(prec_max=16, exp_min=-16, exp_max=16),    # x: a real float to round
    )
    def test_raz_raz(self, p1, exp1, k1, dp, dexp, dk, x: fp.RealFloat):
        """ctx1[RAZ](ctx2[RAZ](x)) == ctx1[RAZ](x)"""
        ctx1, ctx2 = _make_contexts(p1, exp1, k1, dp, dexp, dk, fp.RM.RAZ, fp.RM.RAZ)
        A1 = AbstractFormat.from_context(ctx1)
        A2 = AbstractFormat.from_context(ctx2)
        y1 = ctx1.round(x)
        y2 = ctx2.round(x)
        y3 = ctx1.round(y2)
        self.assertLessEqual(A1, A2, f"Failed: A1={A1}, A2={A2}, x={float(x)}")
        self.assertEqual(y1, y3, f"Failed: y1={float(y1)}, y2={float(y2)}, y3={float(y3)}, x={float(x)}")

    @given(
        st.one_of(st.integers(1, 5), st.just(float('inf'))),  # p1: precision (inf = fixed-point)
        st.integers(-8, 0),                                   # exp1: minimum exponent of fmt1
        st.integers(1, 32),                                   # k1: bound of fmt1 is k1 * 2^exp1
        st.integers(0, 5),                                    # dp: p2 = p1 + dp
        st.integers(0, 5),                                    # dexp: exp2 = exp1 - dexp
        st.integers(0, 8),                                    # dk: b2 = (k1 + dk) * 2^exp1 (rounded to fit fmt2)
        real_floats(prec_max=16, exp_min=-16, exp_max=16),    # x: a real float to round
    )
    def test_rto_rto(self, p1, exp1, k1, dp, dexp, dk, x: fp.RealFloat):
        """ctx1[RTO](ctx2[RTO](x)) == ctx1[RTO](x)"""
        ctx1, ctx2 = _make_contexts(p1, exp1, k1, dp, dexp, dk, fp.RM.RTO, fp.RM.RTO)
        A1 = AbstractFormat.from_context(ctx1)
        A2 = AbstractFormat.from_context(ctx2)
        y1 = ctx1.round(x)
        y2 = ctx2.round(x)
        y3 = ctx1.round(y2)
        self.assertLessEqual(A1, A2, f"Failed: A1={A1}, A2={A2}, x={float(x)}")
        self.assertEqual(y1, y3, f"Failed: y1={float(y1)}, y2={float(y2)}, y3={float(y3)}, x={float(x)}")

    # TODO: restore this test with proper preconditions
    # the problem is that RTO overflows to infinity while RTZ does not
    # @given(
    #     st.one_of(st.integers(1, 5), st.just(float('inf'))),  # p1: precision (inf = fixed-point)
    #     st.integers(-8, 0),                                   # exp1: minimum exponent of fmt1
    #     st.integers(1, 32),                                   # k1: bound of fmt1 is k1 * 2^exp1
    #     st.integers(1, 5),                                    # dp: p2 = p1 + dp
    #     st.integers(1, 5),                                    # dexp: exp2 = exp1 - dexp
    #     st.integers(0, 8),                                    # dk: b2 = (k1 + dk) * 2^exp1 (rounded to fit fmt2)
    #     real_floats(prec_max=16, exp_min=-16, exp_max=16),    # x: a real float to round
    # )
    # def test_rto_rtz(self, p1, exp1, k1, dp, dexp, dk, x: fp.RealFloat):
    #     """ctx1[RTZ](ctx2[RTO](x)) == ctx1[RTZ](x)"""
    #     ctx1, ctx2 = _make_contexts(p1, exp1, k1, dp, dexp, dk, fp.RM.RTZ, fp.RM.RTO)
    #     A1 = AbstractFormat.from_context(ctx1)
    #     A2 = AbstractFormat.from_context(ctx2)
    #     y1 = ctx1.round(x)
    #     y2 = ctx2.round(x)
    #     y3 = ctx1.round(y2)
    #     self.assertLessEqual(A1, A2, f"Failed: A1={A1}, A2={A2}, x={float(x)}")
    #     self.assertEqual(y1, y3, f"Failed: y1={float(y1)}, y2={float(y2)}, y3={float(y3)}, x={float(x)}")

    @given(
        st.one_of(st.integers(1, 5), st.just(float('inf'))),  # p1: precision (inf = fixed-point)
        st.integers(-8, 0),                                   # exp1: minimum exponent of fmt1
        st.integers(1, 32),                                   # k1: bound of fmt1 is k1 * 2^exp1
        st.integers(1, 5),                                    # dp: p2 = p1 + dp
        st.integers(1, 5),                                    # dexp: exp2 = exp1 - dexp
        st.integers(0, 8),                                    # dk: b2 = (k1 + dk) * 2^exp1 (rounded to fit fmt2)
        real_floats(prec_max=16, exp_min=-16, exp_max=16),    # x: a real float to round
    )
    def test_rto_raz(self, p1, exp1, k1, dp, dexp, dk, x: fp.RealFloat):
        """ctx1[RAZ](ctx2[RTO](x)) == ctx1[RAZ](x)"""
        ctx1, ctx2 = _make_contexts(p1, exp1, k1, dp, dexp, dk, fp.RM.RAZ, fp.RM.RTO)
        assume(ctx1.pos_maxval < ctx2.pos_maxval)
        A1 = AbstractFormat.from_context(ctx1)
        A2 = AbstractFormat.from_context(ctx2)
        y1 = ctx1.round(x)
        y2 = ctx2.round(x)
        y3 = ctx1.round(y2)
        self.assertLessEqual(A1, A2, f"Failed: A1={A1}, A2={A2}, x={float(x)}")
        self.assertEqual(y1, y3, f"Failed: y1={float(y1)}, y2={float(y2)}, y3={float(y3)}, x={float(x)}")

    @given(
        st.one_of(st.integers(2, 5)),                         # p1: precision (inf = fixed-point)
        st.integers(-8, 0),                                   # exp1: minimum exponent of fmt1
        st.integers(1, 32),                                   # k1: bound of fmt1 is k1 * 2^exp1
        st.integers(2, 5),                                    # dp: p2 = p1 + dp
        st.integers(2, 5),                                    # dexp: exp2 = exp1 - dexp
        st.integers(0, 8),                                    # dk: b2 = (k1 + dk) * 2^exp1 (rounded to fit fmt2)
        real_floats(prec_max=16, exp_min=-16, exp_max=16),    # x: a real float to round
    )
    def test_rto_rne(self, p1, exp1, k1, dp, dexp, dk, x: fp.RealFloat):
        """ctx1[RNE](ctx2[RTO](x)) == ctx1[RNE](x)"""
        ctx1, ctx2 = _make_contexts(p1, exp1, k1, dp, dexp, dk, fp.RM.RNE, fp.RM.RTO)
        assume(_next_float(ctx1, ctx1.pos_maxval) <= ctx2.pos_maxval)
        A1 = AbstractFormat.from_context(ctx1)
        A2 = AbstractFormat.from_context(ctx2)
        y1 = ctx1.round(x)
        y2 = ctx2.round(x)
        y3 = ctx1.round(y2)
        self.assertLessEqual(A1, A2, f"Failed: A1={A1}, A2={A2}, x={float(x)}")
        self.assertEqual(y1, y3, f"Failed: y1={float(y1)}, y2={float(y2)}, y3={float(y3)}, x={float(x)}")
