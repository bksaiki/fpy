"""Unit tests for :func:`fpy2.strategies.unfold_overflow`.

The transform itself is tested exhaustively in
``tests/unit/transform/test_unfold_overflow.py``; these tests pin the
wrapper's behavior and its composition with the other scheduling operators.
"""

import pytest

import fpy2 as fp

from fpy2.analysis import PartialEval
from fpy2.ast import ContextStmt
from fpy2.ast.visitor import DefaultVisitor
from fpy2.function import Function
from fpy2.number import (
    EFloatContext,
    EFloatNanKind,
    MPFixedContext,
    MPSFloatContext,
    RealFloat,
)
from fpy2.strategies import (
    unfold_overflow,
    float_to_fixed,
    rescale_fixed,
    simplify,
)


def _block_ctxs(ast) -> list:
    """The value of every statically-known block context in *ast*."""
    eval_info = PartialEval.apply(ast)
    found = []

    class _C(DefaultVisitor):
        def _visit_context(self, stmt: ContextStmt, ctx):
            value = eval_info.by_expr.get(stmt.ctx)
            if value is not None:
                found.append(value)
            super()._visit_context(stmt, ctx)

    _C()._visit_function(ast, None)
    return found


@fp.fpy(ctx=fp.REAL)
def _quantized_sum(A):
    acc = 0
    for a in A:
        with fp.FP16:
            aq = fp.round(a)
        with fp.FP64:
            acc += aq
    return acc


_SAMPLE = [0.1, 0.25, -3.5, 1e-6, 7.0, 0.0, -2.5, 70000.0]


def _same(a, b) -> bool:
    if a.isnan or b.isnan:
        return a.isnan and b.isnan
    if a.isinf or b.isinf:
        return a.isinf and b.isinf and a.s == b.s
    return a.as_rational() == b.as_rational() and a.s == b.s


class TestUnfoldOverflow:

    def test_returns_a_function(self):
        out = unfold_overflow(_quantized_sum)
        assert isinstance(out, Function)
        assert out is not _quantized_sum


    def test_removes_the_bound_from_the_context(self):
        assert fp.FP16 in _block_ctxs(_quantized_sum.ast)
        out = unfold_overflow(_quantized_sum)
        ctxs = _block_ctxs(out.ast)
        assert fp.FP16 not in ctxs
        assert any(isinstance(c, MPSFloatContext) for c in ctxs)
        # the FP64 accumulation is untouched: its body is arithmetic, not a round
        assert fp.FP64 in ctxs

    @pytest.mark.parametrize('early_check', [False, True], ids=['plain', 'early_check'])
    def test_preserves_results(self, early_check):
        out = unfold_overflow(_quantized_sum, early_check=early_check)
        assert _same(out(_SAMPLE), _quantized_sum(_SAMPLE))

    def test_idempotent(self):
        """The rewritten program rounds under an unbounded format, which has
        no bound left to take out."""
        once = unfold_overflow(_quantized_sum)
        twice = unfold_overflow(once)
        assert twice.ast.is_equiv(once.ast)


    def test_composes_with_simplify(self):
        out = simplify(
            unfold_overflow(_quantized_sum), enable_const_fold_context=False,
        )
        assert _same(out(_SAMPLE), _quantized_sum(_SAMPLE))


def _quantizer(ctx) -> fp.Function:
    @fp.fpy(ctx=fp.REAL)
    def q(x):
        with ctx:
            y = fp.round(x)
        return y
    return q


def _samples(ctx) -> list:
    B = ctx.maxval().as_real()
    xs = [
        fp.Float(isnan=True), fp.Float(isinf=True), fp.Float(isinf=True, s=True),
        fp.Float(c=0), fp.Float(c=0, s=True),
    ]
    points = [
        B, ctx.infval().as_real(),
        RealFloat(exp=B.exp - 1, c=2 * B.c - 1),   # just below the bound
        RealFloat(exp=B.exp, c=B.c + 1),           # the tie above it
        RealFloat(exp=B.exp + 40, c=B.c),          # far past
        RealFloat(exp=ctx.expmin, c=1),            # smallest subnormal
        RealFloat(exp=ctx.expmin - 1, c=1),        # below it: rounds to zero
        RealFloat(exp=ctx.emin, c=1), RealFloat(exp=-2, c=1), RealFloat(exp=1, c=3),
    ]
    for g in points:
        xs.append(fp.Float(x=g, ctx=fp.REAL))
        xs.append(fp.Float(x=RealFloat(s=True, exp=g.exp, c=g.c), ctx=fp.REAL))
    return xs


_PIPELINE_CTXS = [
    fp.FP16, fp.FP32, fp.FP64, fp.IEEEContext(4, 8),
    fp.IEEEContext(5, 16, fp.RoundingMode.RNE, fp.OverflowMode.SATURATE),
    fp.IEEEContext(5, 16, fp.RoundingMode.RTZ),
    fp.MX_E5M2, fp.MX_E4M3, fp.MX_E3M2, fp.MX_E2M1,
    EFloatContext(4, 8, False, EFloatNanKind.NEG_ZERO, 0),
    EFloatContext(4, 8, False, EFloatNanKind.NONE, 2),
]
_PIPELINE_IDS = [
    'fp16', 'fp32', 'fp64', 'ieee_4_8', 'ieee_sat', 'ieee_rtz',
    'e5m2', 'e4m3', 'e3m2', 'e2m1', 'neg_zero', 'eoffset',
]


class TestPipeline:
    """``unfold_overflow`` then ``float_to_fixed`` then ``rescale_fixed``
    is the full lowering: a float rounding becomes an integer one, with the
    bound stated as program text rather than carried by a format."""

    @pytest.mark.parametrize('early_check', [False, True], ids=['plain', 'early_check'])
    @pytest.mark.parametrize('ctx', _PIPELINE_CTXS, ids=_PIPELINE_IDS)
    def test_preserves_results(self, ctx, early_check):
        q = _quantizer(ctx)
        out = rescale_fixed(float_to_fixed(unfold_overflow(q, early_check=early_check)))
        for x in _samples(ctx):
            assert _same(out(x), q(x)), (ctx, x)

    @pytest.mark.parametrize('ctx', _PIPELINE_CTXS, ids=_PIPELINE_IDS)
    def test_no_overflow_behavior_survives(self, ctx):
        """No rounding left decides what an overflow *becomes* -- that is what
        this operator moved into program text.

        A bound may still survive, but only as ``OverflowMode.ASSERT``: a claim
        that overflow cannot happen, which `float_to_fixed` states so the
        rounding has a width.  ``OVERFLOW``/``SATURATE``/``WRAP`` would each be
        an edge rule the backend has no way to reproduce.
        """
        out = rescale_fixed(float_to_fixed(unfold_overflow(_quantizer(ctx))))

        ctxs = _block_ctxs(out.ast)
        assert not any(isinstance(c, MPSFloatContext) for c in ctxs)
        # the claim, stated over *every* surviving context rather than the
        # fixed-point ones alone
        for c in ctxs:
            ov = getattr(c, 'overflow', None)
            assert ov in (None, fp.OverflowMode.ASSERT), (c, ov)
        rounding = [c for c in ctxs if isinstance(c, MPFixedContext | fp.MPBFixedContext)]
        assert rounding
        for c in rounding:
            # position zero, so the values are integers
            assert c.nmin == -1
            if isinstance(c, fp.MPBFixedContext):
                assert c.overflow is fp.OverflowMode.ASSERT

    def test_target_states_a_claim_not_a_rule(self):
        """`float_to_fixed` takes its unbounded path, so the bound it states is
        the operand's reach under `ASSERT` rather than the source format's
        overflow rule."""
        alone = float_to_fixed(_quantized_sum)
        composed = float_to_fixed(unfold_overflow(_quantized_sum))

        def overflows(fn):
            return {
                c.overflow for c in _block_ctxs(fn.ast)
                if isinstance(c, fp.MPBFixedContext)
            }

        # run alone, the target reproduces FP16's own edge rule
        assert overflows(alone) == {fp.OverflowMode.OVERFLOW}
        # composed, nothing but the claim
        assert overflows(composed) == {fp.OverflowMode.ASSERT}
        assert _same(composed(_SAMPLE), _quantized_sum(_SAMPLE))

    def test_no_upper_clamp(self):
        """`float_to_fixed` clamps the digit position so the *bound* stays on
        representable.  With no bound there is nothing to keep representable, so
        the position follows the exponent alone."""
        alone = float_to_fixed(_quantizer(fp.FP16))
        composed = float_to_fixed(unfold_overflow(_quantizer(fp.FP16)))
        assert 'min(' in alone.format()
        assert 'min(' not in composed.format()

    def test_early_check_bounds_what_is_rounded(self):
        """With the guard in front, only ``|x| < infval`` reaches the
        rounding, which is what bounds the integer the rescaled round produces
        — the clamp's job, done by the check instead."""
        q = _quantizer(fp.FP16)
        out = rescale_fixed(float_to_fixed(unfold_overflow(q, early_check=True)))
        assert str(int(fp.FP16.infval())) in out.format()
        for x in _samples(fp.FP16):
            assert _same(out(x), q(x)), x

    def test_composes_with_simplify(self):
        out = simplify(
            rescale_fixed(float_to_fixed(unfold_overflow(_quantized_sum))),
            enable_const_fold_context=False,
        )
        assert _same(out(_SAMPLE), _quantized_sum(_SAMPLE))
