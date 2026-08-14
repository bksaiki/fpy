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

    def test_rejects_non_function(self):
        with pytest.raises(TypeError):
            unfold_overflow(_quantized_sum.ast)  # type: ignore[arg-type]

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

    def test_where_selects_one_block(self):
        """The wrapper passes `where` through to the transform."""

        @fp.fpy(ctx=fp.REAL)
        def f(a, b):
            with fp.FP16:
                aq = fp.round(a)
            with fp.FP32:
                bq = fp.round(b)
            with fp.FP64:
                s = aq + bq
            return s

        assert fp.FP16 not in _block_ctxs(unfold_overflow(f, 0).ast)
        assert fp.FP32 in _block_ctxs(unfold_overflow(f, 0).ast)
        assert fp.FP16 in _block_ctxs(unfold_overflow(f, 1).ast)
        assert _same(unfold_overflow(f, 0)(0.1, 0.2), f(0.1, 0.2))

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
    grid = [
        B, ctx.infval().as_real(),
        RealFloat(exp=B.exp - 1, c=2 * B.c - 1),   # just below the bound
        RealFloat(exp=B.exp, c=B.c + 1),           # the tie above it
        RealFloat(exp=B.exp + 40, c=B.c),          # far past
        RealFloat(exp=ctx.expmin, c=1),            # smallest subnormal
        RealFloat(exp=ctx.expmin - 1, c=1),        # below it: rounds to zero
        RealFloat(exp=ctx.emin, c=1), RealFloat(exp=-2, c=1), RealFloat(exp=1, c=3),
    ]
    for g in grid:
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
    def test_nothing_bounded_survives(self, ctx):
        """Every rounding left states a digit position and no more: no bound
        for `rescale_fixed` to shift, and no overflow rule for a backend to
        reproduce."""
        out = rescale_fixed(float_to_fixed(unfold_overflow(_quantizer(ctx))))

        ctxs = _block_ctxs(out.ast)
        assert not any(hasattr(c, 'maxval') for c in ctxs)
        assert not any(isinstance(c, MPSFloatContext) for c in ctxs)
        positions = [c.nmin for c in ctxs if isinstance(c, MPFixedContext)]
        assert positions and all(n == -1 for n in positions)

    def test_target_carries_no_bound(self):
        """`float_to_fixed` takes its unbounded path, so the target is
        `MPFixedContext` rather than the `MPBFixedContext` it emits alone."""
        out = float_to_fixed(unfold_overflow(_quantized_sum))
        ctxs = _block_ctxs(out.ast)
        assert not any(isinstance(c, fp.MPBFixedContext) for c in ctxs)
        assert any(isinstance(c, MPFixedContext) for c in ctxs)
        assert _same(out(_SAMPLE), _quantized_sum(_SAMPLE))

    def test_no_upper_clamp(self):
        """`float_to_fixed` clamps the digit position so the *bound* stays on
        the format's grid.  With no bound there is nothing to keep on it, so
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
