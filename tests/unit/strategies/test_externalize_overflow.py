"""Unit tests for :func:`fpy2.strategies.externalize_overflow`.

The transform itself is tested exhaustively in
``tests/unit/transform/test_externalize_overflow.py``; these tests pin the
wrapper's behavior and its composition with the other scheduling operators.
"""

import pytest

import fpy2 as fp

from fpy2.analysis import PartialEval
from fpy2.ast import ContextStmt
from fpy2.ast.visitor import DefaultVisitor
from fpy2.function import Function
from fpy2.number import MPSFloatContext
from fpy2.strategies import (
    externalize_overflow,
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


class TestExternalizeOverflow:

    def test_returns_a_function(self):
        out = externalize_overflow(_quantized_sum)
        assert isinstance(out, Function)
        assert out is not _quantized_sum

    def test_rejects_non_function(self):
        with pytest.raises(TypeError):
            externalize_overflow(_quantized_sum.ast)  # type: ignore[arg-type]

    def test_removes_the_bound_from_the_context(self):
        assert fp.FP16 in _block_ctxs(_quantized_sum.ast)
        out = externalize_overflow(_quantized_sum)
        ctxs = _block_ctxs(out.ast)
        assert fp.FP16 not in ctxs
        assert any(isinstance(c, MPSFloatContext) for c in ctxs)
        # the FP64 accumulation is untouched: its body is arithmetic, not a round
        assert fp.FP64 in ctxs

    @pytest.mark.parametrize('pre_check', [False, True], ids=['post', 'pre_post'])
    def test_preserves_results(self, pre_check):
        out = externalize_overflow(_quantized_sum, pre_check=pre_check)
        assert _same(out(_SAMPLE), _quantized_sum(_SAMPLE))

    def test_idempotent(self):
        """The rewritten program rounds under an unbounded format, which has
        no bound left to take out."""
        once = externalize_overflow(_quantized_sum)
        twice = externalize_overflow(once)
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

        assert fp.FP16 not in _block_ctxs(externalize_overflow(f, 0).ast)
        assert fp.FP32 in _block_ctxs(externalize_overflow(f, 0).ast)
        assert fp.FP16 in _block_ctxs(externalize_overflow(f, 1).ast)
        assert _same(externalize_overflow(f, 0)(0.1, 0.2), f(0.1, 0.2))

    def test_composes_with_simplify(self):
        out = simplify(
            externalize_overflow(_quantized_sum), enable_const_fold_context=False,
        )
        assert _same(out(_SAMPLE), _quantized_sum(_SAMPLE))


class TestPipeline:
    """With the bound out of the context, ``float_to_fixed`` lowers the
    rounding through its unbounded path: the target carries a digit position
    and nothing else."""

    @pytest.mark.parametrize('ctx', [fp.FP16, fp.FP32, fp.IEEEContext(4, 8)],
                             ids=['fp16', 'fp32', 'ieee_4_8'])
    def test_preserves_results(self, ctx):
        @fp.fpy(ctx=fp.REAL)
        def q(x):
            with ctx:
                y = fp.round(x)
            return y

        out = rescale_fixed(float_to_fixed(externalize_overflow(q)))
        B = float(ctx.maxval())
        xs = [
            0.0, -0.0, float('inf'), float('-inf'), float('nan'),
            1.0, -1.0, B, -B, B * 1.001, B * 2,
            2.0 ** ctx.emin, 2.0 ** ctx.expmin, 2.0 ** (ctx.expmin - 1),
            0.1, -0.1, 12.5,
        ]
        for x in xs:
            assert _same(out(x), q(x)), x

    def test_target_carries_no_bound(self):
        """The lowered rounding is unbounded, so nothing downstream has to
        shift a bound or reproduce an overflow."""
        out = float_to_fixed(externalize_overflow(_quantized_sum))
        ctxs = _block_ctxs(out.ast)
        assert not any(isinstance(c, fp.MPBFixedContext) for c in ctxs)
        assert _same(out(_SAMPLE), _quantized_sum(_SAMPLE))
