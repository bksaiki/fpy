"""Unit tests for :func:`fpy2.strategies.float_to_fixed`.

The transform itself is tested exhaustively in
``tests/unit/transform/test_float_to_fixed.py``; these tests pin the wrapper's
behavior and its composition with the other scheduling operators.
"""

import pytest

import fpy2 as fp

from fpy2.analysis import PartialEval
from fpy2.ast import Call, ContextStmt
from fpy2.ast.visitor import DefaultVisitor
from fpy2.function import Function
from fpy2.strategies import float_to_fixed, rescale_fixed, simplify


def _blocks(ast) -> list:
    """Every ``ContextStmt`` in *ast*."""
    found = []

    class _C(DefaultVisitor):
        def _visit_context(self, stmt: ContextStmt, ctx):
            found.append(stmt)
            super()._visit_context(stmt, ctx)

    _C()._visit_function(ast, None)
    return found


def _block_ctxs(ast) -> list:
    """The value of every statically-known block context in *ast*, however
    the context is written."""
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


class TestFloatToFixed:

    def test_returns_a_function(self):
        out = float_to_fixed(_quantized_sum)
        assert isinstance(out, Function)
        assert out is not _quantized_sum

    def test_rejects_non_function(self):
        with pytest.raises(TypeError):
            float_to_fixed(_quantized_sum.ast)  # type: ignore[arg-type]

    def test_removes_the_float_rounding(self):
        assert fp.FP16 in _block_ctxs(_quantized_sum.ast)
        out = float_to_fixed(_quantized_sum)
        assert fp.FP16 not in _block_ctxs(out.ast)
        # the FP64 accumulation is untouched: its body is arithmetic, not a round
        assert fp.FP64 in _block_ctxs(out.ast)

    def test_preserves_results(self):
        out = float_to_fixed(_quantized_sum)
        assert _same(out(_SAMPLE), _quantized_sum(_SAMPLE))

    def test_idempotent(self):
        """The lowered program has no float rounding left to lower."""
        once = float_to_fixed(_quantized_sum)
        twice = float_to_fixed(once)
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

        assert fp.FP16 not in _block_ctxs(float_to_fixed(f, where=0).ast)
        assert fp.FP32 in _block_ctxs(float_to_fixed(f, where=0).ast)
        assert fp.FP16 in _block_ctxs(float_to_fixed(f, where=1).ast)
        assert _same(float_to_fixed(f, where=0)(0.1, 0.2), f(0.1, 0.2))

    def test_composes_with_simplify(self):
        out = simplify(float_to_fixed(_quantized_sum), enable_const_fold_context=False)
        assert _same(out(_SAMPLE), _quantized_sum(_SAMPLE))


class TestPipeline:
    """``float_to_fixed`` then ``rescale_fixed`` is the full lowering: float
    rounding becomes integer rounding, at a scale that is constant where the
    format is and computed where it is not."""

    @pytest.mark.parametrize('ctx', [fp.FP16, fp.FP32, fp.IEEEContext(4, 8)],
                             ids=['fp16', 'fp32', 'ieee_4_8'])
    def test_preserves_results(self, ctx):
        @fp.fpy(ctx=fp.REAL)
        def q(x):
            with ctx:
                y = fp.round(x)
            return y

        out = rescale_fixed(float_to_fixed(q))
        B = float(ctx.maxval())
        xs = [
            0.0, -0.0, float('inf'), float('-inf'), float('nan'),
            1.0, -1.0, B, -B, B * 1.001, B * 2,
            2.0 ** ctx.emin, 2.0 ** ctx.expmin, 2.0 ** (ctx.expmin - 1),
            0.1, -0.1, 12.5,
        ]
        for x in xs:
            assert _same(out(x), q(x)), x

    def test_every_rounding_lands_at_position_zero(self):
        """Nothing rounds under a float format afterwards, and every
        fixed-point block sits at position zero."""
        out = rescale_fixed(float_to_fixed(_quantized_sum))

        positions = []
        for stmt in _blocks(out.ast):
            e = stmt.ctx
            if isinstance(e, Call) and e.fn is fp.MPBFixedContext:
                positions.append(e.args[0].val)
        assert positions and all(p == -1 for p in positions)
        assert fp.FP16 not in _block_ctxs(out.ast)
