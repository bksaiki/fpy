"""Unit tests for :func:`fpy2.strategies.rescale_fixed`.

The transform itself is tested exhaustively in
``tests/unit/transform/test_rescale_fixed.py``; these tests pin the
wrapper's behavior and its composition with ``simplify``.
"""

import pytest

import fpy2 as fp

from fpy2.ast import Call, ContextStmt
from fpy2.ast.visitor import DefaultVisitor
from fpy2.function import Function
from fpy2.strategies import rescale_fixed, simplify


def _fixed_scales(ast) -> list[int]:
    """The scale of every ``FixedContext(...)`` block in *ast*."""
    scales: list[int] = []

    class _C(DefaultVisitor):
        def _visit_context(self, stmt: ContextStmt, ctx):
            e = stmt.ctx
            if isinstance(e, Call) and e.fn is fp.FixedContext:
                scales.append(e.args[1].val)
            super()._visit_context(stmt, ctx)

    _C()._visit_function(ast, None)
    return scales


@fp.fpy(ctx=fp.REAL)
def _quantized_sum(A):
    acc = 0
    for a in A:
        with fp.FixedContext(True, -16, 32):
            aq = fp.round(a)
        with fp.FP64:
            acc += aq
    return acc


_SAMPLE = [0.1, 0.25, -3.5, 1e-6, 7.0, 0.0, -2.5]


class TestRescaleFixed:

    def test_returns_a_function(self):
        out = rescale_fixed(_quantized_sum)
        assert isinstance(out, Function)
        assert out is not _quantized_sum

    def test_rejects_non_function(self):
        with pytest.raises(TypeError):
            rescale_fixed(_quantized_sum.ast)  # type: ignore[arg-type]

    def test_moves_the_format_to_zero(self):
        out = rescale_fixed(_quantized_sum)
        assert _fixed_scales(_quantized_sum.ast) == [-16]
        assert _fixed_scales(out.ast) == [0]

    def test_preserves_results(self):
        out = rescale_fixed(_quantized_sum)
        assert out(_SAMPLE) == _quantized_sum(_SAMPLE)

    def test_idempotent(self):
        """A rescaled block is already at zero, so a second run is a no-op."""
        once = rescale_fixed(_quantized_sum)
        twice = rescale_fixed(once)
        assert twice.ast.is_equiv(once.ast)

    def test_composes_with_simplify(self):
        out = simplify(rescale_fixed(_quantized_sum), enable_const_fold_context=False)
        assert _fixed_scales(out.ast) == [0]
        assert out(_SAMPLE) == _quantized_sum(_SAMPLE)
