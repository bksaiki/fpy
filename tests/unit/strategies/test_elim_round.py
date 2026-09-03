"""Unit tests for :func:`fpy2.strategies.elim_round`.

The transform itself is tested exhaustively in
``tests/unit/transform/test_round_elim.py``; these tests pin the
wrapper's behavior and its composition with ``monomorphize``.
"""


import fpy2 as fp

from fpy2.ast import ContextStmt, ForeignVal
from fpy2.ast.visitor import DefaultVisitor
from fpy2.function import Function
from fpy2.number import REAL
from fpy2.strategies import elim_round, monomorphize
from fpy2.types import RealType


def _count_real_blocks(ast) -> int:
    """Number of ``with fp.REAL:`` blocks in *ast*."""
    count = 0

    class _C(DefaultVisitor):
        def _visit_context(self, stmt: ContextStmt, ctx):
            nonlocal count
            if isinstance(stmt.ctx, ForeignVal) and stmt.ctx.val is REAL:
                count += 1
            super()._visit_context(stmt, ctx)

    _C()._visit_function(ast, None)
    return count


@fp.fpy
def _const_add() -> fp.Real:
    with fp.FP64:
        return 1.0 + 2.0


@fp.fpy
def _prod3(x: fp.Real, y: fp.Real, z: fp.Real) -> fp.Real:
    return (x * y) * z


@fp.fpy
def _opaque_mul(x: fp.Real, y: fp.Real) -> fp.Real:
    with fp.FP64:
        # operand formats unknown — nothing is provably exact
        return x * y


class TestElimRound:

    def test_returns_function(self):
        out = elim_round(_const_add)
        assert isinstance(out, Function)
        # the input is not mutated
        assert _count_real_blocks(_const_add.ast) == 0

    def test_exact_add_hoisted(self):
        out = elim_round(_const_add)
        assert _count_real_blocks(out.ast) >= 1
        assert out() == _const_add()

    def test_not_provable_noop(self):
        out = elim_round(_opaque_mul)
        assert out.ast.is_equiv(_opaque_mul.ast)

    def test_after_monomorphize(self):
        # FP32 * FP32 is exact in FP64: the inner multiply hoists under
        # REAL; the outer one (48-bit significand * FP32) must not.
        sched = monomorphize(_prod3, fp.FP64, [RealType(fp.FP32)] * 3)
        out = elim_round(sched)
        assert _count_real_blocks(out.ast) == 1
        for xyz in ((1.5, 2.5, 3.5), (0.1, -0.25, 4.0), (-3.0, 0.0, 1.0)):
            assert sched(*xyz) == out(*xyz)

    def test_without_monomorphize_noop(self):
        # same function, formats unpinned — nothing is provable
        out = elim_round(_prod3)
        assert out.ast.is_equiv(_prod3.ast)

