"""Unit tests for :func:`fpy2.strategies.elim_round`.

The transform itself is tested exhaustively in
``tests/unit/transform/test_round_elim.py``; these tests pin the
wrapper's behavior and its composition with ``monomorphize``.
"""


import fpy2 as fp

from fpy2.ast import ContextStmt, ForeignVal
from fpy2.ast.visitor import DefaultVisitor
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
def _prod3(x: fp.Real, y: fp.Real, z: fp.Real) -> fp.Real:
    return (x * y) * z


class TestElimRound:

    def test_after_monomorphize(self):
        # FP32 * FP32 is exact in FP64: the inner multiply hoists under
        # REAL; the outer one (48-bit significand * FP32) must not.
        sched = monomorphize(_prod3, fp.FP64, [RealType(fp.FP32)] * 3)
        out = elim_round(sched)
        assert _count_real_blocks(out.ast) == 1
        for xyz in ((1.5, 2.5, 3.5), (0.1, -0.25, 4.0), (-3.0, 0.0, 1.0)):
            assert sched(*xyz) == out(*xyz)


