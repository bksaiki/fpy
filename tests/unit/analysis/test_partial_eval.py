"""Unit tests for :class:`fpy2.analysis.PartialEval`.

Broader coverage (every example + library function) lives in
:mod:`tests.infra.analysis.partial_eval`.
"""

import fpy2 as fp

from fpy2.analysis import PartialEval
from fpy2.ast import ContextStmt, ReturnStmt


def _return_expr(fd):
    """Innermost ``ReturnStmt`` expr; drills through ``ContextStmt``."""
    block = fd.body
    while True:
        last = block.stmts[-1]
        if isinstance(last, ReturnStmt):
            return last.expr
        if isinstance(last, ContextStmt):
            block = last.body
            continue
        raise AssertionError(
            f'unexpected statement kind in test fixture: {type(last).__name__}'
        )


class TestCompareFolding:
    def test_lt_true(self):
        @fp.fpy
        def f():
            with fp.FP64:
                return 1.0 < 2.0

        info = PartialEval.apply(f.ast)
        assert info.by_expr.get(_return_expr(f.ast)) is True

    def test_lt_false(self):
        @fp.fpy
        def f():
            with fp.FP64:
                return 3.0 < 2.0

        info = PartialEval.apply(f.ast)
        assert info.by_expr.get(_return_expr(f.ast)) is False

    def test_chained_lt_true(self):
        @fp.fpy
        def f():
            with fp.FP64:
                return 0.0 < 1.0 < 2.0

        info = PartialEval.apply(f.ast)
        assert info.by_expr.get(_return_expr(f.ast)) is True

    def test_chained_lt_false_at_tail(self):
        """``0 < 1 < 1`` — chain fails at the last comparator."""
        @fp.fpy
        def f():
            with fp.FP64:
                return 0.0 < 1.0 < 1.0

        info = PartialEval.apply(f.ast)
        assert info.by_expr.get(_return_expr(f.ast)) is False

    def test_eq_with_signed_zero(self):
        """IEEE 754: ``-0.0 == 0.0`` is ``True``."""
        @fp.fpy
        def f():
            with fp.FP64:
                return -0.0 == 0.0

        info = PartialEval.apply(f.ast)
        assert info.by_expr.get(_return_expr(f.ast)) is True

    def test_no_fold_outside_context(self):
        @fp.fpy
        def f():
            return 1.0 < 2.0

        info = PartialEval.apply(f.ast)
        assert _return_expr(f.ast) not in info.by_expr


class TestListFolding:
    def test_list_literal_value(self):
        @fp.fpy
        def f():
            with fp.FP64:
                return [1.0, 2.0, 3.0]

        info = PartialEval.apply(f.ast)
        v = info.by_expr.get(_return_expr(f.ast))
        assert isinstance(v, list)
        assert len(v) == 3

    def test_list_ref_folds(self):
        @fp.fpy
        def f():
            with fp.FP64:
                return [10.0, 20.0, 30.0][1]

        info = PartialEval.apply(f.ast)
        v = info.by_expr.get(_return_expr(f.ast))
        assert v is not None and float(v) == 20.0

    def test_list_min_folds(self):
        @fp.fpy
        def f():
            with fp.FP64:
                return min([3.0, 1.0, 2.0])

        info = PartialEval.apply(f.ast)
        v = info.by_expr.get(_return_expr(f.ast))
        assert v is not None and float(v) == 1.0

    def test_list_with_unknown_elt_does_not_fold(self):
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                return min([1.0, x, 2.0])

        info = PartialEval.apply(f.ast)
        assert _return_expr(f.ast) not in info.by_expr


class TestIfExprFolding:
    """``IfExpr`` folds to the chosen branch's value when the
    condition is statically known."""

    def test_cond_true_picks_ift(self):
        @fp.fpy
        def f():
            with fp.FP64:
                return 1.0 if True else 2.0

        info = PartialEval.apply(f.ast)
        v = info.by_expr.get(_return_expr(f.ast))
        assert v is not None and float(v) == 1.0

    def test_cond_false_picks_iff(self):
        @fp.fpy
        def f():
            with fp.FP64:
                return 1.0 if False else 2.0

        info = PartialEval.apply(f.ast)
        v = info.by_expr.get(_return_expr(f.ast))
        assert v is not None and float(v) == 2.0

    def test_cond_unknown_does_not_fold(self):
        @fp.fpy
        def f(c: bool, x: fp.Real) -> fp.Real:
            with fp.FP64:
                return 1.0 if c else x

        info = PartialEval.apply(f.ast)
        assert _return_expr(f.ast) not in info.by_expr

    def test_taken_branch_unknown_does_not_fold(self):
        """Cond is known True but the chosen branch (ift) isn't a
        value — no fold."""
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                return x if True else 2.0

        info = PartialEval.apply(f.ast)
        assert _return_expr(f.ast) not in info.by_expr

    def test_unchosen_branch_unknown_still_folds(self):
        """When cond is True, the iff branch's value is irrelevant —
        fold still proceeds based on ift's value."""
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                return 7.0 if True else x

        info = PartialEval.apply(f.ast)
        v = info.by_expr.get(_return_expr(f.ast))
        assert v is not None and float(v) == 7.0


class TestRobustness:
    """Interpreter exceptions must be swallowed: PE is best-effort."""

    def test_inexact_cast_does_not_crash(self):
        """``fp.cast(1e30)`` under FP32 raises inside the interpreter."""
        @fp.fpy
        def f():
            with fp.FP32:
                return fp.cast(1e30)

        info = PartialEval.apply(f.ast)
        assert _return_expr(f.ast) not in info.by_expr

    def test_division_by_zero_does_not_crash(self):
        """``1.0 / 0.0`` raises through the interpreter."""
        @fp.fpy
        def f():
            with fp.FP64:
                return 1.0 / 0.0

        PartialEval.apply(f.ast)
