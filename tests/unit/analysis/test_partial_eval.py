"""
Unit tests for :class:`fpy2.analysis.PartialEval`.

Scope today: the two recent additions — chained-comparison folding
and best-effort exception handling around the interpreter.  Broader
PE coverage lives in :mod:`tests.infra.analysis.partial_eval` (which
exercises every example + library function).
"""

import fpy2 as fp

from fpy2.analysis import PartialEval
from fpy2.ast import ContextStmt, ReturnStmt


def _return_expr(fd):
    """Drill through ``ContextStmt`` to the innermost return's expr.
    Fixtures here are simple straight-line bodies."""
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
    """``Compare`` nodes fold when every operand is statically known."""

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
        """``-0.0 == 0.0`` is ``True`` under IEEE 754 equality, which
        Python's ``==`` honours — pinning to catch any drift."""
        @fp.fpy
        def f():
            with fp.FP64:
                return -0.0 == 0.0

        info = PartialEval.apply(f.ast)
        assert info.by_expr.get(_return_expr(f.ast)) is True

    def test_no_fold_outside_context(self):
        """Same gate as other ops — without an active context, the
        compare stays unfolded (consistent with ``_visit_*op``)."""
        @fp.fpy
        def f():
            return 1.0 < 2.0

        info = PartialEval.apply(f.ast)
        assert _return_expr(f.ast) not in info.by_expr


class TestListFolding:
    """List literals + indexing + slicing fold when every operand
    is statically known."""

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
        # The interpreter normalizes through its numeric stack;
        # value semantics are what matter.
        assert v is not None and float(v) == 20.0

    def test_list_min_folds(self):
        """``min(xs)`` (AMin reduce-form) folds once the list literal
        is recognized as a value."""
        @fp.fpy
        def f():
            with fp.FP64:
                return min([3.0, 1.0, 2.0])

        info = PartialEval.apply(f.ast)
        v = info.by_expr.get(_return_expr(f.ast))
        assert v is not None and float(v) == 1.0

    def test_list_with_unknown_elt_does_not_fold(self):
        """If any element isn't statically known, the list itself
        isn't a value — and parent ops shouldn't try to fold."""
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                return min([1.0, x, 2.0])

        info = PartialEval.apply(f.ast)
        assert _return_expr(f.ast) not in info.by_expr


class TestRobustness:
    """``PartialEval`` should treat any interpreter exception as
    "not statically foldable" rather than letting it propagate and
    crash the whole analysis."""

    def test_inexact_cast_does_not_crash(self):
        """``fp.cast(1e30)`` under FP32 raises inside the interpreter
        because 1e30 isn't exactly representable.  PE must catch and
        record nothing for that expression."""
        @fp.fpy
        def f():
            with fp.FP32:
                return fp.cast(1e30)

        # No exception should escape.
        info = PartialEval.apply(f.ast)
        # The Cast itself is unfoldable; its argument literal (the
        # Decnum) is still recorded by ``_visit_decnum``.
        assert _return_expr(f.ast) not in info.by_expr

    def test_division_by_zero_does_not_crash(self):
        """``1.0 / 0.0`` under FP64 raises a runtime / overflow error
        through the interpreter; PE catches and drops the binop."""
        @fp.fpy
        def f():
            with fp.FP64:
                return 1.0 / 0.0

        info = PartialEval.apply(f.ast)
        # Whether or not PE records something for this depends on
        # whether the runtime treats 1/0 as Inf (foldable) or raises
        # (unfoldable).  Either is acceptable; the contract is that
        # PE doesn't crash.
        _ = info  # exception escape is the failure mode we're testing
