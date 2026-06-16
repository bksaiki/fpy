"""
Unit tests for :class:`fpy2.transform.ConstFold`.

These tests pin the *current* behaviour of ``ConstFold`` so that
the Phase 3 rewrite (which makes ``ConstFold`` a thin rewriter
on top of :class:`fpy2.analysis.PartialEval`) can be reviewed
diff-by-diff.

What's pinned here:

* Numeric op folding under a concrete rounding context (``with fp.FP64:``).
* Op folding suppressed outside any context.
* ``Round`` / ``Cast`` / ``RoundAt`` preserved as ops (they *are* the
  rounding; folding them would erase the rounding intent).
* ``enable_op=False`` suppresses op folding but still folds the
  ``with``-block's context expression.
* Context-constructor expressions (e.g. ``fp.FP64``) fold to
  ``ForeignVal`` carrying the concrete :class:`Context`.
* Foreign-value substitution into a folded op result.

What's NOT pinned (the refactor introduces these as new capabilities,
covered by separate tests in Phase 3):

* Variable-substitution (``x = 3; return x`` → ``return 3``).
* Compare-op folding (``1.0 < 2.0`` → ``True``).
* Foreign-context substitution via ``Var`` (``CTX = fp.FP32;
  with CTX:`` → folded ContextStmt body).
"""

import fpy2 as fp

from fpy2.ast import (
    Attribute,
    BoolVal,
    BinaryOp,
    ContextStmt,
    ForeignVal,
    Integer,
    Round,
    Cast,
    Decnum,
    Rational,
    ReturnStmt,
    Var,
)
from fpy2.number import Context
from fpy2.transform import ConstFold


def _return_expr(fd):
    """Drill through ``ContextStmt`` to find the innermost ``ReturnStmt``'s
    expression.  The fixtures here are simple straight-line bodies."""
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


def _outer_context_stmt(fd) -> ContextStmt:
    [ctx_stmt] = [s for s in fd.body.stmts if isinstance(s, ContextStmt)]
    return ctx_stmt


# ---------------------------------------------------------------------------
# Op folding
# ---------------------------------------------------------------------------


class TestOpFolding:
    """Numeric op folding behaviour (``enable_op=True``, the default)."""

    def test_add_folds_under_fp64(self):
        @fp.fpy
        def f():
            with fp.FP64:
                return 1.0 + 2.0

        folded = ConstFold.apply(f.ast)
        e = _return_expr(folded)
        assert isinstance(e, Integer), f'expected Integer, got {type(e).__name__}'
        assert e.val == 3

    def test_mul_folds_under_fp64(self):
        @fp.fpy
        def f():
            with fp.FP64:
                return 2.0 * 3.0

        folded = ConstFold.apply(f.ast)
        e = _return_expr(folded)
        assert isinstance(e, Integer)
        assert e.val == 6

    def test_no_fold_outside_context(self):
        """Without an active rounding context the op stays — there's
        no way to round the result soundly."""
        @fp.fpy
        def f():
            return 1.0 + 2.0

        folded = ConstFold.apply(f.ast)
        e = _return_expr(folded)
        assert isinstance(e, BinaryOp), \
            f'op should not fold without ctx; got {type(e).__name__}'

    def test_round_preserved(self):
        """``round`` *is* the rounding operation; folding would erase
        the rounding intent at this AST position."""
        @fp.fpy
        def f():
            with fp.FP64:
                return fp.round(1.5)

        folded = ConstFold.apply(f.ast)
        e = _return_expr(folded)
        assert isinstance(e, Round), f'expected Round preserved, got {type(e).__name__}'

    def test_cast_preserved(self):
        @fp.fpy
        def f():
            with fp.FP64:
                return fp.cast(1.5)

        folded = ConstFold.apply(f.ast)
        e = _return_expr(folded)
        assert isinstance(e, Cast), f'expected Cast preserved, got {type(e).__name__}'

    def test_nested_op_folds(self):
        """``(1 + 2) * 3`` under FP64 → ``9`` (folded bottom-up)."""
        @fp.fpy
        def f():
            with fp.FP64:
                return (1.0 + 2.0) * 3.0

        folded = ConstFold.apply(f.ast)
        e = _return_expr(folded)
        assert isinstance(e, Integer)
        assert e.val == 9


# ---------------------------------------------------------------------------
# Context handling
# ---------------------------------------------------------------------------


class TestContextFolding:
    """``with``-block context expression folding."""

    def test_attribute_context_folded(self):
        """``fp.FP64`` (an ``Attribute``) folds to a ``ForeignVal``
        carrying the concrete :class:`Context`."""
        @fp.fpy
        def f():
            with fp.FP64:
                return 1.0

        folded = ConstFold.apply(f.ast)
        ctx_stmt = _outer_context_stmt(folded)
        assert isinstance(ctx_stmt.ctx, ForeignVal), \
            f'context should fold to ForeignVal; got {type(ctx_stmt.ctx).__name__}'
        assert isinstance(ctx_stmt.ctx.val, Context)


# ---------------------------------------------------------------------------
# Flag semantics
# ---------------------------------------------------------------------------


class TestEnableFlags:
    """``enable_op`` / ``enable_context`` gate the two substitution
    families independently."""

    def test_enable_op_false_keeps_op(self):
        @fp.fpy
        def f():
            with fp.FP64:
                return 1.0 + 2.0

        folded = ConstFold.apply(f.ast, enable_op=False)
        e = _return_expr(folded)
        assert isinstance(e, BinaryOp), \
            f'op should not fold under enable_op=False; got {type(e).__name__}'

    def test_enable_op_false_still_folds_context(self):
        """``with fp.FP64`` still resolves to a concrete context even
        when op folding is disabled — these are independent policies."""
        @fp.fpy
        def f():
            with fp.FP64:
                return 1.0 + 2.0

        folded = ConstFold.apply(f.ast, enable_op=False)
        ctx_stmt = _outer_context_stmt(folded)
        assert isinstance(ctx_stmt.ctx, ForeignVal)
        assert isinstance(ctx_stmt.ctx.val, Context)


# ---------------------------------------------------------------------------
# Foreign-value substitution
# ---------------------------------------------------------------------------


class TestForeignValues:
    """Free Python values referenced from FPy participate in folding."""

    def test_foreign_scalar_in_op(self):
        """A free Python ``float`` flows into an op fold under FP64."""
        K = 2.0

        @fp.fpy
        def f():
            with fp.FP64:
                return K + 1.0

        folded = ConstFold.apply(f.ast)
        e = _return_expr(folded)
        assert isinstance(e, Integer)
        assert e.val == 3
