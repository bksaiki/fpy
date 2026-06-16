"""
Unit tests for :class:`fpy2.transform.ConstFold`.

These tests pin the *current* behaviour of ``ConstFold`` so that
the Phase 3 rewrite (which makes ``ConstFold`` a thin rewriter
on top of :class:`fpy2.analysis.PartialEval`) can be reviewed
diff-by-diff.

What's pinned here:

* Numeric op folding under a concrete rounding context (``with fp.FP64:``).
* Op folding suppressed outside any context.
* ``Round`` / ``Cast`` / ``RoundAt`` fold like any other op when their
  result is statically known — the substituted literal sits at the
  target context's format.
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
    BinaryOp,
    BoolVal,
    ContextStmt,
    ForeignVal,
    Integer,
    Rational,
    ReturnStmt,
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

    def test_round_folds(self):
        """``fp.round(1.5)`` under FP64 folds to its rounded value.
        The literal sits at the target context's format, so the
        rounding intent is preserved by the value (no Round node
        needs to carry it)."""
        @fp.fpy
        def f():
            with fp.FP64:
                return fp.round(1.5)

        folded = ConstFold.apply(f.ast)
        e = _return_expr(folded)
        # 1.5 is exact in FP64, so the round is the identity at 3/2.
        assert isinstance(e, Rational), f'expected Rational; got {type(e).__name__}'
        assert e.p == 3 and e.q == 2

    def test_cast_folds(self):
        @fp.fpy
        def f():
            with fp.FP64:
                return fp.cast(1.5)

        folded = ConstFold.apply(f.ast)
        e = _return_expr(folded)
        assert isinstance(e, Rational), f'expected Rational; got {type(e).__name__}'
        assert e.p == 3 and e.q == 2

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

    def test_fold_through_round(self):
        """A parent op folds *through* its ``Round`` child — the round
        is evaluated and the whole subtree becomes a literal."""
        @fp.fpy
        def f():
            with fp.FP64:
                x = 1.5
                return fp.round(x) + 1.0

        folded = ConstFold.apply(f.ast)
        e = _return_expr(folded)
        assert isinstance(e, Rational), f'expected Rational; got {type(e).__name__}'
        assert e.p == 5 and e.q == 2

    def test_compare_folds_to_bool(self):
        """``Compare`` nodes fold to a literal ``BoolVal`` once all
        operands resolve under the active context."""
        @fp.fpy
        def f():
            with fp.FP64:
                return 1.0 < 2.0

        folded = ConstFold.apply(f.ast)
        e = _return_expr(folded)
        assert isinstance(e, BoolVal), f'expected BoolVal; got {type(e).__name__}'
        assert e.val is True


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


# ---------------------------------------------------------------------------
# New capabilities introduced by reading from PartialEval
# ---------------------------------------------------------------------------


class TestConstantPropagation:
    """The PartialEval-backed rewriter substitutes known values at
    ``Var`` and foreign-value sites — capabilities the previous
    op-only ``ConstFold`` didn't have."""

    def test_local_var_substituted(self):
        """``x = 3; return x`` → ``return 3``."""
        @fp.fpy
        def f():
            x = 3
            return x

        folded = ConstFold.apply(f.ast)
        e = _return_expr(folded)
        assert isinstance(e, Integer), f'expected Integer; got {type(e).__name__}'
        assert e.val == 3

    def test_foreign_int_var_substituted(self):
        """A free Python ``int`` substitutes at the ``Var`` site."""
        K = 7

        @fp.fpy
        def f():
            with fp.FP64:
                return K

        folded = ConstFold.apply(f.ast)
        e = _return_expr(folded)
        assert isinstance(e, Integer)
        assert e.val == 7

    def test_foreign_float_var_substituted(self):
        """A free Python ``float`` substitutes via the dyadic-rational
        normalization (``2.5`` → ``fp.rational(5, 2)``)."""
        K = 2.5

        @fp.fpy
        def f():
            with fp.FP64:
                return K

        folded = ConstFold.apply(f.ast)
        e = _return_expr(folded)
        assert isinstance(e, Rational), f'expected Rational; got {type(e).__name__}'
        assert e.p == 5 and e.q == 2

    def test_foreign_context_var_folded(self):
        """A free Python :class:`Context` substituted at the ``with``
        block's ``Var`` so the body sees a concrete context."""
        CTX = fp.FP32

        @fp.fpy
        def f():
            with CTX:
                return 1.0 + 2.0

        folded = ConstFold.apply(f.ast)
        ctx_stmt = _outer_context_stmt(folded)
        assert isinstance(ctx_stmt.ctx, ForeignVal)
        assert isinstance(ctx_stmt.ctx.val, Context)
        # And the body gets folded under the concrete context.
        e = _return_expr(folded)
        assert isinstance(e, Integer)
        assert e.val == 3

    def test_enable_op_false_blocks_var_sub_numeric(self):
        """``enable_op=False`` suppresses numeric var-substitution
        (the same gate as op folding) but still folds contexts."""
        K = 2.0

        @fp.fpy
        def f():
            with fp.FP64:
                return K + 1.0

        folded = ConstFold.apply(f.ast, enable_op=False)
        # Op didn't fold, K didn't substitute.
        e = _return_expr(folded)
        assert isinstance(e, BinaryOp), \
            f'op should not fold under enable_op=False; got {type(e).__name__}'
        # But the context expression did fold.
        ctx_stmt = _outer_context_stmt(folded)
        assert isinstance(ctx_stmt.ctx, ForeignVal)
