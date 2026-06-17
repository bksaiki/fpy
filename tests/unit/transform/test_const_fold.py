"""Unit tests for :class:`fpy2.transform.ConstFold`."""

import fpy2 as fp

from fpy2.ast import (
    BinaryOp,
    BoolVal,
    ContextStmt,
    ForeignVal,
    Integer,
    Rational,
    ReturnStmt,
    Var,
)
from fpy2.number import Context
from fpy2.transform import ConstFold


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


def _outer_context_stmt(fd) -> ContextStmt:
    [ctx_stmt] = [s for s in fd.body.stmts if isinstance(s, ContextStmt)]
    return ctx_stmt


class TestOpFolding:

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
        """Op stays unfolded without an active rounding context."""
        @fp.fpy
        def f():
            return 1.0 + 2.0

        folded = ConstFold.apply(f.ast)
        e = _return_expr(folded)
        assert isinstance(e, BinaryOp), \
            f'op should not fold without ctx; got {type(e).__name__}'

    def test_round_folds(self):
        """``fp.round(1.5)`` under FP64 folds to its rounded value;
        the literal sits at the target context's format."""
        @fp.fpy
        def f():
            with fp.FP64:
                return fp.round(1.5)

        folded = ConstFold.apply(f.ast)
        e = _return_expr(folded)
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
        @fp.fpy
        def f():
            with fp.FP64:
                return (1.0 + 2.0) * 3.0

        folded = ConstFold.apply(f.ast)
        e = _return_expr(folded)
        assert isinstance(e, Integer)
        assert e.val == 9

    def test_fold_through_round(self):
        """Parent op folds *through* its ``Round`` child."""
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
        @fp.fpy
        def f():
            with fp.FP64:
                return 1.0 < 2.0

        folded = ConstFold.apply(f.ast)
        e = _return_expr(folded)
        assert isinstance(e, BoolVal), f'expected BoolVal; got {type(e).__name__}'
        assert e.val is True

    def test_amin_folds_over_literal_list(self):
        @fp.fpy
        def f():
            with fp.FP64:
                return min([3.0, 1.0, 2.0])

        folded = ConstFold.apply(f.ast)
        e = _return_expr(folded)
        assert isinstance(e, Integer), f'expected Integer; got {type(e).__name__}'
        assert e.val == 1

    def test_list_ref_folds(self):
        @fp.fpy
        def f():
            with fp.FP64:
                xs = [10.0, 20.0, 30.0]
                return xs[1]

        folded = ConstFold.apply(f.ast)
        e = _return_expr(folded)
        assert isinstance(e, Integer)
        assert e.val == 20

    def test_list_literal_substituted_at_var(self):
        @fp.fpy
        def f():
            with fp.FP64:
                xs = [1.0, 2.0]
                return min(xs)

        folded = ConstFold.apply(f.ast)
        e = _return_expr(folded)
        assert isinstance(e, Integer)
        assert e.val == 1

    def test_tuple_destructure_folds(self):
        @fp.fpy
        def f():
            with fp.FP64:
                t = (1.0, 2.0)
                a, b = t
                return a + b

        folded = ConstFold.apply(f.ast)
        e = _return_expr(folded)
        assert isinstance(e, Integer)
        assert e.val == 3


class TestContextFolding:
    def test_attribute_context_folded(self):
        @fp.fpy
        def f():
            with fp.FP64:
                return 1.0

        folded = ConstFold.apply(f.ast)
        ctx_stmt = _outer_context_stmt(folded)
        assert isinstance(ctx_stmt.ctx, ForeignVal), \
            f'context should fold to ForeignVal; got {type(ctx_stmt.ctx).__name__}'
        assert isinstance(ctx_stmt.ctx.val, Context)


class TestEnableFlags:
    """``enable_op`` and ``enable_context`` gate the two substitution
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
        @fp.fpy
        def f():
            with fp.FP64:
                return 1.0 + 2.0

        folded = ConstFold.apply(f.ast, enable_op=False)
        ctx_stmt = _outer_context_stmt(folded)
        assert isinstance(ctx_stmt.ctx, ForeignVal)
        assert isinstance(ctx_stmt.ctx.val, Context)


class TestVariableSubstitution:

    def test_local_var(self):
        @fp.fpy
        def f():
            x = 3
            return x

        folded = ConstFold.apply(f.ast)
        e = _return_expr(folded)
        assert isinstance(e, Integer), f'expected Integer; got {type(e).__name__}'
        assert e.val == 3

    def test_foreign_scalar_in_op(self):
        """Free Python ``float`` flows into an op fold."""
        K = 2.0

        @fp.fpy
        def f():
            with fp.FP64:
                return K + 1.0

        folded = ConstFold.apply(f.ast)
        e = _return_expr(folded)
        assert isinstance(e, Integer)
        assert e.val == 3

    def test_foreign_int_var(self):
        K = 7

        @fp.fpy
        def f():
            with fp.FP64:
                return K

        folded = ConstFold.apply(f.ast)
        e = _return_expr(folded)
        assert isinstance(e, Integer)
        assert e.val == 7

    def test_foreign_float_var(self):
        """Python ``float`` → dyadic-rational AST."""
        K = 2.5

        @fp.fpy
        def f():
            with fp.FP64:
                return K

        folded = ConstFold.apply(f.ast)
        e = _return_expr(folded)
        assert isinstance(e, Rational), f'expected Rational; got {type(e).__name__}'
        assert e.p == 5 and e.q == 2

    def test_foreign_context_var(self):
        """Free ``Context`` substituted at the ``with`` block's
        ``Var`` so the body sees a concrete context."""
        CTX = fp.FP32

        @fp.fpy
        def f():
            with CTX:
                return 1.0 + 2.0

        folded = ConstFold.apply(f.ast)
        ctx_stmt = _outer_context_stmt(folded)
        assert isinstance(ctx_stmt.ctx, ForeignVal)
        assert isinstance(ctx_stmt.ctx.val, Context)
        e = _return_expr(folded)
        assert isinstance(e, Integer)
        assert e.val == 3

    def test_enable_op_false_blocks_numeric_var(self):
        """``enable_op=False`` suppresses numeric var-sub but still
        folds contexts."""
        K = 2.0

        @fp.fpy
        def f():
            with fp.FP64:
                return K + 1.0

        folded = ConstFold.apply(f.ast, enable_op=False)
        e = _return_expr(folded)
        assert isinstance(e, BinaryOp), \
            f'op should not fold under enable_op=False; got {type(e).__name__}'
        ctx_stmt = _outer_context_stmt(folded)
        assert isinstance(ctx_stmt.ctx, ForeignVal)


class TestPhiMergeFolding:
    """SCCP-driven folds at phi nodes: ConstFold substitutes uses of
    a phi-defined variable when both incoming branches assign the
    same value."""

    def test_if_else_same_value(self):
        @fp.fpy
        def f(c: bool) -> fp.Real:
            with fp.FP64:
                if c:
                    x = 5.0
                else:
                    x = 5.0
                return x

        folded = ConstFold.apply(f.ast)
        e = _return_expr(folded)
        assert isinstance(e, Integer), f'expected Integer; got {type(e).__name__}'
        assert e.val == 5

    def test_if_else_different_values(self):
        """``Var(x)`` doesn't fold when branches disagree."""
        @fp.fpy
        def f(c: bool) -> fp.Real:
            with fp.FP64:
                if c:
                    x = 5.0
                else:
                    x = 7.0
                return x

        folded = ConstFold.apply(f.ast)
        e = _return_expr(folded)
        assert isinstance(e, Var), f'expected Var; got {type(e).__name__}'

    def test_if1_same_value(self):
        """``If1Stmt``: pre-if def + in-branch def agree."""
        @fp.fpy
        def f(c: bool) -> fp.Real:
            with fp.FP64:
                x = 5.0
                if c:
                    x = 5.0
                return x

        folded = ConstFold.apply(f.ast)
        e = _return_expr(folded)
        assert isinstance(e, Integer)
        assert e.val == 5

    def test_loop_invariant_value_folds(self):
        """Loop where the body re-assigns the same value the variable
        already had — phi fixpoint stabilizes at the value."""
        @fp.fpy
        def f(c: bool) -> fp.Real:
            with fp.FP64:
                x = 5.0
                while c:
                    x = 5.0
                return x

        folded = ConstFold.apply(f.ast)
        e = _return_expr(folded)
        assert isinstance(e, Integer)
        assert e.val == 5

    def test_loop_mutating_does_not_fold(self):
        """Loop where the body mutates the variable — phi goes to
        ``_TOP``, the return stays as ``Var``."""
        @fp.fpy
        def f(n: fp.Real) -> fp.Real:
            with fp.FP64:
                i = 0
                while i < n:
                    i = i + 1
                return i

        folded = ConstFold.apply(f.ast)
        e = _return_expr(folded)
        assert isinstance(e, Var), f'expected Var; got {type(e).__name__}'
