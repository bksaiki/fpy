"""
`Simplify` — constant folding, copy propagation, dead-code elimination and
context unnesting run to a fixpoint.

What these cover is the *interaction*: each pass has its own tests, and what
is worth pinning here is the work one pass makes available to another. The
cases below are all ones no single pass can do alone.
"""

import fpy2 as fp

from fpy2.ast import ContextStmt
from fpy2.ast.visitor import DefaultVisitor
from fpy2.transform import FuncInline, Simplify, UnnestContext

# spans FP16 subnormals (1e-8), the normal range, and overflow (65600.0)
_VALUES = (1.1, 3.7, 1e-5, 65600.0, 1e-8)


def _nested_pairs(ast) -> int:
    """How many `ContextStmt`s sit directly inside another one's body."""
    count = 0

    class V(DefaultVisitor):
        def _visit_context(self, stmt: ContextStmt, ctx):
            nonlocal count
            count += sum(isinstance(s, ContextStmt) for s in stmt.body.stmts)
            super()._visit_context(stmt, ctx)

    V()._visit_function(ast, None)
    return count


def _agrees(func, ast, values=_VALUES) -> bool:
    rewritten = fp.Function(ast, runtime=func.runtime)
    return all(repr(rewritten(v)) == repr(func(v)) for v in values)


class TestUnnestingInTheLoop:
    """`UnnestContext` is in the fixpoint loop rather than ahead of it,
    because the other three manufacture work for it."""

    def test_it_flattens_what_dead_code_exposes(self):
        """The nested block starts in the *middle* of its parent's body,
        which no unnesting can touch. Removing the dead assignment above it
        moves it to the leading edge."""
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP32:
                dead = x * 2.0
                with fp.FP16:
                    a = x + 1.0
                b = a + 1.0
            return b

        # on its own, the pass correctly declines: the block is in the middle
        assert not UnnestContext.apply_with_status(f.ast)[1]

        out = Simplify.apply(f.ast)
        assert _nested_pairs(out) == 0
        assert _agrees(f, out)

    def test_inlining_leaves_nesting_that_simplify_clears(self):
        """The case from real pipelines: inlining a callee that has its own
        `with` buries it mid-body behind the argument and result copies.
        Copy propagation and dead-code elimination strip those, and only then
        is the block at an edge."""
        @fp.fpy
        def callee(x: fp.Real) -> fp.Real:
            with fp.FP16:
                y = x * 3.0
            return y

        @fp.fpy
        def caller(x: fp.Real) -> fp.Real:
            with fp.FP32:
                a = callee(x)
                b = a + 1.0
            return b

        inlined = FuncInline.apply(caller.ast)
        assert _nested_pairs(inlined) == 1
        assert not UnnestContext.apply_with_status(inlined)[1]

        out = Simplify.apply(inlined)
        assert _nested_pairs(out) == 0
        assert _agrees(caller, out)

    def test_a_second_run_changes_nothing(self):
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP32:
                dead = x * 2.0
                with fp.FP16:
                    a = x + 1.0
                b = a + 1.0
            return b

        once = Simplify.apply(f.ast)
        _twice, changed = Simplify.apply_with_status(once)
        assert not changed


class TestFlags:

    def test_unnesting_can_be_turned_off(self):
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP32:
                dead = x * 2.0
                with fp.FP16:
                    a = x + 1.0
                b = a + 1.0
            return b

        out = Simplify.apply(f.ast, enable_unnest_context=False)
        assert _nested_pairs(out) == 1
        assert _agrees(f, out)

    def test_it_is_on_by_default(self):
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP32:
                a = x + 1.0
                with fp.FP16:
                    b = a * 3.0
            return b

        assert _nested_pairs(Simplify.apply(f.ast)) == 0
