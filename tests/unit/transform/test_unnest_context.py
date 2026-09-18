"""
`UnnestContext` — hoisting a `with` out of the end of another one's body.

Shape tests say the rewrite fired where it should and not where it should not.
The differential tests are what catch a rewrite that fired *wrongly*: values
are chosen to straddle FP16 subnormals and overflow, so swapping two contexts
changes the answer instead of being masked by FP64's range.
"""

import fpy2 as fp

from fpy2.ast import ContextStmt
from fpy2.ast.visitor import DefaultVisitor
from fpy2.transform import DeadCodeEliminate, UnnestContext

# spans FP16 subnormals (1e-8), the normal range, and overflow (65600.0)
_VALUES = (1.1, 3.7, 1e-5, 65600.0, 1e-8)


def _text(func, ast) -> str:
    return ' '.join(fp.Function(ast, runtime=func.runtime).format().split())


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
    """The rewritten function computes what the original computes."""
    rewritten = fp.Function(ast, runtime=func.runtime)
    return all(repr(rewritten(v)) == repr(func(v)) for v in values)


class TestTrailing:

    def test_the_shape_it_states(self):
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP32:
                a = x + 1.0
                with fp.FP16:
                    b = a * 3.0
            return b

        out = UnnestContext.apply(f.ast)
        assert _text(f, out) == (
            '@fp.fpy def f(x): with fp.FP32: a = (x + 1) '
            'with fp.FP16: b = (a * 3) return b'
        )
        assert _nested_pairs(out) == 0
        assert _agrees(f, out)

    def test_it_peels_a_run_of_them(self):
        """Several trailing blocks come out in one traversal, in order."""
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP32:
                a = x + 1.0
                with fp.FP16:
                    b = a * 3.0
                with fp.FP64:
                    c = b * 3.0
            return c

        out = UnnestContext.apply(f.ast)
        assert _nested_pairs(out) == 0
        src = _text(f, out)
        assert src.index('fp.FP32') < src.index('fp.FP16') < src.index('fp.FP64')
        assert _agrees(f, out)

    def test_it_flattens_bottom_up(self):
        """A block nested two deep comes all the way out."""
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP32:
                a = x + 1.0
                with fp.FP16:
                    b = a * 3.0
                    with fp.FP64:
                        c = b * 3.0
            return c

        out = UnnestContext.apply(f.ast)
        assert _nested_pairs(out) == 0
        assert _agrees(f, out)

    def test_the_context_count_is_unchanged(self):
        """The rewrite trades nesting for sibling order; it never duplicates
        or drops a `with`."""
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP32:
                a = x + 1.0
                with fp.FP16:
                    b = a * 3.0
            return b

        def count(ast):
            n = 0

            class V(DefaultVisitor):
                def _visit_context(self, stmt, ctx):
                    nonlocal n
                    n += 1
                    super()._visit_context(stmt, ctx)

            V()._visit_function(ast, None)
            return n

        assert count(UnnestContext.apply(f.ast)) == count(f.ast) == 2


class TestRefusals:

    def test_a_nested_block_in_the_middle_is_left_alone(self):
        """Splitting here would need two copies of the outer `with`, which
        grows the program."""
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP32:
                a = x + 1.0
                with fp.FP16:
                    b = a * 3.0
                c = b + 1.0
            return c

        out, changed = UnnestContext.apply_with_status(f.ast)
        assert not changed
        assert _nested_pairs(out) == 1

    def test_a_sole_nested_block_is_left_to_dead_code(self):
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP32:
                with fp.FP16:
                    b = x * 3.0
            return b

        out, changed = UnnestContext.apply_with_status(f.ast)
        assert not changed
        assert _nested_pairs(out) == 1
        # ...which `DeadCodeEliminate` then drops outright
        assert _nested_pairs(DeadCodeEliminate.apply(f.ast)) == 0

    def test_a_leading_block_is_not_handled_yet(self):
        """Phase 3 adds this case; it needs guards the trailing one does
        not."""
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP32:
                with fp.FP16:
                    a = x * 3.0
                b = a + 1.0
            return b

        out, changed = UnnestContext.apply_with_status(f.ast)
        assert not changed
        assert _nested_pairs(out) == 1

    def test_no_context_statements_at_all(self):
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            return x + 1.0

        _out, changed = UnnestContext.apply_with_status(f.ast)
        assert not changed


class TestIdempotence:
    """One traversal reaches the normal form, so `Simplify` never needs a
    second round on this pass's account."""

    def test_a_second_run_changes_nothing(self):
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP32:
                a = x + 1.0
                with fp.FP16:
                    b = a * 3.0
                    with fp.FP64:
                        c = b * 3.0
                with fp.FP32:
                    d = c * 3.0
            return d

        once, changed = UnnestContext.apply_with_status(f.ast)
        assert changed
        _twice, changed_again = UnnestContext.apply_with_status(once)
        assert not changed_again


class TestTargets:
    """The trailing case needs no guard on the outer target: it stays bound
    where the hoisted block can still read it."""

    def test_the_hoisted_block_may_read_the_outer_target(self):
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP32 as c:
                a = x + 1.0
                with fp.FP16:
                    with c:
                        b = a * 3.0
            return b

        out = UnnestContext.apply(f.ast)
        assert _agrees(f, out)

    def test_the_hoisted_block_may_read_earlier_definitions(self):
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP32:
                a = x + 1.0
                with fp.FP16:
                    b = a * 3.0
            return b

        out = UnnestContext.apply(f.ast)
        assert _nested_pairs(out) == 0
        assert _agrees(f, out)
