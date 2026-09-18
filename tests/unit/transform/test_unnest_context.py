"""
`UnnestContext` — hoisting a `with` out of either edge of another one's body.

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


    def test_a_peel_may_leave_a_sole_nested_block(self):
        """Peeling the trailing block leaves the parent holding one nested
        statement, which this pass declines and `DeadCodeEliminate`
        finishes."""
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP32:
                with fp.FP16:
                    a = x * 3.0
                with fp.FP64:
                    b = a + 1.0
            return b

        out, changed = UnnestContext.apply_with_status(f.ast)
        assert changed
        assert _nested_pairs(out) == 1
        assert _agrees(f, out)
        assert _nested_pairs(DeadCodeEliminate.apply(out)) == 0

    def test_a_live_target_keeps_the_leftover_nest(self):
        """Same peel, but the hoisted block reads the parent's target, so
        `DeadCodeEliminate` may not drop the header that binds it and the
        nesting stays.  Correct, and the reason the pair count is not a
        measure of success on its own."""
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP32 as c:
                with fp.FP16:
                    a = x * 3.0
                with c:
                    b = a + 1.0
            return b

        out, changed = UnnestContext.apply_with_status(f.ast)
        assert changed
        assert _agrees(f, out)
        assert _nested_pairs(DeadCodeEliminate.apply(out)) == 1

    def test_the_hoisted_block_may_read_the_outer_target(self):
        """The trailing case needs no guard on the outer target: it stays
        bound where the hoisted block can still read it."""
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


class TestLeading:
    """Hoisting a block out of the *front* of its parent moves the parent's
    header after it, which the trailing case never does."""

    def test_the_shape_it_states(self):
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP32:
                with fp.FP16:
                    a = x * 3.0
                b = a + 1.0
            return b

        out = UnnestContext.apply(f.ast)
        assert _text(f, out) == (
            '@fp.fpy def f(x): with fp.FP16: a = (x * 3) '
            'with fp.FP32: b = (a + 1) return b'
        )
        assert _nested_pairs(out) == 0
        assert _agrees(f, out)

    def test_it_peels_a_run_of_them(self):
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP32:
                with fp.FP16:
                    a = x * 3.0
                with fp.FP64:
                    b = a * 3.0
                c = b + 1.0
            return c

        out = UnnestContext.apply(f.ast)
        assert _nested_pairs(out) == 0
        src = _text(f, out)
        assert src.index('fp.FP16') < src.index('fp.FP64') < src.index('fp.FP32')
        assert _agrees(f, out)

    def test_both_edges_at_once(self):
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP32:
                with fp.FP16:
                    a = x * 3.0
                b = a + 1.0
                with fp.FP64:
                    c = b * 3.0
            return c

        out = UnnestContext.apply(f.ast)
        assert _nested_pairs(out) == 0
        assert _agrees(f, out)


class TestLeadingGuards:
    """Each refusal below is a way the hoist would be observable."""

    def test_an_impure_header_is_refused(self):
        """The header is evaluated after the hoisted block instead of before
        it, so its evaluation may have no effects to reorder."""
        def make_ctx():          # a foreign callable: impure by default
            return fp.FP32

        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with make_ctx():
                with fp.FP16:
                    a = x * 3.0
                b = a + 1.0
            return b

        _out, changed = UnnestContext.apply_with_status(f.ast)
        assert not changed

    def test_an_impure_hoisted_context_is_refused(self):
        """The hoisted block's own header is evaluated before the enclosing
        one instead of after, so it may have no effects to reorder either."""
        def make_ctx():          # a foreign callable: impure by default
            return fp.FP16

        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP32:
                with make_ctx():
                    a = x * 3.0
                b = a + 1.0
            return b

        _out, changed = UnnestContext.apply_with_status(f.ast)
        assert not changed

    def test_an_impure_context_promoted_into_the_run_is_refused(self):
        """Peeling is bottom-up, so a block nested *inside* the leading one
        can be promoted alongside it.  Its header is checked too, or the
        purity guard would only cover the run's roots."""
        def make_ctx():          # a foreign callable: impure by default
            return fp.FP64

        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP32:
                with fp.FP16:
                    a = x * 3.0
                    with make_ctx():
                        c = a + 1.0
                b = c + 1.0
            return b

        # the program cannot be run -- the interpreter refuses to call a
        # foreign function -- so the shape is the whole assertion
        src = _text(f, UnnestContext.apply(f.ast))
        assert src.index('fp.FP32') < src.index('make_ctx'), src

    def test_a_header_reading_a_rebound_name_is_refused(self):
        """``with c:`` evaluated after a block that rebinds ``c`` would
        install the new context instead of the old one."""
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            c = fp.FP32
            with c:
                with fp.FP16:
                    c = fp.FP64
                    a = x * 3.0
                b = a + 1.0
            return b

        _out, changed = UnnestContext.apply_with_status(f.ast)
        assert not changed

    def test_a_hoisted_block_reading_the_outer_target_is_refused(self):
        """``c`` is bound by the header, which now comes *after* the block
        that reads it."""
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP32 as c:
                with fp.FP16:
                    with c:
                        a = x * 3.0
                b = a + 1.0
            return b

        _out, changed = UnnestContext.apply_with_status(f.ast)
        assert not changed

    def test_the_outer_target_read_through_a_loop_is_refused(self):
        """Same, with the read behind a loop -- the case where the use might
        reach the target's definition through a phi rather than directly."""
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP32 as c:
                with fp.FP16:
                    a = x
                    for _ in range(2):
                        with c:
                            a = a * 3.0
                b = a + 1.0
            return b

        _out, changed = UnnestContext.apply_with_status(f.ast)
        assert not changed

    def test_an_unread_target_is_no_obstacle(self):
        """A `NamedId` target the hoisted block does not read refuses
        nothing -- the guard is a def-use question, not `UnderscoreId` or
        bust."""
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP32 as c:
                with fp.FP16:
                    a = x * 3.0
                b = a + 1.0
            return b

        out, changed = UnnestContext.apply_with_status(f.ast)
        assert changed
        assert _nested_pairs(out) == 0
        assert _agrees(f, out)
