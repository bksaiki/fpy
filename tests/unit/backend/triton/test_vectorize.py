"""`why_not_tileable`: may a loop body be evaluated as a tile?

Splitting a loop preserves semantics; evaluating the inner body as a tile is
what can change the answer, and only through the loop-carried variables.
"""

import pytest

import fpy2 as fp
from fpy2.ast.fpyast import ForStmt
from fpy2.ast.visitor import DefaultVisitor
from fpy2.backend.triton import why_not_tileable


class _Loops(DefaultVisitor):
    def __init__(self):
        super().__init__()
        self.out: list[ForStmt] = []

    def _visit_for(self, s: ForStmt, ctx):
        self.out.append(s)
        return super()._visit_for(s, ctx)


def _innermost(func) -> ForStmt:
    v = _Loops()
    v._visit_function(func.ast, None)
    assert v.out, 'expected a loop'
    return v.out[-1]


def _why(func) -> str | None:
    return why_not_tileable(_innermost(func), func.ast)


class TestTileable:
    def test_a_max_fold_selects_an_operand(self):
        @fp.fpy(ctx=fp.FP64)
        def f(xs: list[fp.Real]):
            m = fp.round(0)
            for x in xs:
                m = max(m, x)
            return m

        assert _why(f) is None

    def test_a_flag_cleared_under_a_guard_is_idempotent(self):
        """`matrix.is_diagonal`'s shape: an `and`-fold that does not look like
        one.  Which iteration cleared the flag does not matter."""
        @fp.fpy(ctx=fp.FP64)
        def f(xs: list[fp.Real]):
            ok: bool = True
            for x in xs:
                if x != 0:
                    ok = False
            return ok

        assert _why(f) is None

    def test_an_element_write_at_the_loop_variable(self):
        @fp.fpy(ctx=fp.FP64)
        def f(xs: list[fp.Real]):
            out = [fp.round(0) for _ in xs]
            for i in range(len(xs)):
                out[i] = xs[i] * 2
            return out

        assert _why(f) is None

    def test_a_nested_write_mixes_the_loop_variable_with_an_invariant(self):
        """`out[i][j]` in the `j` loop: `i` is invariant there, so the
        elements are still distinct."""
        @fp.fpy(ctx=fp.FP64)
        def f(A: list[list[fp.Real]]):
            out = fp.empty(len(A), len(A[0]))
            for i in range(len(A)):
                for j in range(len(A[0])):
                    out[i][j] = A[i][j] * 2
            return out

        assert _why(f) is None

    def test_an_exact_accumulation_may_be_regrouped(self):
        """Under `INTEGER`, which is unbounded, no partial sum rounds."""
        @fp.fpy(ctx=fp.INTEGER)
        def f(xs: list[fp.Real]):
            acc = fp.round(0)
            for x in xs:
                acc = acc + fp.round(x)
            return acc

        assert _why(f) is None


class TestRefuses:
    def test_a_rounded_accumulation(self):
        """The case the whole predicate exists for: regrouping rounded adds
        moves bits, so this stays sequential per lane."""
        @fp.fpy(ctx=fp.FP32)
        def f(xs: list[fp.Real]):
            acc = fp.round(0)
            for x in xs:
                acc = acc + x
            return acc

        assert 'regrouping it moves bits' in (_why(f) or '')

    def test_two_iterations_writing_the_same_element(self):
        """`out[i % 2]` -- refused rather than sent to a dependence test."""
        @fp.fpy(ctx=fp.INTEGER)
        def f(xs: list[fp.Real]):
            out = [fp.round(0) for _ in range(2)]
            for i in range(len(xs)):
                out[i % 2] = xs[i]
            return out

        assert 'cannot show distinct' in (_why(f) or '')

    def test_an_index_that_ignores_the_loop_variable(self):
        @fp.fpy(ctx=fp.INTEGER)
        def f(xs: list[fp.Real], k: fp.Real):
            out = [fp.round(0) for _ in xs]
            for i in range(len(xs)):
                out[k] = xs[i]
            return out

        assert 'same index' in (_why(f) or '')

    def test_reading_the_list_back_while_writing_it(self):
        @fp.fpy(ctx=fp.FP64)
        def f(xs: list[fp.Real]):
            out = [fp.round(0) for _ in xs]
            for i in range(1, len(xs)):
                out[i] = out[i - 1] + xs[i]
            return out

        assert 'read back while being written' in (_why(f) or '')

    def test_a_write_that_ignores_the_carried_value(self):
        @fp.fpy(ctx=fp.FP64)
        def f(xs: list[fp.Real]):
            last = fp.round(0)
            for x in xs:
                last = x * 2
            return last

        assert 'which iteration wrote last' in (_why(f) or '')


class TestApi:
    def test_a_loop_carrying_nothing_is_tileable(self):
        @fp.fpy(ctx=fp.FP64)
        def f(xs: list[fp.Real]):
            for x in xs:
                y = x * 2
            return fp.round(0)

        assert _why(f) is None

    def test_a_non_for_is_a_type_error(self):
        @fp.fpy(ctx=fp.FP64)
        def f(x: fp.Real):
            return x

        with pytest.raises(TypeError, match='ForStmt'):
            why_not_tileable(f.ast.body.stmts[-1], f.ast)
