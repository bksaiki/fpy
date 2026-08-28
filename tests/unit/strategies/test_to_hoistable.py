"""Unit tests for :func:`fpy2.strategies.to_hoistable`.

The transform itself is covered by ``tests/unit/transform/test_hoistable*.py``;
these check the operator wrapper -- that it is the transform under a `Function`,
that its docstring example holds, and that what it unblocks for a schedule
actually becomes reachable.
"""

import pytest

import fpy2 as fp
from fpy2.ast import ContextStmt, ForeignVal, IfExpr, IfStmt
from fpy2.ast.visitor import DefaultVisitor
from fpy2.number import REAL
from fpy2.strategies import comp_to_loop, elim_round, to_anf, to_hoistable
from fpy2.transform import Hoistable


def _count(func, kind) -> int:
    n = 0

    class _C(DefaultVisitor):
        def _visit_expr(self, e, ctx):
            nonlocal n
            if isinstance(e, kind):
                n += 1
            super()._visit_expr(e, ctx)

        def _visit_statement(self, stmt, ctx):
            nonlocal n
            if isinstance(stmt, kind):
                n += 1
            super()._visit_statement(stmt, ctx)

    _C()._visit_function(func.ast if hasattr(func, 'ast') else func, None)
    return n


def _real_blocks(func) -> int:
    """``with fp.REAL:`` blocks in *func* — what `elim_round`'s hoist adds."""
    n = 0

    class _C(DefaultVisitor):
        def _visit_context(self, stmt: ContextStmt, ctx):
            nonlocal n
            if isinstance(stmt.ctx, ForeignVal) and stmt.ctx.val is REAL:
                n += 1
            super()._visit_context(stmt, ctx)

    _C()._visit_function(func.ast, None)
    return n


class TestOperator:
    """The wrapper: a `Function` in, a `Function` out."""

    def test_docstring_example(self):
        @fp.fpy
        def f(a: fp.Real, b: fp.Real, c: bool) -> fp.Real:
            with fp.FP64:
                y = (a * b) + (a - b)
                return y if c else fp.sqrt(y)

        assert to_hoistable(f).ast.format() == (
            '@fp.fpy\n'
            'def f(a, b, c):\n'
            '    with fp.FP64:\n'
            '        y = ((a * b) + (a - b))\n'
            '        if c:\n'
            '            t = y\n'
            '        else:\n'
            '            t = fp.sqrt(y)\n'
            '        return t'
        )

    def test_is_the_transform(self):
        @fp.fpy
        def f(a: fp.Real, b: fp.Real, c: bool) -> fp.Real:
            with fp.FP64:
                return (a * b) + (fp.sqrt(a) if c else b)

        assert to_hoistable(f).ast.format() == Hoistable.apply(f.ast).format()

    def test_it_is_weaker_than_to_anf(self):
        """The reason both exist.  ANF names the two products; this names
        nothing, because nothing is hoisted above them."""

        @fp.fpy
        def f(a: fp.Real, b: fp.Real) -> fp.Real:
            with fp.FP64:
                return (a * b) + (a - b)

        assert to_hoistable(f).ast.format() == f.ast.format()
        assert to_anf(f).ast.format() != f.ast.format()

    def test_keeps_the_runtime_and_records_a_parent(self):
        @fp.fpy
        def f(x: fp.Real, c: bool) -> fp.Real:
            with fp.FP64:
                return fp.sqrt(x) if c else x

        out = to_hoistable(f)
        assert out.runtime is f.runtime
        assert out.parent is f
        assert repr(out(2.0, True)) == repr(f(2.0, True))

    def test_rejects_a_funcdef(self):
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            return x

        with pytest.raises(TypeError):
            to_hoistable(f.ast)                # type: ignore[arg-type]

    def test_takes_no_where(self):
        """Normal form is not a per-site decision, so there is no index to
        pass."""

        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                return x * x

        with pytest.raises(TypeError):
            to_hoistable(f, 0)                 # type: ignore[call-arg]


class TestWhatItUnblocks:
    """A schedule's reason for running it: code that was unreachable."""

    def test_elim_round_reaches_a_ternary_arm_afterwards(self):
        """`elim_round` suppresses its hoist inside a ternary arm -- hoisting an
        operand out of a conditional would evaluate it either way.  Afterwards
        each arm is a block, so the hoist happens."""

        @fp.fpy
        def f(cond: bool) -> fp.Real:
            with fp.FP64:
                return (1.0 + 2.0) if cond else (3.0 + 4.0)

        assert _real_blocks(elim_round(f)) == 0
        flat = to_hoistable(f)
        assert _count(flat, IfExpr) == 0
        assert _count(flat, IfStmt) == 1
        # one per arm, now that each arm has a statement slot
        assert _real_blocks(elim_round(flat)) == 2

    def test_semantics_are_unchanged_by_the_pair(self):
        @fp.fpy
        def f(cond: bool) -> fp.Real:
            with fp.FP64:
                return (1.0 + 2.0) if cond else (3.0 + 4.0)

        lowered = elim_round(to_hoistable(f))
        for cond in (True, False):
            # `==`, not `repr`: eliminating a rounding leaves the value tagged
            # with the `REAL` context it was computed under, which is
            # `elim_round`'s own behavior and not a change of value.
            assert f(cond) == lowered(cond)

    def test_comp_to_loop_first_makes_the_whole_function_hoistable(self):
        """The documented order.  A comprehension's element has no slot until
        `comp_to_loop` has given it the loop body it generates."""

        @fp.fpy
        def f(xs: list[fp.Real], c: bool) -> list[fp.Real]:
            with fp.FP64:
                return [fp.sqrt(x) if c else 0.0 for x in xs]

        assert Hoistable.refusals(to_hoistable(f).ast) != []
        both = to_hoistable(comp_to_loop(f))
        assert Hoistable.refusals(both.ast) == []
        assert repr(both([4.0, 9.0], True)) == repr(f([4.0, 9.0], True))
