"""Unit tests for :func:`fpy2.strategies.to_anf`.

The transform itself is covered by ``tests/unit/transform/test_anf*.py``; these
check the operator wrapper -- that it is the transform under a `Function`, that
its docstring example holds, and that what it unblocks for a schedule actually
becomes reachable.
"""

import pytest

import fpy2 as fp
from fpy2.ast import Assign, ContextStmt, ForeignVal, IfExpr, IfStmt
from fpy2.ast.visitor import DefaultVisitor
from fpy2.number import REAL
from fpy2.strategies import elim_round, to_anf
from fpy2.transform import ANF


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
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                return fp.sqrt(x * x) + 1.0

        assert to_anf(f).ast.format() == (
            '@fp.fpy\n'
            'def f(x):\n'
            '    with fp.FP64:\n'
            '        t = (x * x)\n'
            '        t2 = fp.sqrt(t)\n'
            '        return (t2 + 1)'
        )

    def test_is_the_transform(self):
        @fp.fpy
        def f(a: fp.Real, b: fp.Real) -> fp.Real:
            with fp.FP64:
                return (a * b) + (a - b)

        assert to_anf(f).ast.format() == ANF.apply(f.ast).format()

    def test_keeps_the_runtime_and_records_a_parent(self):
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                return x * x

        out = to_anf(f)
        assert out.runtime is f.runtime
        assert out.parent is f
        assert repr(out(2.0)) == repr(f(2.0))

    def test_rejects_a_funcdef(self):
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            return x

        with pytest.raises(TypeError):
            to_anf(f.ast)                      # type: ignore[arg-type]

    def test_takes_no_where(self):
        """Normal form is not a per-site decision, so there is no index to
        pass."""

        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                return x * x

        with pytest.raises(TypeError):
            to_anf(f, 0)                       # type: ignore[call-arg]


class TestWhatItUnblocks:
    """A schedule's reason for running it: code that was unreachable."""

    def test_elim_round_reaches_a_ternary_arm_afterwards(self):
        """`elim_round` suppresses its hoist inside a ternary arm -- hoisting an
        operand out of a conditional would evaluate it either way.  After
        `to_anf` each arm is a block, so the hoist happens."""

        @fp.fpy
        def f(cond: bool) -> fp.Real:
            with fp.FP64:
                return (1.0 + 2.0) if cond else (3.0 + 4.0)

        assert _real_blocks(elim_round(f)) == 0
        flat = to_anf(f)
        assert _count(flat, IfExpr) == 0
        assert _count(flat, IfStmt) == 1
        # one per arm, now that each arm has a statement slot
        assert _real_blocks(elim_round(flat)) == 2

    def test_semantics_are_unchanged_by_the_pair(self):
        @fp.fpy
        def f(cond: bool) -> fp.Real:
            with fp.FP64:
                return (1.0 + 2.0) if cond else (3.0 + 4.0)

        lowered = elim_round(to_anf(f))
        for cond in (True, False):
            # `==`, not `repr`: eliminating a rounding leaves the value tagged
            # with the `REAL` context it was computed under, which is
            # `elim_round`'s own behavior and not a change of value.
            assert f(cond) == lowered(cond)

    def test_a_bool_tail_becomes_statements(self):
        @fp.fpy
        def f(x: fp.Real) -> bool:
            with fp.FP64:
                y = x > 0.0 or fp.round(x * x) > 1.0
                return y

        flat = to_anf(f)
        assert _count(flat, Assign) >= 2
        assert repr(f(2.0)) == repr(flat(2.0))
        assert repr(f(-2.0)) == repr(flat(-2.0))
