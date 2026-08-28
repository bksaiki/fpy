"""Unit tests for :func:`fpy2.strategies.to_anf`.

The transform itself is covered by ``tests/unit/transform/test_anf*.py``; these
check the operator wrapper -- that it is the transform under a `Function`, that
its docstring example holds, and that it inherits the transform's precondition
rather than quietly satisfying it.
"""

import pytest

import fpy2 as fp
from fpy2.ast import IfExpr, IfStmt
from fpy2.ast.visitor import DefaultVisitor
from fpy2.strategies import to_anf, to_hoistable
from fpy2.transform import ANF
from fpy2.transform.error import TransformError


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


class TestPrecondition:
    """The operator inherits `ANF`'s precondition rather than composing, so each
    strategy stays one rewrite and a schedule spells the order it wants."""

    @staticmethod
    def _needy():
        """A ternary arm that needs a statement: `fp.cast` may assert."""
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                with fp.FP32:
                    y = 0.0 if x > 1e30 else fp.cast(x)
                return y
        return f

    def test_it_refuses_a_ternary_arm_that_needs_a_slot(self):
        with pytest.raises(TransformError, match='ternary arm'):
            to_anf(self._needy())

    def test_the_message_says_to_run_to_hoistable(self):
        with pytest.raises(TransformError, match='Hoistable'):
            to_anf(self._needy())

    def test_to_hoistable_first_is_the_composition(self):
        f = self._needy()
        flat = to_anf(to_hoistable(f))
        assert _count(flat, IfExpr) == 0
        assert _count(flat, IfStmt) == 1
        for x in (1e300, 1.0):
            assert repr(f(x)) == repr(flat(x))

    def test_a_ternary_whose_arms_need_no_slot_is_accepted(self):
        """The gate asks what `ANF` would have to *name*, not whether the
        program is in hoistable form.  Pure arithmetic in an arm needs no
        statement, so the ternary is accepted and left nested."""

        @fp.fpy
        def f(cond: bool) -> fp.Real:
            with fp.FP64:
                return (1.0 + 2.0) if cond else (3.0 + 4.0)

        assert _count(to_anf(f), IfExpr) == 1
        assert _count(to_anf(to_hoistable(f)), IfExpr) == 0
