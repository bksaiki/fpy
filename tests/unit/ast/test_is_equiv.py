"""
Unit tests for ``Ast.is_equiv`` — structural equivalence of AST nodes.

``is_equiv`` is load-bearing beyond its own tests: every transform suite uses
it to assert "this pass left the function alone", so a node that compares
sloppily silently weakens those assertions.  Two shapes have gone wrong here:

- comparing an optional sub-expression with ``==``, which is *identity* for an
  ``Expr`` and so reports structurally-equal nodes as different (a false
  negative — the pass looks like it changed something when it didn't); and
- forgetting a field entirely, which reports different nodes as equivalent (a
  false positive — the more dangerous direction, since a negative test then
  passes even when a pass mangled that field).

Both are covered below.  See ``_opt_is_equiv`` in ``fpy2/ast/fpyast.py``.
"""

import fpy2 as fp

from fpy2.ast.visitor import DefaultTransformVisitor


def _last_expr(f: fp.Function):
    """The expression returned by *f*'s final statement."""
    return f.ast.body.stmts[-1].expr


def _first_stmt(f: fp.Function):
    return f.ast.body.stmts[0]


class TestListSliceIsEquiv:
    """A slice compares on *both* bounds.

    ``stop`` used to be skipped: the method returned inside the ``start``
    match, so ``xs[1:3]`` reported equivalent to ``xs[1:5]`` and to ``xs[1:]``.
    """

    def test_differing_stop_is_not_equivalent(self):
        @fp.fpy
        def a(xs: list[fp.Real]) -> list[fp.Real]:
            return xs[1:3]

        @fp.fpy
        def b(xs: list[fp.Real]) -> list[fp.Real]:
            return xs[1:5]

        assert not _last_expr(a).is_equiv(_last_expr(b))

    def test_absent_stop_is_not_equivalent_to_present(self):
        @fp.fpy
        def a(xs: list[fp.Real]) -> list[fp.Real]:
            return xs[1:3]

        @fp.fpy
        def b(xs: list[fp.Real]) -> list[fp.Real]:
            return xs[1:]

        assert not _last_expr(a).is_equiv(_last_expr(b))
        assert not _last_expr(b).is_equiv(_last_expr(a))

    def test_absent_start_is_not_equivalent_to_present(self):
        @fp.fpy
        def a(xs: list[fp.Real]) -> list[fp.Real]:
            return xs[1:3]

        @fp.fpy
        def b(xs: list[fp.Real]) -> list[fp.Real]:
            return xs[:3]

        assert not _last_expr(a).is_equiv(_last_expr(b))
        assert not _last_expr(b).is_equiv(_last_expr(a))

    def test_identical_slices_are_equivalent(self):
        @fp.fpy
        def a(xs: list[fp.Real]) -> list[fp.Real]:
            return xs[1:3]

        @fp.fpy
        def b(xs: list[fp.Real]) -> list[fp.Real]:
            return xs[1:3]

        assert _last_expr(a).is_equiv(_last_expr(b))

    def test_both_bounds_absent_are_equivalent(self):
        @fp.fpy
        def a(xs: list[fp.Real]) -> list[fp.Real]:
            return xs[:]

        @fp.fpy
        def b(xs: list[fp.Real]) -> list[fp.Real]:
            return xs[:]

        assert _last_expr(a).is_equiv(_last_expr(b))

    def test_survives_a_no_op_rebuild(self):
        @fp.fpy
        def f(xs: list[fp.Real]) -> list[fp.Real]:
            return xs[1:3]

        rebuilt = DefaultTransformVisitor()._visit_function(f.ast, None)
        assert rebuilt.is_equiv(f.ast)


class TestAssertStmtIsEquiv:
    """An assertion compares its message structurally, not by identity.

    ``self.msg == other.msg`` on an ``Expr`` is an identity check, so any
    ``DefaultTransformVisitor`` pass — which rebuilds every node — reported a
    message-carrying assert as changed even when it was untouched.
    """

    def test_survives_a_no_op_rebuild(self):
        @fp.fpy
        def f():
            assert 0 == 0, "a message"
            return 0

        rebuilt = DefaultTransformVisitor()._visit_function(f.ast, None)
        assert rebuilt.is_equiv(f.ast)

    def test_differing_messages_are_not_equivalent(self):
        @fp.fpy
        def f():
            assert 0 == 0, "one"
            return 0

        @fp.fpy
        def g():
            assert 0 == 0, "two"
            return 0

        assert not _first_stmt(f).is_equiv(_first_stmt(g))

    def test_message_and_no_message_are_not_equivalent(self):
        @fp.fpy
        def f():
            assert 0 == 0, "a message"
            return 0

        @fp.fpy
        def g():
            assert 0 == 0
            return 0

        assert not _first_stmt(f).is_equiv(_first_stmt(g))
        assert not _first_stmt(g).is_equiv(_first_stmt(f))

    def test_no_messages_are_equivalent(self):
        @fp.fpy
        def f():
            assert 0 == 0
            return 0

        rebuilt = DefaultTransformVisitor()._visit_function(f.ast, None)
        assert rebuilt.is_equiv(f.ast)
