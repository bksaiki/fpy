"""
Unit tests for the two analyses behind :mod:`fpy2.transform.hoistable`.

:func:`~fpy2.transform.hoistable.lowers` and
:func:`~fpy2.transform.hoistable.lowers_inside` say where the pass emits a
statement; :func:`~fpy2.transform.hoistable.force_names` is the *prefix rule*,
which says what must be named so a lowering does not overtake the operands to
its left.  The rule is the subtle part of the pass -- getting it wrong changes
which exception a program raises -- so it is tested here on its own.
"""

import fpy2 as fp
from fpy2.ast.fpyast import Expr, FuncDef, Stmt
from fpy2.ast.visitor import DefaultVisitor
from fpy2.transform.hoistable import force_names, lowers, lowers_inside

# ----------------------------------------------------------------------
# Helpers


def _stmt(f, index: int = 0) -> Stmt:
    """Statement *index* of `f`'s body."""
    return f.ast.body.stmts[index]


def _forced(node: 'Stmt | Expr') -> set[str]:
    """:func:`force_names` as formatted source, which is what a test can read."""
    return {e.format() for e in force_names(node)}


def _first(func: FuncDef, kind) -> Expr:
    """The first *kind* expression of `func`, in visit order."""
    found: list[Expr] = []

    class _F(DefaultVisitor):
        def _visit_expr(self, e, ctx):
            if isinstance(e, kind):
                found.append(e)
            super()._visit_expr(e, ctx)

    _F()._visit_function(func, None)
    assert found, f'no {kind.__name__} in the function'
    return found[0]


def _expr(f, kind) -> Expr:
    return _first(f.ast, kind)


# ----------------------------------------------------------------------
# `lowers`: where the pass emits a statement


class TestLowers:
    def test_ternary_with_a_compound_arm(self):
        @fp.fpy
        def f(x: fp.Real, c: bool) -> fp.Real:
            return fp.sqrt(x) if c else 0.0
        assert lowers(_expr(f, fp.ast.IfExpr))

    def test_ternary_over_atoms_is_left_alone(self):
        """``x1 if c else x2`` needs no slot: an arm that is already an atom has
        nothing to hoist."""
        @fp.fpy
        def f(x: fp.Real, y: fp.Real, c: bool) -> fp.Real:
            return x if c else y
        assert not lowers(_expr(f, fp.ast.IfExpr))

    def test_chain_with_a_compound_tail(self):
        @fp.fpy
        def f(x: fp.Real, y: fp.Real) -> bool:
            return x > 0.0 and y > 0.0
        assert lowers(_expr(f, fp.ast.And))

    def test_chain_over_atoms_is_left_alone(self):
        @fp.fpy
        def f(a: bool, b: bool) -> bool:
            return a and b
        assert not lowers(_expr(f, fp.ast.And))

    def test_a_first_operand_never_forces_a_lowering(self):
        """The first operand of a chain always runs, so it needs no guard --
        only the tail does."""
        @fp.fpy
        def f(x: fp.Real, b: bool) -> bool:
            return x > 0.0 and b
        assert not lowers(_expr(f, fp.ast.And))

    def test_a_strict_operator_never_lowers(self):
        @fp.fpy
        def f(xs: list[fp.Real], i: int) -> fp.Real:
            return fp.sqrt(xs[i]) + 1.0
        assert not lowers(_expr(f, fp.ast.Add))
        assert not lowers(_expr(f, fp.ast.ListRef))


# ----------------------------------------------------------------------
# `lowers_inside`: whether anything under a node lowers


class TestLowersInside:
    def test_finds_a_lowering_under_a_strict_operator(self):
        @fp.fpy
        def f(x: fp.Real, c: bool) -> fp.Real:
            return 1.0 + fp.sqrt(fp.sqrt(x) if c else 0.0)
        assert lowers_inside(_expr(f, fp.ast.Add))

    def test_a_program_with_no_non_strict_position(self):
        @fp.fpy
        def f(a: fp.Real, b: fp.Real, c: fp.Real, d: fp.Real) -> fp.Real:
            return (a * b) + (c * d)
        assert not lowers_inside(_expr(f, fp.ast.Add))

    def test_an_unlowered_ternary_still_reports_its_condition(self):
        """The arms are sealed but the condition is not: it runs whenever the
        ternary does, so it takes the ternary's own slot."""
        @fp.fpy
        def f(x: fp.Real, y: fp.Real, b: bool) -> fp.Real:
            return x if (b and fp.sqrt(x) > y) else y
        outer = _expr(f, fp.ast.IfExpr)
        assert not lowers(outer)
        assert lowers_inside(outer)

    def test_a_comprehension_is_sealed(self):
        """Its element runs once per iteration, so the pass hoists nothing out
        of it however much the element would want a slot."""
        @fp.fpy
        def f(xs: list[fp.Real], c: bool) -> list[fp.Real]:
            return [fp.sqrt(x) if c else 0.0 for x in xs]
        comp = _expr(f, fp.ast.ListComp)
        assert lowers(comp.elt)
        assert not lowers_inside(comp)


# ----------------------------------------------------------------------
# `force_names`: the prefix rule


class TestForceNames:
    def test_a_left_operand_is_named(self):
        """The ternary hoists above the statement, so `fp.sqrt(a)` must be named
        to keep its place."""
        @fp.fpy
        def f(a: fp.Real, b: fp.Real, c: bool) -> fp.Real:
            return fp.sqrt(a) + (fp.sqrt(b) if c else 0.0)
        assert _forced(_stmt(f)) == {'fp.sqrt(a)'}

    def test_a_right_operand_is_not(self):
        """Nothing runs before the ternary, so nothing is delayed by it."""
        @fp.fpy
        def f(a: fp.Real, b: fp.Real, c: bool) -> fp.Real:
            return (fp.sqrt(b) if c else 0.0) + fp.sqrt(a)
        assert _forced(_stmt(f)) == set()

    def test_the_maximal_subtree_is_named_not_its_parts(self):
        @fp.fpy
        def f(a: fp.Real, b: fp.Real, c: bool) -> fp.Real:
            return (fp.sqrt(a) * fp.sqrt(b)) + (fp.sqrt(b) if c else 0.0)
        assert _forced(_stmt(f)) == {'(fp.sqrt(a) * fp.sqrt(b))'}

    def test_an_atom_needs_no_name(self):
        @fp.fpy
        def f(a: fp.Real, b: fp.Real, c: bool) -> fp.Real:
            return a + (fp.sqrt(b) if c else 0.0)
        assert _forced(_stmt(f)) == set()

    def test_only_up_to_the_last_lowering(self):
        """An operand *after* the last lowering hoists nothing, so it overtakes
        nobody and stays inline."""
        @fp.fpy
        def f(a: fp.Real, b: fp.Real, c: bool) -> fp.Real:
            return fp.sqrt(a) + (fp.sqrt(b) if c else 0.0) + fp.sqrt(b)
        assert _forced(_stmt(f)) == {'fp.sqrt(a)'}

    def test_an_index_runs_before_the_value_it_is_assigned(self):
        @fp.fpy
        def f(xs: list[fp.Real], i: int, b: fp.Real, c: bool) -> list[fp.Real]:
            xs[i + 1] = fp.sqrt(b) if c else 0.0
            return xs
        assert _forced(_stmt(f)) == {'(i + 1)'}

    def test_an_atomic_index_needs_no_name(self):
        @fp.fpy
        def f(xs: list[fp.Real], i: int, b: fp.Real, c: bool) -> list[fp.Real]:
            xs[i] = fp.sqrt(b) if c else 0.0
            return xs
        assert _forced(_stmt(f)) == set()

    def test_an_assertion_test_runs_before_its_message(self):
        @fp.fpy
        def f(x: fp.Real, c: bool) -> fp.Real:
            assert fp.sqrt(x) > 0.0, (fp.sqrt(x) if c else 0.0)
            return x
        assert _forced(_stmt(f)) == {'fp.sqrt(x) > 0'}

    def test_a_lowering_in_a_ternary_arm_forces_nothing_outside_it(self):
        """The arm becomes a block of its own, so a lowering inside it lands
        there rather than above the statement."""
        @fp.fpy
        def f(a: fp.Real, b: fp.Real, c: bool, d: bool) -> fp.Real:
            return fp.sqrt(a) + ((fp.sqrt(b) if d else 0.0) if c else 0.0)
        assert _forced(_stmt(f)) == {'fp.sqrt(a)'}

    def test_a_lowered_ternary_never_names_its_own_arm(self):
        """Naming an arm would evaluate it unconditionally."""
        @fp.fpy
        def f(a: fp.Real, b: fp.Real, c: bool, d: bool) -> fp.Real:
            return fp.sqrt(a) if c else (fp.sqrt(b) if d else 0.0)
        assert _forced(_stmt(f)) == set()

    def test_a_chain_never_names_its_own_tail(self):
        @fp.fpy
        def f(x: fp.Real, y: fp.Real) -> bool:
            return x > 0.0 and y > 0.0 and fp.sqrt(y) > 0.0
        assert _forced(_stmt(f)) == set()

    def test_a_comprehension_is_sealed(self):
        @fp.fpy
        def f(xs: list[fp.Real], a: fp.Real, c: bool) -> list[fp.Real]:
            return [fp.sqrt(a) + (fp.sqrt(x) if c else 0.0) for x in xs]
        assert _forced(_stmt(f)) == set()

    def test_a_program_with_no_lowering_forces_no_name(self):
        """ANF would name both products here."""
        @fp.fpy
        def f(a: fp.Real, b: fp.Real, c: fp.Real, d: fp.Real) -> fp.Real:
            return (a * b) + (c * d)
        assert _forced(_stmt(f)) == set()

    def test_a_statement_of_a_nested_block_is_not_reached(self):
        """Each statement gets its own call: its temporaries belong to its own
        block, not to an enclosing one."""
        @fp.fpy
        def f(a: fp.Real, b: fp.Real, c: bool) -> fp.Real:
            if c:
                y = fp.sqrt(a) + (fp.sqrt(b) if c else 0.0)
            else:
                y = a
            return y
        if_stmt = _stmt(f)
        assert _forced(if_stmt) == set()
        assert _forced(if_stmt.ift.stmts[0]) == {'fp.sqrt(a)'}
