"""
Unit tests for the :class:`fpy2.transform.Hoistable` transform.

Fresh names come from ``Gensym``, so a hand-written golden AST is brittle.  These
tests assert, as ``test_anf.py`` does:

1. **The invariant** — after the pass, `Hoistable.refusals` is empty, so a
   temporary may be hoisted out of anywhere.  A comprehension is the exception,
   and only until :class:`~fpy2.transform.CompToLoop` has run.
2. **Weakness** — a program with no non-strict position comes back unchanged.
   This is the whole point of the pass existing beside `ANF`, and nothing else
   checks it: a regression that started flattening more would still be correct.
3. **Order of evaluation** — a lowering hoists above the operands to its left,
   so those must be named.  Getting this wrong changes which exception a program
   raises, which :class:`TestOrdering` witnesses directly.
4. **Semantic equivalence** through the interpreter, and idempotence.

``test_hoistable_analysis.py`` covers the prefix rule on its own, and
``test_hoistable_profile.py`` pins how little the pass does to the corpus.
"""

import pytest

import fpy2 as fp
from fpy2 import Function
from fpy2.ast.fpyast import (
    Assign,
    FuncDef,
    ContextStmt,
    If1Stmt,
    IfExpr,
    IfStmt,
    Not,
    StmtBlock,
    Var,
    WhileStmt,
)
from fpy2.ast.visitor import DefaultVisitor
from fpy2.transform import ANF, CompToLoop
from fpy2.transform.hoistable import Hoistable
from fpy2.transform.path import walk_stmts

# ----------------------------------------------------------------------
# Helpers


def _apply(f: Function) -> Function:
    return Function(Hoistable.apply(f.ast), runtime=f.runtime)


def _run(f: Function, args):
    """``(result, None)`` or ``(None, exception name)``.

    FPy has undefined behavior, so a program that raises must raise the *same
    way* after the pass -- an operand evaluated out of turn would not.
    """
    try:
        return repr(f(*args)), None
    except Exception as e:                       # noqa: BLE001
        return None, type(e).__name__


def _count(node, kind) -> int:
    """How many *kind* nodes are in *node* (a `FuncDef`, `StmtBlock` or
    `Stmt`)."""
    n = 0

    class _C(DefaultVisitor):
        def _visit_statement(self, stmt, ctx):
            nonlocal n
            if isinstance(stmt, kind):
                n += 1
            super()._visit_statement(stmt, ctx)

        def _visit_expr(self, e, ctx):
            nonlocal n
            if isinstance(e, kind):
                n += 1
            super()._visit_expr(e, ctx)

    if isinstance(node, FuncDef):
        _C()._visit_function(node, None)
    elif isinstance(node, StmtBlock):
        _C()._visit_block(node, None)
    else:
        _C()._visit_statement(node, None)
    return n


def _first(node, kind):
    """The first *kind* node in *node* (a `FuncDef` or a `StmtBlock`), in visit
    order."""
    found: list = []

    class _F(DefaultVisitor):
        def _visit_expr(self, e, ctx):
            if isinstance(e, kind):
                found.append(e)
            super()._visit_expr(e, ctx)

        def _visit_statement(self, stmt, ctx):
            if isinstance(stmt, kind):
                found.append(stmt)
            super()._visit_statement(stmt, ctx)

    if isinstance(node, FuncDef):
        _F()._visit_function(node, None)
    else:
        _F()._visit_block(node, None)
    assert found, f'no {kind.__name__} in the function'
    return found[0]


def _loop_of(func) -> WhileStmt:
    """The one `while` statement of `func`."""
    found = [s for s in func.body.stmts if isinstance(s, WhileStmt)]
    assert len(found) == 1, f'{len(found)} loops, expected 1'
    return found[0]


def _assigns_to(func, name: str) -> int:
    """How many assignments in `func` target a name starting with *name*."""
    n = 0

    class _C(DefaultVisitor):
        def _visit_assign(self, stmt: Assign, ctx):
            nonlocal n
            if str(stmt.target).startswith(name):
                n += 1
            super()._visit_assign(stmt, ctx)

    _C()._visit_function(func, None)
    return n


# ----------------------------------------------------------------------
# Programs, reused by the invariant, semantics and idempotence tests


@fp.fpy
def _ternary(a: fp.Real, b: fp.Real, c: bool) -> fp.Real:
    return fp.sqrt(a) + (fp.sqrt(b) if c else 0.0)


@fp.fpy
def _nested_ternary(a: fp.Real, b: fp.Real, c: bool, d: bool) -> fp.Real:
    return fp.sqrt(a) + ((fp.sqrt(b) if d else 0.0) if c else 0.0)


@fp.fpy
def _chain(x: fp.Real, y: fp.Real) -> bool:
    return x > 0.0 and fp.sqrt(y) > 0.0 and fp.sqrt(x) > 0.0


@fp.fpy
def _loop(x: fp.Real, n: fp.Real) -> fp.Real:
    i = 0.0
    while i < n:
        x = x + i
        i = i + 1.0
    return x


@fp.fpy
def _loop_with_a_ternary_condition(x: fp.Real, n: fp.Real, c: bool) -> fp.Real:
    while (fp.sqrt(x) if c else x) < n:
        x = x + 1.0
    return x


@fp.fpy
def _indexed(xs: list[fp.Real], i: int, b: fp.Real, c: bool) -> list[fp.Real]:
    xs[i + 1] = fp.sqrt(b) if c else 0.0
    return xs


@fp.fpy
def _computed_context(x: fp.Real, nb: int, c: bool) -> fp.Real:
    with fp.IEEEContext(8, 24 if c else nb + 1):
        y = x * x
    return y


@fp.fpy
def _straight_line(a: fp.Real, b: fp.Real, c: fp.Real, d: fp.Real) -> fp.Real:
    return (a * b) + (c * d)


_PROGRAMS = [
    (_ternary, [4.0, 9.0, True]),
    (_ternary, [4.0, 9.0, False]),
    (_nested_ternary, [4.0, 9.0, True, False]),
    (_chain, [4.0, 9.0]),
    (_chain, [-4.0, 9.0]),
    (_loop, [1.0, 4.0]),
    (_loop_with_a_ternary_condition, [1.0, 4.0, True]),
    (_indexed, [[1.0, 2.0, 3.0], 1, 9.0, True]),
    (_computed_context, [1.5, 10, False]),
    (_straight_line, [1.0, 2.0, 3.0, 4.0]),
]


# ----------------------------------------------------------------------
# The invariant


class TestInvariant:
    @pytest.mark.parametrize('f', [f for f, _args in _PROGRAMS], ids=lambda f: f.name)
    def test_nothing_is_left_unhoistable(self, f):
        assert Hoistable.refusals(Hoistable.apply(f.ast)) == []

    def test_a_comprehension_is_the_one_refusal(self):
        @fp.fpy
        def f(xs: list[fp.Real], c: bool) -> list[fp.Real]:
            return [fp.sqrt(x) if c else 0.0 for x in xs]

        reasons = [w for _e, w in Hoistable.refusals(Hoistable.apply(f.ast))]
        assert reasons == ["a comprehension's element runs once per iteration"]

    def test_comp_to_loop_first_clears_it(self):
        """Why `CompToLoop` runs first: the loop body it generates is the slot
        the element lacked, and the ternary in it then lowers."""
        @fp.fpy
        def f(xs: list[fp.Real], c: bool) -> list[fp.Real]:
            return [fp.sqrt(x) if c else 0.0 for x in xs]

        out = Hoistable.apply(CompToLoop.apply(f.ast))
        assert Hoistable.refusals(out) == []
        assert _count(out, IfExpr) == 0


# ----------------------------------------------------------------------
# Weakness: what the pass leaves alone


class TestWeakness:
    def test_a_program_with_no_non_strict_position_is_unchanged(self):
        before = _straight_line.ast.format()
        assert Hoistable.apply(_straight_line.ast).format() == before

    def test_anf_would_have_flattened_it(self):
        """States the difference the pass exists for, so a regression that
        started atomizing would fail here rather than pass everything."""
        anf = ANF.apply(_straight_line.ast)
        assert anf.format() != _straight_line.ast.format()
        assert _assigns_to(anf, 't') == 2

    def test_an_atomic_loop_condition_is_not_rotated(self):
        @fp.fpy
        def f(b: bool, x: fp.Real) -> fp.Real:
            while b:
                x = x + 1.0
                b = x < 0.0
            return x

        out = Hoistable.apply(f.ast)
        assert out.format() == f.ast.format()

    def test_a_ternary_over_atoms_stays_an_expression(self):
        @fp.fpy
        def f(x: fp.Real, y: fp.Real, c: bool) -> fp.Real:
            return x if c else y

        out = Hoistable.apply(f.ast)
        assert _count(out, IfExpr) == 1
        assert _count(out, IfStmt) == 0


# ----------------------------------------------------------------------
# Order of evaluation


@fp.fpy
def _needs_positive_g(x: fp.Real) -> fp.Real:
    assert x > 0.0, 'g'
    return x


@fp.fpy
def _needs_positive_h(x: fp.Real) -> fp.Real:
    assert x > 0.0, 'h'
    return x


class TestOrdering:
    def test_a_left_operand_is_not_overtaken_by_a_lowering(self):
        """The regression the prefix rule exists for.  Lowering the ternary
        without naming `g(a)` would run `h(b)` first, and both raise."""
        @fp.fpy
        def f(a: fp.Real, b: fp.Real, c: bool) -> fp.Real:
            return _needs_positive_g(a) + (_needs_positive_h(b) if c else 0.0)

        args = [-1.0, -1.0, True]
        with pytest.raises(Exception) as before:
            f(*args)
        with pytest.raises(Exception) as after:
            _apply(f)(*args)
        assert 'g' in str(before.value)
        assert str(after.value) == str(before.value)

    def test_an_index_is_not_overtaken(self):
        @fp.fpy
        def f(xs: list[fp.Real], i: fp.Real, b: fp.Real, c: bool) -> list[fp.Real]:
            xs[_needs_positive_g(i)] = _needs_positive_h(b) if c else 0.0
            return xs

        args = [[1.0, 2.0], -1.0, -1.0, True]
        assert _run(_apply(f), args) == _run(f, args)
        assert _run(f, args)[1] is not None      # it does raise


# ----------------------------------------------------------------------
# The lowerings


class TestLowering:
    def test_a_ternary_becomes_an_if_statement(self):
        out = Hoistable.apply(_ternary.ast)
        assert _count(out, IfExpr) == 0
        assert _count(out, IfStmt) == 1

    def test_nested_ternaries_become_one_ladder_not_a_chain_of_copies(self):
        out = Hoistable.apply(_nested_ternary.ast)
        assert _count(out, IfStmt) == 2
        assert _assigns_to(out, 't') == 4        # one per arm, none a copy

    def test_a_chain_becomes_flat_guarded_statements(self):
        out = Hoistable.apply(_chain.ast)
        assert _count(out, If1Stmt) == 2         # one guard per operand after the first

    def test_a_context_expression_is_lowered_under_real(self):
        """**E-Context** evaluates it under ``REAL``, so what is hoisted out of
        it goes in a ``with fp.REAL:`` block of its own."""
        out = Hoistable.apply(_computed_context.ast)
        assert _count(out, ContextStmt) == 2
        assert out.body.stmts[0].ctx.format() == 'REAL'


class TestRotation:
    def test_the_condition_ends_up_a_name_assigned_twice(self):
        out = Hoistable.apply(_loop.ast)
        loop = _loop_of(out)
        assert isinstance(loop.cond, Var)
        assert _assigns_to(out, 'c') == 2        # before the loop, and at its end

    def test_the_second_copy_is_the_last_statement_of_the_body(self):
        loop = _loop_of(Hoistable.apply(_loop.ast))
        assert loop.body.stmts[-1].target == loop.cond.name

    def test_a_body_that_always_returns_gets_no_second_copy(self):
        """The loop runs at most one iteration, and a statement after the
        ``return`` would be unreachable."""
        @fp.fpy
        def f(x: fp.Real, n: fp.Real) -> fp.Real:
            while x < n:
                return x
            return n

        out = Hoistable.apply(f.ast)
        assert _assigns_to(out, 'c') == 1

    def test_rotation_creates_the_slot_a_ternary_in_the_condition_needs(self):
        """The lowerings compose: a rotated condition is a slot, so the ternary
        in it lowers -- in both copies."""
        out = Hoistable.apply(_loop_with_a_ternary_condition.ast)
        assert _count(out, IfExpr) == 0
        assert _count(out, IfStmt) == 2


# ----------------------------------------------------------------------
# Semantics


class TestSemantics:
    @pytest.mark.parametrize(
        ('f', 'args'), _PROGRAMS, ids=[f.name for f, _a in _PROGRAMS],
    )
    def test_the_interpreter_agrees(self, f, args):
        assert _run(_apply(f), args) == _run(f, args)

    @pytest.mark.parametrize('f', [f for f, _args in _PROGRAMS], ids=lambda f: f.name)
    def test_idempotence(self, f):
        once = Hoistable.apply(f.ast)
        assert Hoistable.apply(once).format() == once.format()


# ----------------------------------------------------------------------
# Ported from `test_anf.py`, which performed these lowerings until `ANF` was
# made to require them instead.  Each covers something the tests above do not.


class TestPortedRotation:
    def test_nested_loops_each_rotate(self):
        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                acc = 0.0
                i = 0.0
                while max(xs) > acc:
                    while min(xs) < i:
                        i = i - 1.0
                    acc = acc + 1.0
                return acc

        out = Hoistable.apply(f.ast)
        outer = _first(out, WhileStmt)
        assert isinstance(outer.cond, Var)
        inner = _first(outer.body, WhileStmt)
        assert isinstance(inner.cond, Var)
        assert inner.cond.name != outer.cond.name
        assert repr(f([1.0, 2.0])) == repr(_apply(f)([1.0, 2.0]))


class TestPortedTernary:
    def test_the_whole_right_hand_side_assigns_the_target_directly(self):
        """No temporary, and so no copy: nothing runs after this pass to remove
        one."""

        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                y = max([x, 0.0]) if x > 0.0 else x
                return y

        branch = _first(Hoistable.apply(f.ast), IfStmt)
        for arm in (branch.ift, branch.iff):
            assign = arm.stmts[-1]
            assert isinstance(assign, Assign)
            assert str(assign.target) == 'y'

    def test_lowered_mid_expression(self):
        """The statement goes before the one that reads it."""

        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                y = 1.0 + (max([x, 0.0]) if x > 0.0 else x)
                return y

        block = Hoistable.apply(f.ast).body.stmts[0].body
        assert isinstance(block.stmts[0], IfStmt)
        assert isinstance(block.stmts[1], Assign)

    def test_a_ladder_of_ternaries_makes_no_intermediate_copy(self):
        """Each arm branches on the same name rather than through a copy."""

        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                y = (
                    max([x, 1.0]) if x > 0.0
                    else (max([x, 2.0]) if x > -5.0 else 3.0)
                )
                return y

        out = Hoistable.apply(f.ast)
        assert _count(out, IfStmt) == 2
        targets = {
            str(stmt.target)
            for _p, stmt in walk_stmts(out) if isinstance(stmt, Assign)
        }
        assert targets == {'y'}

    def test_only_one_arm_is_evaluated(self):
        """`fp.cast` raises where the value is not representable, so an eagerly
        evaluated arm would raise where FPy returns."""

        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                with fp.FP32:
                    y = 0.0 if x > 1e30 else fp.cast(x)
                return y

        assert repr(f(1e300)) == repr(_apply(f)(1e300))


class TestPortedChain:
    def test_or_guards_on_the_negated_accumulator(self):
        @fp.fpy
        def f(a: fp.Real, b: fp.Real) -> bool:
            with fp.FP64:
                y = a > 0.0 or max([b, 1.0]) > 0.0
                return y

        out = Hoistable.apply(f.ast)
        assert _count(out, If1Stmt) == 1
        assert isinstance(_first(out, If1Stmt).cond, Not)
        assert repr(f(-1.0, 2.0)) == repr(_apply(f)(-1.0, 2.0))

    def test_and_guards_on_the_accumulator(self):
        @fp.fpy
        def f(a: fp.Real, b: fp.Real) -> bool:
            with fp.FP64:
                y = a > 0.0 and max([b, 1.0]) > 0.0
                return y

        out = Hoistable.apply(f.ast)
        assert isinstance(_first(out, If1Stmt).cond, Var)
        assert repr(f(1.0, -1.0)) == repr(_apply(f)(1.0, -1.0))

    def test_the_guards_are_flat(self):
        """One guard per operand after the first, none nested inside another:
        an `or` whose accumulator is already true fails every later guard."""

        @fp.fpy
        def f(a: fp.Real, b: fp.Real, c: fp.Real) -> bool:
            with fp.FP64:
                y = a > 0.0 and max([b, 1.0]) > 0.0 and max([c, 1.0]) > 0.0
                return y

        out = Hoistable.apply(f.ast)
        assert _count(out, If1Stmt) == 2
        for _p, stmt in walk_stmts(out):
            if isinstance(stmt, If1Stmt):
                assert _count(stmt.body, If1Stmt) == 0

    def test_the_accumulator_is_the_target(self):
        """No temporary and no copy where the chain is the whole right-hand
        side."""

        @fp.fpy
        def f(a: fp.Real, b: fp.Real) -> bool:
            with fp.FP64:
                y = a > 0.0 or max([b, 1.0]) > 0.0
                return y

        out = Hoistable.apply(f.ast)
        copies = [
            stmt for _p, stmt in walk_stmts(out)
            if isinstance(stmt, Assign) and isinstance(stmt.expr, Var)
        ]
        assert copies == []

    def test_short_circuit_is_preserved(self):
        """`fp.cast` raises where the value is not representable, so an eagerly
        evaluated tail would raise where FPy returns."""

        @fp.fpy
        def f(x: fp.Real) -> bool:
            with fp.FP64:
                with fp.FP32:
                    y = x > 1e30 or fp.cast(x) > 0.0
                return y

        assert _count(Hoistable.apply(f.ast), If1Stmt) == 1
        assert _apply(f)(1e300) is True

    def test_composes_with_rotation(self):
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                y = x
                while y > 0.0 and max([y, 1.0]) > 0.5:
                    y = y - 1.0
                return y

        out = Hoistable.apply(f.ast)
        assert isinstance(_first(out, WhileStmt).cond, Var)
        assert _count(out, If1Stmt) == 2      # one before the loop, one inside
        assert repr(f(3.0)) == repr(_apply(f)(3.0))

    def test_the_accumulator_never_clobbers_an_operand(self):
        """A chain assigns its target before the later operands run, so one that
        *reads* the target must not accumulate into it."""

        @fp.fpy
        def f(b: fp.Real, c: bool, d: bool) -> bool:
            x = c
            x = (b > 0.0) or all([x, d])
            return x

        for args in ((-1.0, True, True), (1.0, False, False), (-1.0, False, True)):
            assert repr(f(*args)) == repr(_apply(f)(*args)), args

    def test_a_pure_chain_is_lowered_too(self):
        """The gate that changed.  `ANF` left a chain of pure comparisons alone,
        so that `ValueClassInfer._implied` could match the `And` and drop a
        runtime check.  A total guarantee cannot make that exception: an operand
        after the first has nowhere to put a statement whether or not it wants
        one today.  The check is still dropped -- `_implied` reads the lowered
        ladder now, see `test_value_class.py::TestALoweredChain`.
        """

        @fp.fpy
        def f(a: fp.Real, b: fp.Real) -> bool:
            with fp.FP64:
                y = a > 0.0 or b > 0.0
                return y

        out = Hoistable.apply(f.ast)
        assert _count(out, If1Stmt) == 1
        assert repr(f(-1.0, 2.0)) == repr(_apply(f)(-1.0, 2.0))
