"""
Unit tests for the :class:`fpy2.transform.ANF` transform.

Fresh names come from ``Gensym``, so a hand-written golden AST is brittle.  These
tests assert, as ``test_comp_to_loop.py`` does:

1. **Structural shape** — after the pass, every proper subexpression in a
   position it names is an atom.
2. **Sealed positions** — a ``while`` condition, an ``IfExpr`` arm, an
   ``and`` / ``or`` tail and a comprehension keep their nesting, since hoisting
   one out would evaluate it on a path FPy does not take.
3. **Semantic equivalence** through the interpreter, and idempotence.

A ``while`` condition is the one sealed position with a lowering — the loop is
rotated — so it has parts of its own: :class:`TestNeedsSlot` for the predicate
that decides whether to rotate, and :class:`TestRotation` for the rewrite.  A
ternary is the other, becoming an ``IfStmt``: :class:`TestTernaryLowering`.
"""

import fpy2 as fp
from fpy2 import Function
from fpy2.ast.fpyast import (
    And,
    Assign,
    Attribute,
    Call,
    Empty,
    Enumerate,
    Expr,
    Fst,
    FuncDef,
    IfExpr,
    IfStmt,
    ListComp,
    ListExpr,
    ListRef,
    ListSlice,
    NullaryOp,
    Or,
    Range1,
    Range2,
    Range3,
    Snd,
    TupleExpr,
    ValueExpr,
    Var,
    WhileStmt,
    Zip,
)
from fpy2.ast.visitor import DefaultVisitor
from fpy2.transform import ANF
from fpy2.transform.anf import needs_slot
from fpy2.transform.path import sub_exprs, walk_stmts

# ----------------------------------------------------------------------
# Helpers

# Restated here rather than imported, so the test states the property
# independently of how the pass computes it.
_ATOM = (Var, ValueExpr, NullaryOp)
_OPAQUE = (
    ListExpr, TupleExpr, ListComp, ListSlice, Empty, Zip, Enumerate,
    Range1, Range2, Range3, Call, Fst, Snd, ListRef, IfExpr, Attribute,
)
"""Forms the pass does not name because each is or may hold an aggregate."""

_SEALS = (IfExpr, And, Or, ListComp)
"""Forms holding a conditionally- or repeatedly-evaluated subexpression."""


def _anf(f) -> Function:
    return Function(ANF.apply(f.ast), runtime=f.runtime)


def _unnamed(func: FuncDef) -> list[Expr]:
    """Non-atomic subexpressions the pass left in a position it names.

    Descends only where the pass hoists: not into a sealed form, and not into a
    ``while`` condition.  An opaque form is not itself a violation — the pass
    declines to name one — but its children are still visited.
    """
    bad: list[Expr] = []

    def descend(e: Expr) -> None:
        if isinstance(e, _SEALS):
            return
        for _field, _i, sub in sub_exprs(e):
            if not isinstance(sub, _ATOM + _OPAQUE):
                bad.append(sub)
            descend(sub)

    for _path, stmt in walk_stmts(func):
        for field, _i, e in sub_exprs(stmt):
            if isinstance(stmt, WhileStmt) and field == 'cond':
                continue
            descend(e)
    return bad


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


def _count(func: FuncDef, kind) -> int:
    n = 0
    for _path, stmt in walk_stmts(func):
        if isinstance(stmt, kind):
            n += 1
    return n


# ----------------------------------------------------------------------
# 1. Structural shape


class TestFlattening:
    """Every proper subexpression the pass names becomes an atom."""

    def test_binary_tree(self):
        @fp.fpy
        def f(a: fp.Real, b: fp.Real, c: fp.Real, d: fp.Real) -> fp.Real:
            with fp.FP64:
                y = (a * b) + (c * d)
                return y

        out = ANF.apply(f.ast)
        assert _unnamed(out) == []
        # two operand bindings joined the original assign
        assert _count(out, Assign) == 3

    def test_return_operands(self):
        @fp.fpy
        def f(a: fp.Real, b: fp.Real) -> fp.Real:
            with fp.FP64:
                return (a * b) + (a - b)

        out = ANF.apply(f.ast)
        assert _unnamed(out) == []
        assert _count(out, Assign) == 2

    def test_indexed_assign_index_and_value(self):
        @fp.fpy
        def f(xs: list[fp.Real], i: fp.Real) -> list[fp.Real]:
            with fp.FP64:
                xs[i + 1] = (i * i) + 1.0
                return xs

        out = ANF.apply(f.ast)
        assert _unnamed(out) == []

    def test_for_iterable_and_body(self):
        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                acc = 0.0
                for v in xs:
                    acc = acc + (v * v)
                return acc

        out = ANF.apply(f.ast)
        assert _unnamed(out) == []

    def test_assert_test(self):
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                assert (x * x) > 0.0
                return x

        out = ANF.apply(f.ast)
        assert _unnamed(out) == []

    def test_no_function_level_context(self):
        """A program with no ``with`` block still flattens."""

        @fp.fpy
        def f(a: fp.Real, b: fp.Real) -> fp.Real:
            return (a * b) + (a - b)

        out = ANF.apply(f.ast)
        assert _unnamed(out) == []

    def test_an_aggregate_is_not_named_but_its_elements_are(self):
        """Naming a list would give it a place of its own; its elements are
        scalars and are named."""

        @fp.fpy
        def f(a: fp.Real, b: fp.Real) -> list[fp.Real]:
            with fp.FP64:
                y = [a * b, a + b]
                return y

        out = ANF.apply(f.ast)
        assert _unnamed(out) == []
        # the two elements, bound before the assign that still holds the list
        assert _count(out, Assign) == 3
        assert isinstance(_first(out, ListExpr), ListExpr)
        assert all(isinstance(e, Var) for e in _first(out, ListExpr).elts)


class TestContextBoundary:
    """A temporary never leaves the block whose statement needs it."""

    def test_temp_stays_inside_the_with(self):
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                y = x + 1.0
                with fp.FP32:
                    z = (y * y) + (x * x)
                return z

        out = ANF.apply(f.ast)
        assert _unnamed(out) == []
        # the outer block keeps its two statements; both new bindings are inner
        outer = out.body.stmts[0]
        assert len(outer.body.stmts) == 3          # y = ..., with ..., return z
        inner = outer.body.stmts[1]
        assert len(inner.body.stmts) == 3          # two temps, then z = ...


# ----------------------------------------------------------------------
# 2. Sealed positions


class TestSealedPositions:
    """A conditionally- or repeatedly-evaluated position keeps its nesting."""

    def test_a_pure_while_condition_is_untouched(self):
        """Nothing in it needs a place, so it stays an expression and the loop
        is not rotated."""

        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                y = x
                while (y * y) > 1.0:
                    y = y - 1.0
                return y

        before = _first(f.ast, WhileStmt)
        out = ANF.apply(f.ast)
        after = _first(out, WhileStmt)
        assert after.cond.is_equiv(before.cond)
        # nothing was hoisted in front of the loop either
        block = out.body.stmts[0].body
        assert isinstance(block.stmts[1], WhileStmt)

    def test_a_ternary_over_atoms_is_untouched(self):
        """`x1 if c else x2` is already in normal form; only the condition,
        which runs whenever the ternary does, takes a slot."""

        @fp.fpy
        def f(a: fp.Real, b: fp.Real, x: fp.Real) -> fp.Real:
            with fp.FP64:
                y = a if (x + 1.0) > 0.0 else b
                return y

        before = _first(f.ast, IfExpr)
        after = _first(ANF.apply(f.ast), IfExpr)
        assert after.ift.is_equiv(before.ift)
        assert after.iff.is_equiv(before.iff)
        assert isinstance(after.cond, Var)

    def test_bool_tail_is_untouched_but_the_head_is_named(self):
        @fp.fpy
        def f(x: fp.Real) -> bool:
            with fp.FP64:
                y = (x * x) > 1.0 or (x + x) < 0.0
                return y

        before = _first(f.ast, Or)
        after = _first(ANF.apply(f.ast), Or)
        assert isinstance(after.args[0], Var)
        assert after.args[1].is_equiv(before.args[1])

    def test_comprehension_is_untouched(self):
        @fp.fpy
        def f(xs: list[fp.Real]) -> list[fp.Real]:
            with fp.FP64:
                y = [(v * v) + 1.0 for v in xs]
                return y

        before = _first(f.ast, ListComp)
        after = _first(ANF.apply(f.ast), ListComp)
        assert after.elt.is_equiv(before.elt)

    def test_a_sealed_position_seals_what_it_contains(self):
        """An `IfExpr` inside a `while` condition is sealed twice over: its own
        condition is not hoisted either."""

        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                y = x
                while ((x * x) if x > 0.0 else (x + x)) > y:
                    y = y + 1.0
                return y

        before = _first(f.ast, WhileStmt)
        after = _first(ANF.apply(f.ast), WhileStmt)
        assert after.cond.is_equiv(before.cond)


# ----------------------------------------------------------------------
# 2b. `while` rotation


def _while_cond(f) -> Expr:
    return _first(f.ast, WhileStmt).cond


class TestNeedsSlot:
    """What the predicate deciding a rotation admits."""

    def test_arithmetic_and_comparison_are_slot_free(self):
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                y = x
                while ((y * y) + 1.0) > 1.0:
                    y = y - 1.0
                return y

        assert not needs_slot(_while_cond(f))

    def test_a_bool_chain_is_slot_free(self):
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                y = x
                while y > 0.0 and (y * y) > 1.0:
                    y = y - 1.0
                return y

        assert not needs_slot(_while_cond(f))

    def test_len_is_slot_free(self):
        """A boundary case: `len` reads a size and allocates nothing."""

        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                y = 0.0
                while len(xs) > y:
                    y = y + 1.0
                return y

        assert not needs_slot(_while_cond(f))

    def test_a_fold_needs_a_slot(self):
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                y = x
                while max([y, 0.0]) > 0.0:
                    y = y - 1.0
                return y

        assert needs_slot(_while_cond(f))

    def test_a_rounding_needs_a_slot(self):
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                y = x
                while fp.round(y) > 0.0:
                    y = y - 1.0
                return y

        assert needs_slot(_while_cond(f))

    def test_a_subscript_needs_a_slot(self):
        """Not in the slot-free whitelist, so it needs one by default."""

        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                y = 0.0
                while xs[0] > y:
                    y = y + 1.0
                return y

        assert needs_slot(_while_cond(f))


class TestRotation:
    """`c = cond; while c: body; c = cond` -- FPy's own evaluation order."""

    def test_the_condition_becomes_a_name(self):
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                y = x
                while max([y, 0.0]) > 0.0:
                    y = y - 1.0
                return y

        out = ANF.apply(f.ast)
        loop = _first(out, WhileStmt)
        assert isinstance(loop.cond, Var)
        # bound in the statement just before the loop, and again at the end of
        # the body -- the two slots the condition's own evaluations sit in
        block = out.body.stmts[0].body
        i = block.stmts.index(loop)
        pre, tail = block.stmts[i - 1], loop.body.stmts[-1]
        assert isinstance(pre, Assign) and isinstance(tail, Assign)
        assert pre.target == tail.target == loop.cond.name

    def test_a_body_that_always_returns_gets_no_second_copy(self):
        """The loop runs at most one iteration, so the condition is evaluated
        exactly once -- and a statement after the `return` is unreachable."""

        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                y = x
                while max([y, 0.0]) > 0.0:
                    return y - 1.0
                return y

        out = ANF.apply(f.ast)
        loop = _first(out, WhileStmt)
        assert len(loop.body.stmts) == 1
        assert repr(f(3.0)) == repr(_anf(f)(3.0))

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

        out = ANF.apply(f.ast)
        outer = _first(out, WhileStmt)
        assert isinstance(outer.cond, Var)
        inner = _first(outer.body, WhileStmt)
        assert isinstance(inner.cond, Var)
        assert inner.cond.name != outer.cond.name
        assert repr(f([1.0, 2.0])) == repr(_anf(f)([1.0, 2.0]))

    def test_rotation_is_idempotent(self):
        """After rotating, the condition is a name, which needs no slot."""

        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                y = x
                while max([y, 0.0]) > 0.0:
                    y = y - 1.0
                return y

        once = ANF.apply(f.ast)
        assert ANF.apply(once).format() == once.format()


# ----------------------------------------------------------------------
# 2c. Ternary lowering


class TestTernaryLowering:
    """An arm that needs a place makes the ternary an ``IfStmt``."""

    def test_lowered_when_an_arm_is_not_an_atom(self):
        """Even pure arithmetic: an `IfStmt` flattens the arms for free, so
        there is nothing to weigh against doing it."""

        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                y = (x * x) if x > 0.0 else (x + x)
                return y

        out = ANF.apply(f.ast)
        assert _count(out, IfStmt) == 1
        arms = [_first(out, IfStmt).ift, _first(out, IfStmt).iff]
        for arm in arms:
            assign = arm.stmts[-1]
            assert isinstance(assign, Assign) and str(assign.target) == 'y'
        assert repr(f(2.0)) == repr(_anf(f)(2.0))

    def test_lowered_when_an_arm_needs_a_slot(self):
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                y = max([x, 0.0]) if x > 0.0 else x
                return y

        out = ANF.apply(f.ast)
        assert _count(out, IfStmt) == 1
        assert _unnamed(out) == []
        # the ternary is gone entirely
        class _NoTernary(DefaultVisitor):
            def _visit_if_expr(self, e, ctx):
                raise AssertionError('a ternary survived')

        _NoTernary()._visit_function(out, None)

    def test_the_whole_right_hand_side_assigns_the_target_directly(self):
        """No temporary, and so no copy: nothing runs after this pass to remove
        one."""

        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                y = max([x, 0.0]) if x > 0.0 else x
                return y

        out = ANF.apply(f.ast)
        branch = _first(out, IfStmt)
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

        out = ANF.apply(f.ast)
        block = out.body.stmts[0].body
        assert isinstance(block.stmts[0], IfStmt)
        assert isinstance(block.stmts[1], Assign)
        assert _unnamed(out) == []

    def test_a_chain_becomes_one_ladder(self):
        """Each arm branches on the same name rather than through a copy."""

        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                y = (
                    max([x, 1.0]) if x > 0.0
                    else (max([x, 2.0]) if x > -5.0 else 3.0)
                )
                return y

        out = ANF.apply(f.ast)
        assert _count(out, IfStmt) == 2
        # one binding per arm, all of `y`: no intermediate copy anywhere
        targets = {
            str(stmt.target)
            for _p, stmt in walk_stmts(out) if isinstance(stmt, Assign)
        }
        assert targets == {'y'}

    def test_composes_with_rotation(self):
        """A rotated condition is in a slot, so a ternary inside it lowers —
        once before the loop and once in the body."""

        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                y = x
                while (max([y, 0.0]) if y > 0.0 else 0.0) > 0.5:
                    y = y - 1.0
                return y

        out = ANF.apply(f.ast)
        assert _count(out, IfStmt) == 2
        assert isinstance(_first(out, WhileStmt).cond, Var)
        assert repr(f(3.0)) == repr(_anf(f)(3.0))

    def test_not_lowered_in_a_position_with_no_slot(self):
        """Inside an `or` tail there is nowhere to put the statement, so the
        ternary stays one."""

        @fp.fpy
        def f(x: fp.Real) -> bool:
            with fp.FP64:
                y = x > 100.0 or (max([x, 0.0]) if x > 0.0 else x) > 1.0
                return y

        out = ANF.apply(f.ast)
        assert _count(out, IfStmt) == 0
        assert isinstance(_first(out, IfExpr), IfExpr)

    def test_idempotent(self):
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                y = max([x, 0.0]) if x > 0.0 else x
                return y

        once = ANF.apply(f.ast)
        assert ANF.apply(once).format() == once.format()


# ----------------------------------------------------------------------
# 3. Semantics


class TestSemantics:
    """The pass changes no value, and a second application changes nothing."""

    def test_arithmetic(self):
        @fp.fpy
        def f(a: fp.Real, b: fp.Real) -> fp.Real:
            with fp.FP64:
                return (a * b) + (a - b) * (a + b)

        assert repr(f(2.0, 3.0)) == repr(_anf(f)(2.0, 3.0))

    def test_while_loop(self):
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                y = x
                while max([y, 0.0]) > 0.0:
                    y = y - 1.0
                return y

        assert repr(f(3.0)) == repr(_anf(f)(3.0))

    def test_short_circuit_order_is_preserved(self):
        """The tail must not be evaluated when the head decides it.

        ``fp.cast`` raises where the value is not representable, so an
        eagerly-evaluated tail would raise where FPy returns.
        """

        @fp.fpy
        def f(x: fp.Real) -> bool:
            with fp.FP64:
                with fp.FP32:
                    y = x > 1e30 or fp.cast(x) > 0.0
                return y

        assert f(1e300) is True
        assert _anf(f)(1e300) is True

    def test_ternary_takes_only_one_arm(self):
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                with fp.FP32:
                    y = 0.0 if x > 1e30 else fp.cast(x)
                return y

        assert repr(f(1e300)) == repr(_anf(f)(1e300))

    def test_loop_and_comprehension(self):
        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                ys = [(v * v) + 1.0 for v in xs]
                acc = 0.0
                for v in ys:
                    acc = acc + (v * v)
                return acc

        assert repr(f([1.0, 2.0, 3.0])) == repr(_anf(f)([1.0, 2.0, 3.0]))

    def test_nested_contexts(self):
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                y = x + 1.0
                with fp.FP32:
                    z = (y * y) + (x * x)
                return z

        assert repr(f(2.0)) == repr(_anf(f)(2.0))

    def test_idempotent(self):
        @fp.fpy
        def f(a: fp.Real, b: fp.Real) -> fp.Real:
            with fp.FP64:
                y = (a * b) + (a - b)
                for v in [a, b]:
                    y = y + (v * v)
                return y

        once = ANF.apply(f.ast)
        twice = ANF.apply(once)
        assert twice.format() == once.format()
