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


def _first(func: FuncDef, kind):
    """The first *kind* node in *func*, in visit order."""
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

    _F()._visit_function(func, None)
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

    def test_while_condition_is_untouched(self):
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                y = x
                while max([y, 0.0]) > 0.0:
                    y = y - 1.0
                return y

        before = _first(f.ast, WhileStmt)
        out = ANF.apply(f.ast)
        after = _first(out, WhileStmt)
        assert after.cond.is_equiv(before.cond)
        # and nothing was hoisted in front of the loop
        block = out.body.stmts[0].body
        assert isinstance(block.stmts[1], WhileStmt)

    def test_ternary_arms_are_untouched_but_the_condition_is_named(self):
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                y = (x * x) if (x + 1.0) > 0.0 else (x - x)
                return y

        before = _first(f.ast, IfExpr)
        after = _first(ANF.apply(f.ast), IfExpr)
        assert after.ift.is_equiv(before.ift)
        assert after.iff.is_equiv(before.iff)
        # the condition runs whenever the ternary does, so it takes its slot
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
