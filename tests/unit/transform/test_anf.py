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

The pass does not *create* the slots it needs; it requires them, and
:class:`TestPrecondition` covers the refusal.  The lowerings that create them
belong to :class:`~fpy2.transform.Hoistable` and are tested in
``test_hoistable.py``, so a sealed position here only ever holds something that
needed no slot in the first place — which is why :class:`TestNeedsSlot` covers
that predicate directly.

:class:`TestRefusals` covers the residue report, and
``test_anf_profile.py`` pins how large that residue is across the corpus.
"""

import pytest

import fpy2 as fp
from fpy2 import Function
from fpy2.analysis import DefineUse, TypeInfer
from fpy2.ast.fpyast import (
    And,
    Assign,
    Attribute,
    Call,
    Compare,
    ContextStmt,
    Empty,
    Enumerate,
    Expr,
    ForeignVal,
    Fst,
    FuncDef,
    If1Stmt,
    IfExpr,
    IfStmt,
    ListComp,
    ListExpr,
    ListRef,
    ListSlice,
    Not,
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
from fpy2.number import REAL
from fpy2.transform import ANF, Hoistable
from fpy2.transform.error import TransformError
from fpy2.transform.anf import needs_slot
from fpy2.transform.path import sub_exprs, walk_stmts
from fpy2.types import BoolType, RealType

# ----------------------------------------------------------------------
# Helpers

# Restated here rather than imported, so the test states the property
# independently of how the pass computes it.
_ATOM = (Var, ValueExpr, NullaryOp)
_SCALAR = (RealType, BoolType)
"""Types whose values the pass binds to a name; anything else stays inline."""

_SEALS = (IfExpr, And, Or, ListComp)
"""Forms holding a conditionally- or repeatedly-evaluated subexpression."""


def _anf(f) -> Function:
    return Function(ANF.apply(f.ast), runtime=f.runtime)


def _unnamed(func: FuncDef) -> list[Expr]:
    """Scalar-typed subexpressions the pass left in a position it names.

    Descends only where the pass hoists: not into a sealed form, and not into a
    ``while`` condition.  An aggregate-valued subexpression is not a violation —
    naming one would create a place — but its children are still visited, so a
    scalar inside an inline spine still counts.
    """
    du = DefineUse.analyze(func)
    types = TypeInfer.check(func, def_use=du)
    bad: list[Expr] = []

    def descend(e: Expr) -> None:
        if isinstance(e, _SEALS):
            return
        for _field, _i, sub in sub_exprs(e):
            if not isinstance(sub, _ATOM) and isinstance(
                types.by_expr.get(sub), _SCALAR,
            ):
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


def _count(node, kind) -> int:
    """How many *kind* statements are in *node* (a `FuncDef` or a `StmtBlock`)."""
    n = 0

    class _C(DefaultVisitor):
        def _visit_statement(self, stmt, ctx):
            nonlocal n
            if isinstance(stmt, kind):
                n += 1
            super()._visit_statement(stmt, ctx)

    if isinstance(node, FuncDef):
        _C()._visit_function(node, None)
    else:
        _C()._visit_block(node, None)
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
# 1b. Type-directed atomicity


class TestTypeDirectedAtomicity:
    """What is named is decided by type, not by node kind."""

    def test_nested_calls_unfold(self):
        @fp.fpy
        def g(x: fp.Real) -> fp.Real:
            with fp.FP64:
                return x * x

        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                return g(g(x * 2.0)) + 1.0

        out = ANF.apply(f.ast)
        assert _unnamed(out) == []
        # each call now stands alone
        for _p, stmt in walk_stmts(out):
            if isinstance(stmt, Assign):
                assert not isinstance(stmt.expr, Call) or all(
                    isinstance(a, Var) for a in stmt.expr.args
                )

    def test_a_projection_chain_is_named_at_its_scalar(self):
        """The spine stays inline, so no name ever holds a list."""

        @fp.fpy
        def f(xss: list[list[fp.Real]], i: fp.Real) -> fp.Real:
            with fp.FP64:
                return xss[i + 1][0] * 2.0

        out = ANF.apply(f.ast)
        assert _unnamed(out) == []
        named = [
            stmt.expr for _p, stmt in walk_stmts(out)
            if isinstance(stmt, Assign) and isinstance(stmt.expr, ListRef)
        ]
        assert len(named) == 1
        # bound at the outer subscript; the inner one is still nested in it
        assert isinstance(named[0].value, ListRef)

    def test_an_aggregate_valued_expression_is_not_named(self):
        @fp.fpy
        def f(xss: list[list[fp.Real]], i: fp.Real) -> fp.Real:
            with fp.FP64:
                return sum(xss[i + 1])

        out = ANF.apply(f.ast)
        # the list-typed subscript stays inline inside the fold
        assert not [
            stmt for _p, stmt in walk_stmts(out)
            if isinstance(stmt, Assign) and isinstance(stmt.expr, ListRef)
        ]
        assert isinstance(_first(out, ListRef), ListRef)

    def test_a_context_expression_is_not_named(self):
        """`fp.FP64` types as a context, not a scalar."""

        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                return x * x

        out = ANF.apply(f.ast)
        assert isinstance(out.body.stmts[0].ctx, Attribute)

    def test_a_chain_is_named_once(self):
        """A chain is one scalar-typed expression, so it takes one name -- and
        no assignment anywhere is a bare copy of another."""

        @fp.fpy
        def f(a: fp.Real, b: fp.Real) -> list[bool]:
            with fp.FP64:
                y = [a > 0.0 or b > 0.0, True]
                return y

        out = ANF.apply(f.ast)
        elts = _first(out, ListExpr).elts
        assert all(isinstance(e, (Var, ValueExpr)) for e in elts)
        # no assignment is a bare copy of another name
        copies = [
            stmt for _p, stmt in walk_stmts(out)
            if isinstance(stmt, Assign) and isinstance(stmt.expr, Var)
        ]
        assert copies == []


# ----------------------------------------------------------------------
# 2. Sealed positions


class TestContextExpression:
    """**E-Context** evaluates a `with`'s context expression under `REAL`."""

    def test_temporaries_go_in_a_real_block(self):
        """Not the enclosing block, which rounds under something else: a
        constructor's arguments are precisions and bitwidths, and rounding them
        would corrupt the context being built."""

        @fp.fpy
        def f() -> fp.Real:
            es = 2
            nb = 8
            with fp.IEEEContext(es + 2, nb + 2):
                return fp.round(1.0)

        out = ANF.apply(f.ast)
        block = out.body.stmts
        hoist, use = block[2], block[3]
        assert isinstance(hoist, ContextStmt)
        assert isinstance(hoist.ctx, ForeignVal) and hoist.ctx.val is REAL
        assert len(hoist.body.stmts) == 2
        # the `with` that follows now reads the names the REAL block bound
        assert isinstance(use, ContextStmt)
        assert all(isinstance(a, Var) for a in use.ctx.args)
        assert repr(f()) == repr(_anf(f)())

    def test_no_block_where_nothing_is_hoisted(self):
        """A context expression already in normal form gets no preamble."""

        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                return x * x

        out = ANF.apply(f.ast)
        assert len(out.body.stmts) == 1
        assert isinstance(out.body.stmts[0], ContextStmt)


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

    def test_a_pure_bool_chain_is_untouched(self):
        """Nothing in it needs a place, so it stays one expression -- which is
        what a value-class analysis reads to drop a guard."""

        @fp.fpy
        def f(p: bool, q: bool) -> bool:
            y = p and q
            return y

        before = _first(f.ast, And)
        after = _first(ANF.apply(f.ast), And)
        assert len(after.args) == len(before.args)
        assert all(a.is_equiv(b) for a, b in zip(after.args, before.args))

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
        """A *pure* condition: one needing a place is a precondition failure,
        and its semantics are `Hoistable`'s to preserve."""

        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                y = x
                while y > 0.0:
                    y = y - 1.0
                return y

        assert repr(f(3.0)) == repr(_anf(f)(3.0))

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


# ----------------------------------------------------------------------
# 4. The residue


class TestRefusals:
    """What the pass reports it could not normalize."""

    def test_empty_for_a_flattened_program(self):
        @fp.fpy
        def f(a: fp.Real, b: fp.Real) -> fp.Real:
            with fp.FP64:
                y = (a * b) + (a - b)
                return y

        assert ANF.refusals(ANF.apply(f.ast)) == []

    def test_the_check_is_not_vacuous(self):
        """The same three positions, un-normalized, are all reported."""

        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                with fp.FP32:
                    y = x
                    while max([y, 0.0]) > 0.0:
                        y = y - 1.0
                    a = 0.0 if x > 1e30 else fp.cast(x)
                    b = 1.0 if (x > 1e30 or fp.cast(x) > 0.0) else 2.0
                return y + a + b

        why = {reason for _e, reason in ANF.refusals(f.ast)}
        assert why == {
            'a `while` condition is re-evaluated every iteration',
            'a ternary arm is evaluated conditionally',
            'a short-circuited operand may not be evaluated',
        }

    def test_a_comprehension_element_is_reported(self):
        @fp.fpy
        def f(xs: list[fp.Real]) -> list[fp.Real]:
            with fp.FP64:
                return [fp.round(v) for v in xs]

        why = [reason for _e, reason in ANF.refusals(ANF.apply(f.ast))]
        assert "a comprehension's element runs once per iteration" in why

    def test_a_slot_free_comprehension_is_not_reported(self):
        """Arithmetic over the target needs no place, so the element is clean
        even though the pass does not flatten inside it."""

        @fp.fpy
        def f(xs: list[fp.Real], ys: list[fp.Real]) -> list[fp.Real]:
            with fp.FP64:
                return [(v * v) + 1.0 for v in ys]

        assert ANF.refusals(ANF.apply(f.ast)) == []


# ----------------------------------------------------------------------
# 5. The precondition


class TestPrecondition:
    """The pass raises rather than normalize what it cannot.

    Its invariant is that every proper subexpression of a statement is an atom.
    Where a sealed position holds something needing a place, the pass would have
    to emit a statement it has nowhere to put -- so it cannot honour the
    invariant, and says so instead of returning a program that looks normalized
    and is not.  :class:`~fpy2.transform.Hoistable` is what a caller runs to make
    it able to.
    """

    @staticmethod
    def _needs_a_slot_in_a_ternary_arm():
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                with fp.FP32:
                    y = 0.0 if x > 1e30 else fp.cast(x)
                return y
        return f

    def test_a_ternary_arm_is_refused(self):
        f = self._needs_a_slot_in_a_ternary_arm()
        with pytest.raises(TransformError, match='ternary arm'):
            ANF.apply(f.ast)

    def test_a_short_circuited_operand_is_refused(self):
        @fp.fpy
        def f(x: fp.Real) -> bool:
            with fp.FP64:
                with fp.FP32:
                    y = x > 1e30 or fp.cast(x) > 0.0
                return y

        with pytest.raises(TransformError, match='short-circuited'):
            ANF.apply(f.ast)

    def test_a_while_condition_is_refused(self):
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                y = x
                while max([y, 0.0]) > 0.0:
                    y = y - 1.0
                return y

        with pytest.raises(TransformError, match='while'):
            ANF.apply(f.ast)

    def test_the_message_names_the_position_and_the_remedy(self):
        f = self._needs_a_slot_in_a_ternary_arm()
        with pytest.raises(TransformError) as e:
            ANF.apply(f.ast)
        assert 'fp.cast(x)' in str(e.value)
        assert 'Hoistable' in str(e.value)

    def test_a_pure_while_condition_is_accepted(self):
        """The gate is narrow on purpose: it asks what this pass would have to
        name, not whether the program is in hoistable form.  Nothing in `i < n`
        needs a place, so there is nothing to refuse -- even though `Hoistable`
        would rotate the loop."""

        @fp.fpy
        def f(i: fp.Real, n: fp.Real) -> fp.Real:
            with fp.FP64:
                while i < n:
                    i = i + 1.0
                return i

        out = ANF.apply(f.ast)
        assert isinstance(_first(out, WhileStmt).cond, Compare)

    def test_a_ternary_over_atoms_is_accepted(self):
        @fp.fpy
        def f(x: fp.Real, y: fp.Real, c: bool) -> fp.Real:
            with fp.FP64:
                return (x * y) + (x if c else y)

        out = ANF.apply(f.ast)
        assert isinstance(_first(out, IfExpr), IfExpr)

    def test_a_comprehension_is_not_a_precondition_failure(self):
        """It is reported, not refused: the cpp emitter gives the element the
        loop body it generates, so declining to normalize inside one is a shape
        nothing gets wrong."""

        @fp.fpy
        def f(xs: list[fp.Real]) -> list[fp.Real]:
            with fp.FP64:
                return [fp.round(v) for v in xs]

        out = ANF.apply(f.ast)              # does not raise
        why = [reason for _e, reason in ANF.refusals(out)]
        assert why == ["a comprehension's element runs once per iteration"]

    def test_hoistable_first_always_satisfies_it(self):
        """The pairing the cpp pipeline relies on."""

        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                with fp.FP32:
                    y = x
                    while max([y, 0.0]) > 0.0:
                        y = y - 1.0
                    a = 0.0 if x > 1e30 else fp.cast(x)
                    b = 1.0 if (x > 1e30 or fp.cast(x) > 0.0) else 2.0
                return y + a + b

        out = ANF.apply(Hoistable.apply(f.ast))     # does not raise
        assert ANF.refusals(out) == []
        assert repr(f(2.0)) == repr(Function(out, runtime=f.runtime)(2.0))
