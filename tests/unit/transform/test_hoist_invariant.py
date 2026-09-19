"""
Regression net for `HoistInvariant` — Phase 1 of `docs/todos/hoist-invariant.md`.

Nothing hoists yet.  These assertions pin what the tree does *today*, so that the
diffs in Phase 3 (the transform) and Phase 5 (its entry into `Simplify`) show
exactly which statements moved.  Every assertion that is meant to flip later
says so on the line above it.

The sweeps include the empty list on purpose: a zero-trip loop is where hoisting
changes *whether* an invariant expression is evaluated at all, and the decision
recorded in the plan is to hoist anyway.
"""

import fpy2 as fp
import fpy2.strategies as st

from fpy2.analysis import DefineUse
from fpy2.ast import Assign, ForStmt, StmtBlock, WhileStmt
from fpy2.ast.visitor import DefaultVisitor
from fpy2.transform.hoist_invariant import _invariants
from fpy2.utils import NamedId

# zero-trip, one-trip, and a spread that straddles the FP16 subnormal and
# overflow boundaries, so a hoist that changed rounding would show up
_VALUES = ([], [1.0], [1.1, 3.7, 1e-5], [65600.0, 1e-8, 2.0, -4.5])


def _text(func, ast) -> str:
    return ' '.join(fp.Function(ast, runtime=func.runtime).format().split())


def _block_names(block: StmtBlock) -> set[str]:
    """Names assigned by a direct child of *block*."""
    return {
        str(s.target) for s in block.stmts
        if isinstance(s, Assign) and isinstance(s.target, NamedId)
    }


def _body_names(ast) -> set[str]:
    """Names assigned directly in some loop body."""
    found: set[str] = set()

    class V(DefaultVisitor):
        def _visit_for(self, stmt: ForStmt, ctx):
            found.update(_block_names(stmt.body))
            super()._visit_for(stmt, ctx)

        def _visit_while(self, stmt: WhileStmt, ctx):
            found.update(_block_names(stmt.body))
            super()._visit_while(stmt, ctx)

    V()._visit_function(ast, None)
    return found


def _agrees(func, ast, values=_VALUES) -> bool:
    """The rewritten function computes what the original computes, bit for
    bit — same value, same flags, same context tag on the result."""
    rewritten = fp.Function(ast, runtime=func.runtime)
    return all(repr(rewritten(v)) == repr(func(v)) for v in values)


def _agrees_by_value(func, ast, values=_VALUES) -> bool:
    """Same, but comparing only the number.

    `rescale_fixed` moves where the rounding happens, so a result can come back
    carrying a different context tag: on a one-element list the source's `sum`
    returns the rounded element still tagged `MPFixedContext`, while the
    rescaled schedule's final unscaling multiply re-tags it `RealContext`.
    Same `exp`, same `c`, same flags.
    """
    rewritten = fp.Function(ast, runtime=func.runtime)
    return all(rewritten(v) == func(v) for v in values)


@fp.fpy(ctx=fp.REAL)
def scaled_sum(xs: list[fp.Real]) -> fp.Real:
    n = len(xs)
    acc = 0.0
    for x in xs:
        c = n + 1
        acc = acc + c * x
    return acc


@fp.fpy(ctx=fp.REAL)
def scaled_while(xs: list[fp.Real]) -> fp.Real:
    n = len(xs)
    acc = 0.0
    i = 0
    while i < n:
        c = n + 1
        acc = acc + c * xs[i]
        i = i + 1
    return acc


class TestTheLoopsThemselves:
    """`c` is invariant in both bodies and nothing moves it."""

    def test_a_for_body_keeps_its_invariant(self):
        # flips in Phase 3: `c` leaves the body
        assert 'c' in _body_names(scaled_sum.ast)

    def test_a_while_body_keeps_its_invariant(self):
        # flips in Phase 3: `c` leaves the body
        assert 'c' in _body_names(scaled_while.ast)


class TestSimplify:
    """`Simplify` has no pass that relocates a statement."""

    def test_it_leaves_a_for_invariant_alone(self):
        out = st.simplify(scaled_sum)
        # flips in Phase 5: `simplify` gains `HoistInvariant`
        assert 'c' in _body_names(out.ast)
        assert _agrees(scaled_sum, out.ast)

    def test_it_leaves_a_while_invariant_alone(self):
        out = st.simplify(scaled_while)
        # flips in Phase 5: `simplify` gains `HoistInvariant`
        assert 'c' in _body_names(out.ast)
        assert _agrees(scaled_while, out.ast)


class TestRescaleFixedOutput:
    """The motivating schedule, and the wart this PR exists to remove."""

    P = 12

    @staticmethod
    @fp.fpy(ctx=fp.REAL)
    def fused_sum(xs: list[fp.Real]) -> fp.Real:
        if all([fp.isfinite(x) for x in xs]):
            e = max([fp.logb(x) for x in xs])
            with fp.MPFixedContext(e - 12, rm=fp.RM.RTZ, enable_neg_zero=False):
                ts = [fp.round(x) for x in xs]
            return sum(ts)
        else:
            with fp.FP32:
                return sum(xs)

    @staticmethod
    def _schedule(func):
        return st.simplify(st.rescale_fixed(st.comp_to_loop(st.fuse(func))))

    def test_the_scale_is_recomputed_every_iteration(self):
        out = self._schedule(self.fused_sum)
        # flips in Phase 3: `_k` is bound once, above the loop
        assert '_k' in _body_names(out.ast)

    def test_the_scale_is_written_as_a_power_of_two(self):
        """What `HoistScale` will look for in PR 2: an invariant factor
        multiplying each element, written into the result list."""
        src = _text(self.fused_sum, self._schedule(self.fused_sum).ast)
        assert '_k = ((e - 12) + 1)' in src
        assert '(2 ** _k)' in src

    def test_the_schedule_preserves_the_source_semantics(self):
        """Not about hoisting — the baseline the later phases are measured
        against.  The empty list is excluded: `max([])` has no value."""
        out = self._schedule(self.fused_sum)
        assert _agrees_by_value(self.fused_sum, out.ast, values=_VALUES[1:])


# ----------------------------------------------------------------------
# Phase 2: the invariance query


def _loops(ast) -> list:
    """Every `for` and `while` in *ast*, outermost first."""
    found = []

    class V(DefaultVisitor):
        def _visit_for(self, stmt: ForStmt, ctx):
            found.append(stmt)
            super()._visit_for(stmt, ctx)

        def _visit_while(self, stmt: WhileStmt, ctx):
            found.append(stmt)
            super()._visit_while(stmt, ctx)

    V()._visit_function(ast, None)
    return found


def _query(func, which: int = 0) -> list[str]:
    """The names `_invariants` says may leave the *which*-th loop of *func*."""
    def_use = DefineUse.analyze(func.ast)
    return [str(s.target) for s in _invariants(_loops(func.ast)[which], def_use)]


def _foreign(x):
    return x


class TestTheQueryFinds:

    def test_an_invariant_in_a_for_body(self):
        assert _query(scaled_sum) == ['c']

    def test_an_invariant_in_a_while_body(self):
        assert _query(scaled_while) == ['c']

    def test_one_round_only(self):
        """`b` reads `a`, which is still inside the loop, so it stays behind
        until `a` has moved.  The caller re-runs the query."""
        @fp.fpy(ctx=fp.REAL)
        def f(xs: list[fp.Real]) -> fp.Real:
            n = len(xs)
            acc = 0.0
            for x in xs:
                a = n + 1
                b = a * 2
                acc = acc + b * x
            return acc

        assert _query(f) == ['a']


class TestTheQueryRefuses:

    def test_a_read_of_the_loop_target(self):
        @fp.fpy(ctx=fp.REAL)
        def f(xs: list[fp.Real]) -> fp.Real:
            acc = 0.0
            for x in xs:
                c = x + 1
                acc = acc + c
            return acc

        assert _query(f) == []

    def test_a_read_of_a_name_the_body_rebinds(self):
        """`m` reaches `c` through the loop's phi, not from before it."""
        @fp.fpy(ctx=fp.REAL)
        def f(xs: list[fp.Real]) -> fp.Real:
            m = 1.0
            acc = 0.0
            for x in xs:
                c = m + 1
                m = m + 1
                acc = acc + c * x
            return acc

        assert _query(f) == []

    def test_a_target_the_body_binds_twice(self):
        @fp.fpy(ctx=fp.REAL)
        def f(xs: list[fp.Real]) -> fp.Real:
            n = len(xs)
            acc = 0.0
            for x in xs:
                c = n + 1
                acc = acc + c * x
                c = n + 2
            return acc

        assert _query(f) == []

    def test_a_target_read_after_the_loop(self):
        """A zero-trip loop would leave the pre-loop value in place; hoisting
        would put the invariant one there instead."""
        @fp.fpy(ctx=fp.REAL)
        def f(xs: list[fp.Real]) -> fp.Real:
            n = len(xs)
            c = 0.0
            acc = 0.0
            for x in xs:
                c = n + 1
                acc = acc + c * x
            return acc + c

        assert _query(f) == []

    def test_a_tuple_target(self):
        @fp.fpy(ctx=fp.REAL)
        def f(xs: list[fp.Real]) -> fp.Real:
            n = len(xs)
            acc = 0.0
            for x in xs:
                a, b = (n + 1, n + 2)
                acc = acc + (a + b) * x
            return acc

        assert _query(f) == []

    def test_a_statement_under_a_nested_with(self):
        """Not a direct child of the body: it would land outside the `with`
        and be rounded differently."""
        @fp.fpy(ctx=fp.REAL)
        def f(xs: list[fp.Real]) -> fp.Real:
            n = len(xs)
            acc = 0.0
            for x in xs:
                with fp.FP32:
                    c = n + 1
                acc = acc + c * x
            return acc

        assert _query(f) == []

    def test_an_impure_expression(self):
        @fp.fpy(ctx=fp.REAL)
        def f(xs: list[fp.Real]) -> fp.Real:
            acc = 0.0
            for x in xs:
                c = _foreign(1.0)
                acc = acc + c * x
            return acc

        assert _query(f) == []


class TestTheQueryOnTheMotivatingExample:

    def test_it_finds_the_scale(self):
        out = TestRescaleFixedOutput._schedule(TestRescaleFixedOutput.fused_sum)
        assert '_k' in _query(out, which=2)
