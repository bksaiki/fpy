"""
Unit tests for the digit-bound solver: its answers, its statistics, and how
a store is given one.
"""

import math
from functools import partial

import pytest

from fpy2.analysis.digit_bound import (
    DigitBoundStore,
    SolverStats,
    Z3Solver,
    solver_factory,
)

from .test_digit_bound import TestFusedSum


def _store() -> DigitBoundStore:
    return DigitBoundStore(solver=Z3Solver())


def _checks(solver) -> list:
    """Every satisfiability check *solver* makes from here on."""
    made: list = []
    check = solver._check
    solver._check = lambda *a: made.append(a) or check(*a)
    return made


class TestAnswers:
    """The answers, on the shapes the store builds."""

    def test_a_guard_binds_only_when_assumed(self):
        s = _store()
        x = s.var('x')
        s.le(x, 10)
        g = s.literal()
        s.le(x, 3, guard=(g,))
        assert s.maximum(x) == 10
        assert s.maximum(x, frozenset({g})) == 3

    def test_a_guarded_max(self):
        """A `max` and a guard on one component, asked both ways around."""
        s = _store()
        x, y = s.var('x'), s.var('y')
        s.le(y, 3)
        s.le_max(x, [y, y + 5])
        g = s.literal()
        s.le(x, 2, guard=(g,))
        assert s.maximum(x) == 8
        assert s.maximum(x, frozenset({g})) == 2
        assert s.maximum(x) == 8      # the literal is released after its query

    def test_a_constant_term(self):
        s = _store()
        a = s.var('a')
        s.le(a, 1)
        assert s.maximum(a - a + 7) == 7

    def test_the_fused_sum(self):
        for F in (13, 24, 35):
            s, v = TestFusedSum._store(F)
            s = _rebuilt(s)
            assert s.prec(v['lp'], v['n'] + 1) == F + 2
            assert s.prec(v['lc'], v['n'] + 1) == F + 1

    def test_a_decision_agrees_with_the_maximum(self):
        s = _store()
        x, y = s.var('x'), s.var('y')
        s.le(y, 3)
        s.le_max(x, [y, y + 5])
        g = s.literal()
        s.le(x, 2, guard=(g,))
        for term, k, lits in ((x, 8, ()), (x, 9, ()), (x, 2, (g,)), (x, 3, (g,)),
                              (x - y, 5, ()), (x - y, 6, ())):
            assert s.reaches([(term, k)], frozenset(lits)) == (
                s.maximum(term, frozenset(lits)) >= k), (term, k, lits)

    def test_a_variable_nothing_constrains_reaches_anything(self):
        """Asked of the backend directly: the store settles a free term
        itself, so only a caller past it meets a component with nothing in
        it."""
        s = _store()
        x = s.var('x')
        assert s._solver.reaches(x, 10**6) is True
        assert s._solver.reaches(x - x + 3, 4) is False

    def test_an_unrelated_large_constant_costs_nothing(self):
        """A component's bounds come from its own constants: one far away
        must not widen another's search, or its box."""
        s = _store()
        x, big = s.var('x'), s.var('big')
        s.le(big, 1 << 50)
        s.ge(big, -(1 << 50))
        s.le(x, 5)
        assert s.maximum(x) == 5


def _rebuilt(store: DigitBoundStore) -> DigitBoundStore:
    """*store*'s constraints, restated into a store over a fresh solver."""
    out = DigitBoundStore(solver=Z3Solver())
    out._vars = store._vars
    for c in store._constraints:
        out._add(c)
    return out


class TestStatistics:
    """The solver counts what it was told and what it was asked."""

    def test_stating_constraints_is_timed(self):
        solver = Z3Solver()
        s = DigitBoundStore(solver=solver)
        x, y = s.var('x'), s.var('y')
        g = s.literal()
        s.le(x, 3)
        s.eq(y, x)
        s.le_max(x, [y, 4], guard=(g,))
        st = solver.stats
        assert st.queries == 0
        assert st.solver_time > 0     # stating them is timed too

    def test_queries_are_counted_and_timed(self):
        solver = Z3Solver()
        s = DigitBoundStore(solver=solver)
        x = s.var('x')
        s.le(x, 3)
        s.maximum(x)
        s.maximums([x + 1, x + 2])
        st = solver.stats
        assert st.queries == 2
        assert st.solver_time > 0
        assert st.degraded == 0

    def test_each_query_is_traced(self):
        solver = Z3Solver(keep_trace=True)
        s = DigitBoundStore(solver=solver)
        x, y, z = s.var('x'), s.var('y'), s.var('z')
        s.le(x, 3)
        s.le(y, x)
        s.le(z, 9)
        s.maximum(y)
        s.maximum(z)
        trace = solver.trace
        # each query sees only its own component: {x, y}, then {z}
        assert [q.constraints for q in trace] == [2, 1]
        assert sum(q.seconds for q in trace) <= solver.stats.solver_time

    def test_a_query_leaves_components_apart(self):
        solver = Z3Solver(keep_trace=True)
        s = DigitBoundStore(solver=solver)
        x, y, z = s.var('x'), s.var('y'), s.var('z')
        s.le(x, 3)
        s.le(y, x)
        s.le(z, 9)
        assert s.maximums([y, z]) == [3, 9]
        s.maximum(z + 1)
        trace = solver.trace
        # the query saw both components, and left them apart
        assert [q.constraints for q in trace] == [3, 1]

    def test_queries_are_traced_only_when_asked(self):
        solver = Z3Solver()
        s = DigitBoundStore(solver=solver)
        x = s.var('x')
        s.le(x, 3)
        s.maximum(x)
        s.reaches([(x, 9)])
        assert solver.stats.queries == 2
        assert solver.trace == []

    def test_a_decision_is_one_query(self):
        solver = Z3Solver()
        s = DigitBoundStore(solver=solver)
        x = s.var('x')
        s.le(x, 3)
        checks = _checks(solver)
        assert not s.reaches([(x, 4)])
        assert solver.stats.queries == 1
        assert len(checks) == 1

    def test_a_cached_answer_does_not_reach_the_solver(self):
        solver = Z3Solver()
        s = DigitBoundStore(solver=solver)
        x = s.var('x')
        s.le(x, 3)
        s.maximum(x)
        s.maximum(x)
        assert solver.stats.queries == 1


class TestStatsSum:

    def test_they_add_field_by_field(self):
        a = SolverStats(solver_time=1.0, queries=2)
        b = SolverStats(solver_time=0.5, queries=1, degraded=1)
        assert sum((a, b), SolverStats()) == SolverStats(
            solver_time=1.5, queries=3, degraded=1)


class TestChoosingABackend:

    def test_z3_by_default(self):
        assert isinstance(DigitBoundStore()._solver, Z3Solver)

    def test_a_block_chooses_and_keeps_its_solvers(self):
        make = partial(Z3Solver, timeout_ms=5)
        with solver_factory(make) as solvers:
            a, b = DigitBoundStore(), DigitBoundStore()
        assert solvers == [a._solver, b._solver]
        assert a._solver.timeout_ms == 5
        # and only inside it
        assert DigitBoundStore()._solver.timeout_ms != 5

    def test_a_handed_solver_wins(self):
        solver = Z3Solver()
        with solver_factory(Z3Solver) as solvers:
            s = DigitBoundStore(solver=solver)
        assert s._solver is solver
        assert solvers == []


def _tradeoff(s: DigitBoundStore):
    """Two variables whose extremes are never attained together: `msb` reaches
    10 only where `lsb >= 2`, and `lsb` reaches -5 only where `msb <= 3`."""
    msb, lsb = s.var('msb'), s.var('lsb')
    s.ge(msb, 0)
    s.le(msb, 10)
    s.ge(lsb, -5)
    s.le(lsb, 5)
    s.le(msb - lsb, 8)
    return [msb - lsb + 1, -lsb, msb]


class TestMaximizeAll:
    """Several objectives asked together, each answered as if alone."""

    def test_each_is_its_own_maximum(self):
        s = _store()
        terms = _tradeoff(s)
        assert s.maximums(terms) == [9, 5, 10]
        alone = _store()
        assert [alone.maximum(t) for t in _tradeoff(alone)] == [9, 5, 10]

    def test_an_unsatisfiable_system_bounds_every_one_below(self):
        s = _store()
        terms = _tradeoff(s)
        s.le(terms[2], -1)
        assert s.maximums(terms) == [-math.inf] * 3

    def test_they_are_one_query(self):
        solver = Z3Solver()
        s = DigitBoundStore(solver=solver)
        s.maximums(_tradeoff(s))
        assert solver.stats.queries == 1

    def test_settled_terms_do_not_reach_the_solver(self):
        solver = Z3Solver()
        s = DigitBoundStore(solver=solver)
        terms = _tradeoff(s)
        s.maximum(terms[2])
        free = s.var('free')
        assert s.maximums([*terms, free, terms[0]]) == [9, 5, 10, math.inf, 9]
        # `msb` alone, then the other two together
        assert solver.stats.queries == 2

    def test_bisection_searches_each_in_turn(self):
        solver = Z3Solver()
        s = DigitBoundStore(solver=solver)
        checks = _checks(solver)
        assert s.maximums(_tradeoff(s)) == [9, 5, 10]
        assert solver.stats.queries == 1
        assert len(checks) > 3   # several checks per objective

    def test_an_unbounded_objective_is_inf(self):
        s = _store()
        x, y = s.var('x'), s.var('y')
        s.le(x, 3)
        s.ge(y, 0)
        assert s.maximums([x, y, -y]) == [3, math.inf, 0]
