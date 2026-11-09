"""
Unit tests for the digit-bound solver backends: that each answers as z3 does,
that each keeps its statistics, and how a store is given one.
"""

import math

import pytest

from fpy2.analysis.digit_bound import (
    BACKENDS,
    DigitBoundStore,
    SolverStats,
    Z3Solver,
    default_solver,
    make_solver,
    solver_backend,
)
from fpy2.analysis.digit_bound.backends import ENV_VAR

from .test_digit_bound import TestBisectionMatchesOptimization, TestFusedSum

BACKEND_PARAMS = list(BACKENDS)


def _store(backend: str) -> DigitBoundStore:
    return DigitBoundStore(solver=make_solver(backend))


def _checks(solver) -> list:
    """Every satisfiability check *solver* makes from here on."""
    made: list = []
    check = solver._check
    solver._check = lambda *a: made.append(a) or check(*a)
    return made


@pytest.mark.parametrize('backend', BACKEND_PARAMS)
class TestEveryBackendAgrees:
    """Each backend gives the answers z3 does, on the shapes the store builds."""

    @pytest.mark.parametrize(
        'name,build,want', TestBisectionMatchesOptimization.CASES,
        ids=[c[0] for c in TestBisectionMatchesOptimization.CASES])
    def test_it_finds_the_maximum(self, backend, name, build, want):
        s = _store(backend)
        x, y = s.var('x'), s.var('y')
        _, obj = build(s, x, y)
        assert s.maximum(obj) == want

    def test_a_guard_binds_only_when_assumed(self, backend):
        s = _store(backend)
        x = s.var('x')
        s.le(x, 10)
        g = s.literal()
        s.le(x, 3, guard=(g,))
        assert s.maximum(x) == 10
        assert s.maximum(x, frozenset({g})) == 3

    def test_a_guarded_max(self, backend):
        """A `max` and a guard on one component, asked both ways around."""
        s = _store(backend)
        x, y = s.var('x'), s.var('y')
        s.le(y, 3)
        s.le_max(x, [y, y + 5])
        g = s.literal()
        s.le(x, 2, guard=(g,))
        assert s.maximum(x) == 8
        assert s.maximum(x, frozenset({g})) == 2
        assert s.maximum(x) == 8      # the literal is released after its query

    def test_a_constant_term(self, backend):
        s = _store(backend)
        a = s.var('a')
        s.le(a, 1)
        assert s.maximum(a - a + 7) == 7

    def test_the_fused_sum(self, backend):
        for F in (13, 24, 35):
            s, v = TestFusedSum._store(F)
            s = _rebuilt(s, backend)
            assert s.prec(v['lp'], v['n'] + 1) == F + 2
            assert s.prec(v['lc'], v['n'] + 1) == F + 1

    def test_a_decision_agrees_with_the_maximum(self, backend):
        s = _store(backend)
        x, y = s.var('x'), s.var('y')
        s.le(y, 3)
        s.le_max(x, [y, y + 5])
        g = s.literal()
        s.le(x, 2, guard=(g,))
        for term, k, lits in ((x, 8, ()), (x, 9, ()), (x, 2, (g,)), (x, 3, (g,)),
                              (x - y, 5, ()), (x - y, 6, ())):
            assert s.reaches([(term, k)], frozenset(lits)) == (
                s.maximum(term, frozenset(lits)) >= k), (term, k, lits)

    def test_a_variable_nothing_constrains_reaches_anything(self, backend):
        """Asked of the backend directly: the store settles a free term
        itself, so only a caller past it meets a component with nothing in
        it."""
        s = _store(backend)
        x = s.var('x')
        assert s._solver.reaches(x, 10**6) is True
        assert s._solver.reaches(x - x + 3, 4) is False

    def test_an_unrelated_large_constant_costs_nothing(self, backend):
        """A component's bounds come from its own constants: one far away
        must not widen another's search, or its box."""
        s = _store(backend)
        x, big = s.var('x'), s.var('big')
        s.le(big, 1 << 50)
        s.ge(big, -(1 << 50))
        s.le(x, 5)
        assert s.maximum(x) == 5


def _rebuilt(store: DigitBoundStore, backend: str) -> DigitBoundStore:
    """*store*'s constraints, restated into a store over *backend*."""
    out = DigitBoundStore(solver=make_solver(backend))
    out._vars = store._vars
    for c in store._constraints:
        out._add(c)
    return out


@pytest.mark.parametrize('backend', BACKEND_PARAMS)
class TestStatistics:
    """Every backend counts what it was told and what it was asked."""

    def test_constraints_are_counted(self, backend):
        solver = make_solver(backend)
        s = DigitBoundStore(solver=solver)
        x, y = s.var('x'), s.var('y')
        g = s.literal()
        s.le(x, 3)
        s.eq(y, x)
        s.le_max(x, [y, 4], guard=(g,))
        st = solver.stats
        assert st.constraints == 3
        assert st.queries == 0
        assert st.solver_time > 0     # stating them is timed too

    def test_queries_are_counted_and_timed(self, backend):
        solver = make_solver(backend)
        s = DigitBoundStore(solver=solver)
        x = s.var('x')
        s.le(x, 3)
        s.maximum(x)
        s.maximums([x + 1, x + 2])
        st = solver.stats
        assert st.queries == 2
        assert st.solver_time > 0
        assert st.degraded == 0

    def test_each_query_is_traced(self, backend):
        solver = make_solver(backend, keep_trace=True)
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

    def test_queries_are_traced_only_when_asked(self, backend):
        solver = make_solver(backend)
        s = DigitBoundStore(solver=solver)
        x = s.var('x')
        s.le(x, 3)
        s.maximum(x)
        s.reaches([(x, 9)])
        assert solver.stats.queries == 2
        assert solver.trace == []

    def test_a_decision_is_one_query(self, backend):
        solver = make_solver(backend)
        s = DigitBoundStore(solver=solver)
        x = s.var('x')
        s.le(x, 3)
        checks = _checks(solver)
        assert not s.reaches([(x, 4)])
        assert solver.stats.queries == 1
        assert len(checks) == 1

    def test_bisect_says_how_it_searches(self, backend):
        """`bisect` is what the evaluation reports as the backend's method:
        bisection finds a maximum by satisfiability checks, an optimizer
        without any."""
        solver = make_solver(backend)
        s = DigitBoundStore(solver=solver)
        x = s.var('x')
        s.le(x, 1000)
        checks = _checks(solver)
        assert s.maximum(x) == 1000
        assert bool(checks) == solver.bisect

    def test_a_cached_answer_does_not_reach_the_solver(self, backend):
        solver = make_solver(backend)
        s = DigitBoundStore(solver=solver)
        x = s.var('x')
        s.le(x, 3)
        s.maximum(x)
        s.maximum(x)
        assert solver.stats.queries == 1


class TestStatsSum:

    def test_they_add_field_by_field(self):
        a = SolverStats(solver_time=1.0, queries=2, constraints=3)
        b = SolverStats(solver_time=0.5, queries=1, degraded=1)
        assert sum((a, b), SolverStats()) == SolverStats(
            solver_time=1.5, constraints=3, queries=3, degraded=1)


class TestChoosingABackend:

    def test_z3_by_default(self, monkeypatch):
        monkeypatch.delenv(ENV_VAR, raising=False)
        assert isinstance(DigitBoundStore()._solver, Z3Solver)

    @pytest.mark.parametrize('backend', BACKEND_PARAMS)
    def test_the_environment_names_the_default(self, monkeypatch, backend):
        monkeypatch.setenv(ENV_VAR, backend)
        assert type(default_solver()) is type(make_solver(backend))

    def test_z3_opt_optimizes(self):
        solver = make_solver('z3-opt')
        assert isinstance(solver, Z3Solver) and solver.bisect is False

    @pytest.mark.parametrize('backend', BACKEND_PARAMS)
    def test_a_block_chooses_and_keeps_its_solvers(self, monkeypatch, backend):
        monkeypatch.delenv(ENV_VAR, raising=False)
        with solver_backend(backend) as solvers:
            a, b = DigitBoundStore(), DigitBoundStore()
        assert [a._solver, b._solver] == solvers
        assert type(solvers[0]) is type(make_solver(backend))
        # and only inside it
        assert isinstance(DigitBoundStore()._solver, Z3Solver)

    def test_a_block_overrides_the_environment(self, monkeypatch):
        monkeypatch.setenv(ENV_VAR, 'no-such-backend')
        with solver_backend('z3') as solvers:
            DigitBoundStore()
        assert len(solvers) == 1

    def test_options_reach_the_backend(self):
        with solver_backend('z3', bisect=False, timeout_ms=5, keep_trace=True) as solvers:
            DigitBoundStore()
        assert solvers[0].bisect is False
        assert solvers[0].timeout_ms == 5
        assert solvers[0].keep_trace is True

    def test_an_unknown_backend_is_refused(self):
        with pytest.raises(ValueError, match='unknown solver backend'):
            make_solver('no-such-backend')

    def test_a_handed_solver_wins(self):
        solver = Z3Solver()
        with solver_backend('z3') as solvers:
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


@pytest.mark.parametrize('backend', BACKEND_PARAMS)
class TestMaximizeAll:
    """Several objectives asked together, each answered as if alone."""

    def test_each_is_its_own_maximum(self, backend):
        s = _store(backend)
        terms = _tradeoff(s)
        assert s.maximums(terms) == [9, 5, 10]
        alone = _store(backend)
        assert [alone.maximum(t) for t in _tradeoff(alone)] == [9, 5, 10]

    def test_an_unsatisfiable_system_bounds_every_one_below(self, backend):
        s = _store(backend)
        terms = _tradeoff(s)
        s.le(terms[2], -1)
        assert s.maximums(terms) == [-math.inf] * 3

    def test_they_are_one_query(self, backend):
        solver = make_solver(backend)
        s = DigitBoundStore(solver=solver)
        s.maximums(_tradeoff(s))
        assert solver.stats.queries == 1

    def test_settled_terms_do_not_reach_the_solver(self, backend):
        solver = make_solver(backend)
        s = DigitBoundStore(solver=solver)
        terms = _tradeoff(s)
        s.maximum(terms[2])
        free = s.var('free')
        assert s.maximums([*terms, free, terms[0]]) == [9, 5, 10, math.inf, 9]
        # `msb` alone, then the other two together
        assert solver.stats.queries == 2


class TestZ3OptimizesTogether:
    """z3's optimizer answers several objectives in one check."""

    def test_one_query_answers_them_all(self):
        solver = make_solver('z3-opt')
        s = DigitBoundStore(solver=solver)
        assert s.maximums(_tradeoff(s)) == [9, 5, 10]
        assert solver.stats.queries == 1

    def test_bisection_searches_each_in_turn(self):
        solver = make_solver('z3')
        s = DigitBoundStore(solver=solver)
        checks = _checks(solver)
        assert s.maximums(_tradeoff(s)) == [9, 5, 10]
        assert solver.stats.queries == 1
        assert len(checks) > 3   # several checks per objective

    def test_an_unbounded_objective_is_inf(self):
        s = _store('z3-opt')
        x, y = s.var('x'), s.var('y')
        s.le(x, 3)
        s.ge(y, 0)
        assert s.maximums([x, y, -y]) == [3, math.inf, 0]
