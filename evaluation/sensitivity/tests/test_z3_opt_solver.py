"""
The `z3-opt` backend: each query's maxima from one `Optimize` check.

    pytest -n 8 evaluation/sensitivity/tests
"""

import math

from z3_opt_solver import Z3OptSolver

from fpy2.analysis.digit_bound import DigitBoundStore


def _store(**opts) -> tuple[DigitBoundStore, Z3OptSolver]:
    solver = Z3OptSolver(**opts)
    return DigitBoundStore(solver=solver), solver


def _tradeoff(s: DigitBoundStore):
    """Two variables whose extremes are never attained together."""
    msb, lsb = s.var('msb'), s.var('lsb')
    s.ge(msb, 0)
    s.le(msb, 10)
    s.ge(lsb, -5)
    s.le(lsb, 5)
    s.le(msb - lsb, 8)
    return [msb - lsb + 1, -lsb, msb]


def test_one_check_answers_every_objective():
    s, solver = _store()
    checks = []
    check = solver._check
    solver._check = lambda *a: checks.append(a) or check(*a)
    assert s.maximums(_tradeoff(s)) == [9, 5, 10]
    assert solver.stats.queries == 1
    assert checks == []   # no bisection


def test_unbounded_and_unsatisfiable():
    s, _ = _store()
    x, y = s.var('x'), s.var('y')
    s.le(x, 3)
    s.ge(y, 0)
    assert s.maximums([x, y, -y]) == [3, math.inf, 0]
    s.le(y, -1)
    assert s.maximums([x, y]) == [-math.inf, -math.inf]


def test_a_guard_binds_only_when_assumed():
    s, _ = _store()
    x = s.var('x')
    s.le(x, 10)
    g = s.literal()
    s.le(x, 3, guard=(g,))
    assert s.maximum(x) == 10
    assert s.maximum(x, frozenset({g})) == 3
    assert not s.reaches([(x, 4)], frozenset({g}))


def test_an_optimizer_learns_what_comes_after_it():
    """Constraints stated, and components merged in, after its first query."""
    s, _ = _store()
    x, y, z = s.var('x'), s.var('y'), s.var('z')
    s.le(x, 10)
    assert s.maximum(x) == 10
    s.le(x, 7)
    assert s.maximum(x) == 7
    s.le(y, 2)
    s.le(z, 9)
    s.le(x, y)   # merges `y` into `x`'s component
    assert s.maximums([x, z]) == [2, 9]


def test_a_query_merges_the_components_it_names():
    s, solver = _store(keep_trace=True)
    x, y, z = s.var('x'), s.var('y'), s.var('z')
    s.le(x, 3)
    s.le(y, x)
    s.le(z, 9)
    assert s.maximums([y, z]) == [3, 9]
    s.maximum(z + 1)
    # one search answered the query, so its components stay merged
    assert [q.constraints for q in solver.trace] == [3, 3]
