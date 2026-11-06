"""
The z3 backend for the digit-bound store, and its default.

One incremental z3 solver per component.  A maximum is found by bisection
over satisfiability checks, or, with ``bisect=False``, by z3's ``Optimize``,
which answers every objective of a query in one check.
"""

import math
from collections.abc import Sequence
from typing import TypeAlias

import z3

from .solver import Constraint, Rel, Status, Term, _ComponentSolver, _Var

__all__ = ['Z3Solver']

_Z3Solver: TypeAlias = 'z3.Solver | z3.Optimize'


class Z3Solver(_ComponentSolver[_Z3Solver]):
    """:class:`~fpy2.analysis.digit_bound.solver.Solver` backed by z3.

    The store is quantifier-free linear integer arithmetic over a handful of
    variables; *timeout_ms* guards against a pathological one rather than an
    expected path.
    """

    bisect: bool
    """Find the maximum by refuting ``objective >= k`` rather than by asking
    z3 to optimize.

    Refutation reuses the plain solver already built for the component,
    where `Optimize` is a second encoding of the same constraints.  Off only
    so `test_both_backends_agree` can check the two against each other.
    """

    _env: dict[_Var, z3.ArithRef]

    _encoded: dict[Term, z3.ArithRef]
    """Each term's z3 form, built once: the same terms recur across one
    store's constraints."""

    _lits: dict[int, z3.BoolRef]
    """Each guard literal's z3 form."""

    def __init__(self, timeout_ms: int = 10_000, bisect: bool = True):
        super().__init__(timeout_ms)
        self.bisect = bisect
        self._env = {}
        self._encoded = {}
        self._lits = {}

    @property
    def _optimizes(self) -> bool:
        return not self.bisect

    def _to_z3(self, term: Term) -> z3.ArithRef:
        cached = self._encoded.get(term)
        if cached is not None:
            return cached
        acc = z3.IntVal(term.const)
        for v, c in term.coeffs:
            z = self._env.get(v)
            if z is None:
                z = self._env[v] = z3.Int(f'{v.name}#{v.index}')
            acc = acc + c * z
        self._encoded[term] = acc
        return acc

    def _lit(self, i: int) -> z3.BoolRef:
        lit = self._lits.get(i)
        if lit is None:
            lit = self._lits[i] = z3.Bool(f'g#{i}')
        return lit

    def _new(self, optimize: bool) -> _Z3Solver:
        s = z3.Optimize() if optimize else z3.Solver()
        s.set('timeout', self.timeout_ms)
        if optimize:
            # each objective's own optimum, not the first's and then the
            # next's under it (`lex`, the default): see :meth:`_maxima`
            s.set(priority='box')
        return s

    def _encode(self, into: _Z3Solver, constraint: Constraint) -> None:
        lhs = self._to_z3(constraint.lhs)
        rhs = [self._to_z3(r) for r in constraint.rhs]
        match constraint.op:
            case Rel.EQ:
                rel = lhs == rhs[0]
            case Rel.LE:
                rel = lhs <= rhs[0]
            case Rel.LE_MAX:
                rel = z3.Or([lhs <= r for r in rhs])
        if constraint.guard:
            rel = z3.Implies(z3.And([self._lit(g) for g in constraint.guard]), rel)
        into.add(rel)

    @staticmethod
    def _status_of(result: z3.CheckSatResult) -> Status:
        if result == z3.sat:
            return Status.SAT
        if result == z3.unsat:
            return Status.UNSAT
        return Status.UNKNOWN

    def _check(
        self, into: _Z3Solver, lits: tuple[int, ...],
        bound: tuple[Term, int] | None = None,
    ) -> Status:
        assumptions = [self._lit(g) for g in lits]
        if bound is not None:
            t, k = bound
            assumptions.insert(0, self._to_z3(t) >= k)
        return self._status_of(into.check(*assumptions))

    def _value(self, into: _Z3Solver, objective: Term) -> int | None:
        found = into.model().eval(self._to_z3(objective), model_completion=True)
        return found.as_long() if z3.is_int_value(found) else None

    def _maxima(
        self, solver: _Z3Solver, objectives: Sequence[Term], lits: tuple[int, ...],
    ) -> list[int | float]:
        """Every objective from one ``Optimize`` check, where z3 optimizes.

        Box priority finds each objective's optimum independently, which is
        what the bounds of one format need: the widest value and the finest
        digit need not occur together.
        """
        if self.bisect:
            return super()._maxima(solver, objectives, lits)
        return self._optimize(solver, objectives, lits)

    def _optimize(
        self, solver: _Z3Solver, objectives: Sequence[Term], lits: tuple[int, ...],
    ) -> list[int | float]:
        """Each objective's maximum from ``Optimize``: one search, run by z3."""
        assert isinstance(solver, z3.Optimize)
        solver.push()
        try:
            handles = [solver.maximize(self._to_z3(t)) for t in objectives]
            status = self._record(self._status_of(
                solver.check(*[self._lit(g) for g in lits])))
            if status is Status.UNSAT:
                return [-math.inf] * len(objectives)
            if status is not Status.SAT:
                return [math.inf] * len(objectives)
            # an unbounded objective comes back as the symbolic `oo`
            return [b.as_long() if z3.is_int_value(b := solver.upper(h)) else math.inf
                    for h in handles]
        finally:
            solver.pop()
