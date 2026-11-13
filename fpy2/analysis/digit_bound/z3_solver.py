"""
The z3 backend for the digit-bound store, and its default.

One incremental z3 solver per component.  A maximum is found by bisection
over satisfiability checks.
"""

import z3

from .solver import Constraint, Rel, Status, Term, _ComponentSolver, _Var

__all__ = ['Z3Solver']

class Z3Solver(_ComponentSolver[z3.Solver]):
    """:class:`~fpy2.analysis.digit_bound.solver.Solver` backed by z3.

    The store is quantifier-free linear integer arithmetic over a handful of
    variables; *timeout_ms* guards against a pathological one rather than an
    expected path.
    """

    _env: dict[_Var, z3.ArithRef]

    _encoded: dict[Term, z3.ArithRef]
    """Each term's z3 form, built once: the same terms recur across one
    store's constraints."""

    _lits: dict[int, z3.BoolRef]
    """Each guard literal's z3 form."""

    def __init__(self, timeout_ms: int = 10_000, keep_trace: bool = False):
        super().__init__(timeout_ms, keep_trace)
        self._env = {}
        self._encoded = {}
        self._lits = {}

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

    def _new(self) -> z3.Solver:
        s = z3.Solver()
        s.set('timeout', self.timeout_ms)
        return s

    def _encode(self, into: z3.Solver, constraint: Constraint) -> None:
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
        self, into: z3.Solver, lits: tuple[int, ...],
        bound: tuple[Term, int] | None = None,
    ) -> Status:
        assumptions = [self._lit(g) for g in lits]
        if bound is not None:
            t, k = bound
            assumptions.insert(0, self._to_z3(t) >= k)
        return self._status_of(into.check(*assumptions))

    def _value(self, into: z3.Solver, objective: Term) -> int | None:
        found = into.model().eval(self._to_z3(objective), model_completion=True)
        return found.as_long() if z3.is_int_value(found) else None
