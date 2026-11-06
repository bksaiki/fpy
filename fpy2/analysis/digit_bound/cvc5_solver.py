"""
A cvc5 backend for the digit-bound store, for comparison with z3.

The same encoding as :class:`~fpy2.analysis.digit_bound.z3_solver.Z3Solver`,
through cvc5's base API.  cvc5 does not optimize, so every maximum is found
by bisection, as z3 finds it by default.  Needs the optional ``cvc5``
package.
"""

import cvc5
from cvc5 import Kind

from .solver import Constraint, Rel, Status, Term, _ComponentSolver, _Var

__all__ = ['Cvc5Solver']


class Cvc5Solver(_ComponentSolver[cvc5.Solver]):
    """:class:`~fpy2.analysis.digit_bound.solver.Solver` backed by cvc5.

    One incremental cvc5 solver per component, in ``QF_LIA``, over one term
    manager.  *timeout_ms* bounds each check (``tlimit-per``).
    """

    _tm: cvc5.TermManager

    _env: dict[_Var, cvc5.Term]

    _encoded: dict[Term, cvc5.Term]
    """Each term's cvc5 form, built once."""

    _lits: dict[int, cvc5.Term]
    """Each guard literal's cvc5 form."""

    def __init__(self, timeout_ms: int = 10_000):
        super().__init__(timeout_ms)
        self._tm = cvc5.TermManager()
        self._env = {}
        self._encoded = {}
        self._lits = {}

    def _int(self, k: int) -> cvc5.Term:
        # `mkInteger` takes a C int; its string form takes any integer
        return self._tm.mkInteger(str(k))

    def _app(self, kind: Kind, args: list[cvc5.Term]) -> cvc5.Term:
        """*kind* over *args*, or the one argument alone: cvc5's n-ary kinds
        take at least two."""
        return args[0] if len(args) == 1 else self._tm.mkTerm(kind, *args)

    def _to_cvc5(self, term: Term) -> cvc5.Term:
        cached = self._encoded.get(term)
        if cached is not None:
            return cached
        parts = []
        for v, c in term.coeffs:
            x = self._env.get(v)
            if x is None:
                x = self._env[v] = self._tm.mkConst(
                    self._tm.getIntegerSort(), f'{v.name}#{v.index}')
            parts.append(x if c == 1 else self._tm.mkTerm(Kind.MULT, self._int(c), x))
        if term.const or not parts:
            parts.append(self._int(term.const))
        out = self._encoded[term] = self._app(Kind.ADD, parts)
        return out

    def _lit(self, i: int) -> cvc5.Term:
        lit = self._lits.get(i)
        if lit is None:
            lit = self._lits[i] = self._tm.mkConst(self._tm.getBooleanSort(), f'g#{i}')
        return lit

    def _new(self, optimize: bool) -> cvc5.Solver:
        s = cvc5.Solver(self._tm)
        s.setOption('incremental', 'true')
        s.setOption('produce-models', 'true')
        s.setOption('tlimit-per', str(self.timeout_ms))
        s.setLogic('QF_LIA')
        return s

    def _encode(self, into: cvc5.Solver, constraint: Constraint) -> None:
        lhs = self._to_cvc5(constraint.lhs)
        rhs = [self._to_cvc5(r) for r in constraint.rhs]
        tm = self._tm
        match constraint.op:
            case Rel.EQ:
                rel = tm.mkTerm(Kind.EQUAL, lhs, rhs[0])
            case Rel.LE:
                rel = tm.mkTerm(Kind.LEQ, lhs, rhs[0])
            case Rel.LE_MAX:
                rel = self._app(Kind.OR, [tm.mkTerm(Kind.LEQ, lhs, r) for r in rhs])
        if constraint.guard:
            guard = self._app(Kind.AND, [self._lit(g) for g in constraint.guard])
            rel = tm.mkTerm(Kind.IMPLIES, guard, rel)
        into.assertFormula(rel)

    def _check(
        self, into: cvc5.Solver, lits: tuple[int, ...],
        bound: tuple[Term, int] | None = None,
    ) -> Status:
        assumptions = [self._lit(g) for g in lits]
        if bound is not None:
            t, k = bound
            assumptions.insert(0, self._tm.mkTerm(Kind.GEQ, self._to_cvc5(t), self._int(k)))
        result = into.checkSatAssuming(*assumptions)
        if result.isSat():
            return Status.SAT
        if result.isUnsat():
            return Status.UNSAT
        return Status.UNKNOWN

    def _value(self, into: cvc5.Solver, objective: Term) -> int | None:
        found = into.getValue(self._to_cvc5(objective))
        return found.getIntegerValue() if found.isIntegerValue() else None
