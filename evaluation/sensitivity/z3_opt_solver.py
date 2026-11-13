"""
The ``z3-opt`` backend: z3 finding each query's maxima with ``Optimize``, in
one search, rather than by bisection.
"""

import math
from collections.abc import Sequence

import z3

from fpy2.analysis.digit_bound import Z3Solver
from fpy2.analysis.digit_bound.solver import Status, Term

__all__ = ['Z3OptSolver']


class Z3OptSolver(Z3Solver):
    """:class:`~fpy2.analysis.digit_bound.Z3Solver` answering each query's
    maxima with one ``Optimize`` check.  A decision needs no optimizer, so
    :meth:`reaches` still asks the component's plain solver."""

    _opt: dict[int, tuple[z3.Optimize, int]]
    """Each component's optimizer, by root, and how many of the component's
    constraints it holds.  A component's constraints are only appended to,
    so the rest are stated just before its next check."""

    def __init__(self, timeout_ms: int = 10_000, keep_trace: bool = False):
        super().__init__(timeout_ms, keep_trace)
        self._opt = {}

    def maximize(
        self, objectives: Sequence[Term], assuming: frozenset[int] = frozenset(),
    ) -> list[int | float]:
        named = {v.index for t in objectives for v, _ in t.coeffs}
        with self._query(named):
            # one search answers the whole query, so one optimizer holds
            # every objective's component
            return self._maxima(self._optimizer(named), objectives, tuple(sorted(assuming)))

    def _optimizer(self, named: set[int]) -> z3.Optimize:
        root = self._root(named)
        held = self._opt.get(root)
        if held is None:
            opt = z3.Optimize()
            opt.set('timeout', self.timeout_ms)
            # each objective's own optimum, not the first's and then the
            # next's under it (`lex`, the default)
            opt.set(priority='box')
            held = (opt, 0)
        opt, n = held
        group = self._group.get(root, [])
        for c in group[n:]:
            self._encode(opt, c)
        self._opt[root] = (opt, len(group))
        return opt

    def _maxima(
        self, opt: z3.Optimize, objectives: Sequence[Term], lits: tuple[int, ...],
    ) -> list[int | float]:
        opt.push()
        try:
            handles = [opt.maximize(self._to_z3(t)) for t in objectives]
            status = self._record(self._status_of(opt.check(*[self._lit(g) for g in lits])))
            if status is Status.UNSAT:
                return [-math.inf] * len(objectives)
            if status is not Status.SAT:
                return [math.inf] * len(objectives)
            # an unbounded objective comes back as the symbolic `oo`
            return [b.as_long() if z3.is_int_value(b := opt.upper(h)) else math.inf
                    for h in handles]
        finally:
            opt.pop()
