"""
A HiGHS backend for the digit-bound store, for comparison with z3.

The store as a mixed-integer program: one integer column per variable, a 0/1
column per guard literal, and big-M rows for guards and ``max``.  HiGHS
optimizes natively, so a maximum is one solve.  Needs the ``highspy``
package, which FPy does not depend on (``evaluation/pyproject.toml`` has it).

**Not a sound default.**  Big-M needs every column bounded, to ``[-B, B]``,
and a ``B`` too small removes solutions: a maximum can come back below the
true one, or a satisfiable system infeasible.  Where every coefficient is a
unit, as nearly all the store's are, each disjunct is a system of difference
constraints, whose every face has a point within the total of its constants
of the origin.  ``B`` is four times past that, for the component's own
constants, so the box cuts no optimum and empties no system.  A coefficient
beyond a unit can need more, which ``B`` scales by only once.

HiGHS also works in floating point.  The answer is read from the dual bound
and floored with a tolerance; a tolerance only ever loosens a big-M row.  A
box past :data:`_MAX_BOX` would lose integers to rounding, so a component
needing one answers ``inf``, degraded.  See ``docs/todos/solver-backends.md``,
section 3.3.
"""

import math
from dataclasses import dataclass, field

import highspy
import numpy as np

from .solver import Constraint, Rel, Status, Term, _ComponentSolver

__all__ = ['HighsSolver']

_TOL = 1e-6
"""How far below an integer a floating-point bound may land and still
round up to it."""

_MAX_BOX = 1 << 40
"""The widest box solved: a big-M row over it stays well inside the integers
a double holds exactly."""

_INF = highspy.kHighsInf

_OPTIMAL = highspy.HighsModelStatus.kOptimal
_INFEASIBLE = highspy.HighsModelStatus.kInfeasible
_BOUNDED_STOPS = frozenset({
    highspy.HighsModelStatus.kTimeLimit,
    highspy.HighsModelStatus.kIterationLimit,
    highspy.HighsModelStatus.kSolutionLimit,
    highspy.HighsModelStatus.kInterrupt,
})
"""Statuses that stop a search early, but with its dual bound still valid."""


@dataclass
class _Model:
    """One component's mixed-integer program.

    Rows are added when a query needs them, not as constraints arrive: a
    constraint can widen the box, and the rows already added were sized for
    the old one.
    """

    constraints: list[Constraint] = field(default_factory=list)
    """Everything stated into this component, in order."""
    span: int = 0
    """Total magnitude of their constants."""
    scale: int = 1
    """Their largest coefficient magnitude."""

    h: highspy.Highs | None = None
    box: int = 0
    """Every variable column's bound, ``[-box, box]``."""
    done: int = 0
    """How many of :attr:`constraints` have rows."""
    cols: dict[int, int] = field(default_factory=dict)
    """Each variable's column, by variable index."""
    lits: dict[int, int] = field(default_factory=dict)
    """Each guard literal's 0/1 column."""


@dataclass(frozen=True)
class _Result:
    """What one solve found, read before the model changes again."""

    status: highspy.HighsModelStatus
    dual_bound: float
    objective: float


def _ix(xs: list[int]) -> np.ndarray:
    return np.array(xs, dtype=np.int32)


def _vals(xs: list[float]) -> np.ndarray:
    return np.array(xs, dtype=np.float64)


class HighsSolver(_ComponentSolver[_Model]):
    """:class:`~fpy2.analysis.digit_bound.solver.Solver` backed by HiGHS.

    One model per component, kept across queries: a constraint stated into a
    live one becomes rows at its next query, and an assumed literal fixes its
    column for one solve.  *timeout_ms* bounds each solve.
    """

    bisect = False

    # -- the model ---------------------------------------------------------

    def _new(self, optimize: bool) -> _Model:
        return _Model()

    def _encode(self, into: _Model, constraint: Constraint) -> None:
        into.constraints.append(constraint)
        for t in (constraint.lhs, *constraint.rhs):
            into.span += abs(t.const)
            for _, k in t.coeffs:
                into.scale = max(into.scale, abs(k))

    def _fit(self, m: _Model, need: int = 0) -> bool:
        """Give *m* rows for everything stated into it, over a box past its
        own constants and past *need*; ``False`` where that box is too wide
        to solve.

        The box is a power of two, so growing it rebuilds a model rarely.
        """
        want = max(4 * (m.span + 1) * m.scale, need, 1)
        want = 1 << (want - 1).bit_length()
        if want > _MAX_BOX:
            return False
        if m.h is None or m.box < want:
            h = highspy.Highs()
            h.setOptionValue('output_flag', False)
            h.setOptionValue('threads', 1)
            h.setOptionValue('mip_rel_gap', 0.0)
            h.setOptionValue('time_limit', self.timeout_ms / 1000)
            h.changeObjectiveSense(highspy.ObjSense.kMaximize)
            m.h, m.box, m.done = h, want, 0
            m.cols.clear()
            m.lits.clear()
        for c in m.constraints[m.done:]:
            self._rows(m, c)
        m.done = len(m.constraints)
        return True

    def _column(self, m: _Model, lo: float, hi: float) -> int:
        """A fresh integer column over ``[lo, hi]``."""
        assert m.h is not None
        col = m.h.getNumCol()
        m.h.addCol(0.0, lo, hi, 0, _ix([]), _vals([]))
        m.h.changeColsIntegrality(1, _ix([col]), np.array([highspy.HighsVarType.kInteger]))
        return col

    def _var(self, m: _Model, index: int) -> int:
        col = m.cols.get(index)
        if col is None:
            col = m.cols[index] = self._column(m, -m.box, m.box)
        return col

    def _lit(self, m: _Model, i: int) -> int:
        col = m.lits.get(i)
        if col is None:
            col = m.lits[i] = self._column(m, 0, 1)
        return col

    def _reach(self, m: _Model, t: Term) -> int:
        """The greatest value *t* takes over the box."""
        return sum(abs(c) for _, c in t.coeffs) * m.box + t.const

    def _row(
        self, m: _Model, d: Term, guard: tuple[int, ...],
        extra: tuple[tuple[int, float], ...] = (), slack: float = 0.0,
    ) -> None:
        """The row ``d <= slack``, plus the columns *extra*, holding only
        where every literal in *guard* does."""
        assert m.h is not None
        idx = [self._var(m, v.index) for v, _ in d.coeffs]
        val = [float(c) for _, c in d.coeffs]
        for col, c in extra:
            idx.append(col)
            val.append(c)
        upper = slack - d.const
        if guard:
            # d <= M * (|g| - sum(b_g)): free where any literal is off
            big = max(self._reach(m, d) + sum(abs(c) for _, c in extra) - slack, 0)
            for g in guard:
                idx.append(self._lit(m, g))
                val.append(float(big))
            upper += big * len(guard)
        m.h.addRow(-_INF, upper, len(idx), _ix(idx), _vals(val))

    def _rows(self, m: _Model, constraint: Constraint) -> None:
        assert m.h is not None
        lhs, guard = constraint.lhs, constraint.guard
        match constraint.op:
            case Rel.LE:
                self._row(m, lhs - constraint.rhs[0], guard)
            case Rel.EQ:
                d = lhs - constraint.rhs[0]
                self._row(m, d, guard)
                self._row(m, -d, guard)
            case Rel.LE_MAX if len(constraint.rhs) == 1:
                self._row(m, lhs - constraint.rhs[0], guard)
            case Rel.LE_MAX:
                # one selector per term: lhs - r_i <= M_i * (1 - s_i), and
                # some selector on -- or a literal off, where guarded
                sels = []
                for r in constraint.rhs:
                    d = lhs - r
                    s = self._column(m, 0, 1)
                    big = float(max(self._reach(m, d), 0))
                    self._row(m, d, (), ((s, big),), slack=big)
                    sels.append(s)
                idx = sels + [self._lit(m, g) for g in guard]
                val = [1.0] * len(sels) + [-1.0] * len(guard)
                m.h.addRow(1 - len(guard), _INF, len(idx), _ix(idx), _vals(val))

    # -- solving -----------------------------------------------------------

    def _fix(self, m: _Model, lits: tuple[int, ...], lo: float) -> None:
        """Set the lower bound of each of *lits*' columns in *m* to *lo*.  A
        literal *m* has no column for bears on nothing in it."""
        assert m.h is not None
        cols = [m.lits[g] for g in lits if g in m.lits]
        if cols:
            n = len(cols)
            m.h.changeColsBounds(n, _ix(cols), _vals([lo] * n), _vals([1.0] * n))

    def _run(self, m: _Model, lits: tuple[int, ...]) -> _Result:
        """Solve *m* where the literals *lits* hold.

        The result is read before the literals are released: changing a
        bound discards HiGHS's solution and status."""
        assert m.h is not None
        self._fix(m, lits, 1.0)
        try:
            m.h.run()
            info = m.h.getInfo()
            return _Result(
                m.h.getModelStatus(), info.mip_dual_bound, info.objective_function_value)
        finally:
            self._fix(m, lits, 0.0)

    def _check(
        self, into: _Model, lits: tuple[int, ...],
        bound: tuple[Term, int] | None = None,
    ) -> Status:
        if not into.constraints:
            # nothing is stated, so every variable is free and a bound over
            # one is met: only a constant bound can fail
            ok = bound is None or bool(bound[0].coeffs) or bound[0].const >= bound[1]
            return Status.SAT if ok else Status.UNSAT
        # the box must reach far enough for `t >= k` to be satisfiable
        need = 0 if bound is None else 2 * abs(bound[1] - bound[0].const)
        if not self._fit(into, need):
            return Status.UNKNOWN
        h = into.h
        assert h is not None
        row = None
        if bound is not None:
            t, k = bound
            idx = [self._var(into, v.index) for v, _ in t.coeffs]
            val = [float(c) for _, c in t.coeffs]
            row = h.getNumRow()
            h.addRow(k - t.const, _INF, len(idx), _ix(idx), _vals(val))
        try:
            result = self._run(into, lits)
        finally:
            if row is not None:
                h.deleteRows(1, _ix([row]))
        if result.status == _OPTIMAL:
            return Status.SAT
        if result.status == _INFEASIBLE:
            return Status.UNSAT
        return Status.UNKNOWN

    def _maximum(
        self, solver: _Model, objective: Term, lits: tuple[int, ...],
    ) -> int | float:
        if not objective.coeffs:
            # a constant: only an unsatisfiable system bounds it below itself
            unsat = self._status(solver, lits) is Status.UNSAT
            return -math.inf if unsat else objective.const

        m = solver
        # the component's own reach: a bound past it reads as unbounded, and
        # the box must reach past it, or an unbounded objective would stop at
        # the box and read as finite
        scale = max([m.scale, *(abs(c) for _, c in objective.coeffs)])
        ceiling = (m.span + abs(objective.const)) * scale
        if not self._fit(m, 2 * ceiling):
            self._record(Status.UNKNOWN)
            return math.inf

        h = m.h
        assert h is not None
        cols = [self._var(m, v.index) for v, _ in objective.coeffs]
        n = len(cols)
        h.changeColsCost(n, _ix(cols), _vals([float(c) for _, c in objective.coeffs]))
        try:
            result = self._run(m, lits)
        finally:
            h.changeColsCost(n, _ix(cols), _vals([0.0] * n))

        status = result.status
        if status == _INFEASIBLE:
            self._record(Status.UNSAT)
            return -math.inf
        if status != _OPTIMAL and status not in _BOUNDED_STOPS:
            self._record(Status.UNKNOWN)
            return math.inf
        # the dual bound is an upper bound however the search stopped; the
        # objective value is only one the program attains
        bound = result.dual_bound
        if not math.isfinite(bound):
            if status != _OPTIMAL:
                self._record(Status.UNKNOWN)
                return math.inf
            bound = result.objective
        self._record(Status.SAT if status == _OPTIMAL else Status.UNKNOWN)
        answer = math.floor(bound + _TOL) + objective.const
        return math.inf if answer > ceiling else answer
