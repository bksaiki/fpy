"""
The solver interface for the digit-bound store.

The constraint language -- affine terms over integer variables -- the
interface a backend answers questions about it through, and what every
backend shares.  Each backend is a module of its own: z3 in :mod:`z3_solver`,
cvc5 beside it.  :mod:`fpy2.analysis.digit_bound.store`
states the constraints; nothing here walks an AST or knows what a variable
means.
"""

import math
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import astuple, dataclass
from enum import Enum, auto
from typing import Generic, NamedTuple, Protocol, TypeVar

from ...utils import Unionfind

__all__ = [
    'Constraint',
    'MeasuredSolver',
    'QueryRecord',
    'Solver',
    'SolverStats',
    'Term',
]



@dataclass(frozen=True)
class _Var:
    """An integer variable.  ``index`` orders terms deterministically, which
    ``id()`` would not."""

    index: int
    name: str

    def __repr__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Term:
    """An affine combination of variables, ``sum(coeff * var) + const``.

    Arithmetic builds terms; the store's :meth:`~DigitBoundStore.le` and friends
    turn them into constraints.  ``==`` is structural equality, not a
    constraint.
    """

    coeffs: tuple[tuple[_Var, int], ...]
    const: int

    @staticmethod
    def _build(coeffs: dict[_Var, int], const: int) -> 'Term':
        items = sorted(
            ((v, c) for v, c in coeffs.items() if c != 0),
            key=lambda kv: kv[0].index,
        )
        return Term(tuple(items), const)

    def __add__(self, other: 'Term | int') -> 'Term':
        if isinstance(other, int):
            return Term(self.coeffs, self.const + other)
        if not isinstance(other, Term):
            return NotImplemented
        coeffs = dict(self.coeffs)
        for v, c in other.coeffs:
            coeffs[v] = coeffs.get(v, 0) + c
        return Term._build(coeffs, self.const + other.const)

    __radd__ = __add__

    def __neg__(self) -> 'Term':
        return Term(tuple((v, -c) for v, c in self.coeffs), -self.const)

    def __sub__(self, other: 'Term | int') -> 'Term':
        return self + (-other)

    def __rsub__(self, other: 'Term | int') -> 'Term':
        return (-self) + other

    def rename(self, subst: 'dict[int, Term]') -> 'Term':
        """This term with every variable *subst* names replaced by its image."""
        out = Term((), self.const)
        for v, c in self.coeffs:
            out = out + subst.get(v.index, Term(((v, 1),), 0)) * c
        return out

    def __mul__(self, k: int) -> 'Term':
        if not isinstance(k, int):
            return NotImplemented
        return Term._build({v: c * k for v, c in self.coeffs}, self.const * k)

    __rmul__ = __mul__

    def __repr__(self) -> str:
        parts = [f'{c}*{v}' if c != 1 else str(v) for v, c in self.coeffs]
        if self.const or not parts:
            parts.append(str(self.const))
        return ' + '.join(parts)


class Rel(Enum):
    """A constraint's relation between its sides."""

    LE = auto()
    EQ = auto()
    LE_MAX = auto()
    """``lhs <= max(rhs)``: a disjunction rather than anything affine, and the
    shape every upper bound through a ``max`` takes."""


@dataclass(frozen=True)
class Constraint:
    """One constraint of the store.

    A *guard* makes it conditional: it holds where every literal named does,
    and a query says which literals it assumes.
    """

    op: Rel
    lhs: Term
    rhs: tuple[Term, ...]
    guard: tuple[int, ...] = ()


class Solver(Protocol):
    """Backend interface, kept small so the store is not tied to z3.

    Incremental: the store states each constraint once, as it is made, and
    asks many objectives against the accumulated system.
    """

    def assume(self, constraint: Constraint) -> None:
        """State *constraint*.  Constraints only ever narrow."""
        ...

    def maximize(
        self, objectives: Sequence[Term], assuming: frozenset[int] = frozenset(),
    ) -> list[int | float]:
        """An upper bound on the greatest value each of *objectives* can take
        under what was assumed, and the literals *assuming* -- every caller
        may use each answer as one.  One call is one query.

        The maxima are independent: each may be attained by a different
        solution, as the bounds of one format are.  A backend that can
        optimize several objectives in one search may ask them together.

        ``math.inf`` where unbounded **or** undecided: a backend that cannot
        answer must degrade rather than raise.  ``-math.inf`` when the
        constraints are unsatisfiable.
        """
        ...

    def reaches(
        self, objective: Term, k: int, assuming: frozenset[int] = frozenset(),
    ) -> bool:
        """Can *objective* reach *k* under what was assumed, and the literals
        *assuming*?  One query, and a decision rather than an optimization:
        one satisfiability check, where a maximum is a search.

        ``True`` where undecided: a ``False`` is a refutation every caller
        may rely on.
        """
        ...


class MeasuredSolver(Solver, Protocol):
    """A backend that reports what it did, as every one here does: what
    the sensitivity evaluation reads, and nothing the store needs."""

    stats: 'SolverStats'
    """What the backend has done so far."""

    trace: 'list[QueryRecord]'
    """Each query the backend answered, in order, where it was built to keep
    them (``keep_trace``); empty otherwise."""


@dataclass
class SolverStats:
    """What a backend has done, and the time it took.

    Always kept.  Each update is a counter or a clock read, which a solver
    check outweighs by orders of magnitude.
    """

    solver_time: float = 0.0
    """Seconds in the backend: stating constraints, which z3 and cvc5 encode
    into live solvers as they arrive, and answering queries."""
    queries: int = 0
    """Calls to :meth:`Solver.maximize` and :meth:`Solver.reaches`."""
    degraded: int = 0
    """Queries with an answer loosened because the backend could not decide: a timeout or an
    unknown."""
    constraints: int = 0
    """Calls to :meth:`Solver.assume`."""

    def __add__(self, other: 'SolverStats') -> 'SolverStats':
        return SolverStats(*(a + b for a, b in zip(astuple(self), astuple(other))))


class QueryRecord(NamedTuple):
    """One query, as :attr:`Solver.trace` keeps it."""

    constraints: int
    """Constraints in the component it was asked of: all the backend saw."""
    seconds: float
    """Time in the backend answering it."""


class Status(Enum):
    """A backend call's result."""

    SAT = auto()
    UNSAT = auto()
    UNKNOWN = auto()


_S = TypeVar('_S')
"""A backend's solver for one component."""


class _ComponentSolver(Generic[_S]):
    """What every backend shares: components, bisection, and the
    statistics.

    A backend supplies :meth:`_new`, :meth:`_encode`, :meth:`_check` and
    :meth:`_value`; one that optimizes natively clears :attr:`bisect` and
    overrides :meth:`_maximum`, or :meth:`_maxima` to optimize a query's
    objectives together, and needs no :meth:`_value`.
    """

    timeout_ms: int

    stats: SolverStats

    keep_trace: bool
    """Whether each query is kept in :attr:`trace`: only the sensitivity
    evaluation reads it, so an ordinary compilation does not pay for it."""

    trace: list[QueryRecord]

    bisect: bool = True
    """Whether a maximum is found by bisection, rather than asked of a
    separate, optimizing solver."""

    _components: Unionfind[int]
    """Variable indices grouped into components: constraints that share no
    variable cannot bear on each other, so they never meet a solver
    together."""

    _group: dict[int, list[Constraint]]
    """Constraints by component root, kept unencoded until asked for."""

    _built: dict[tuple[int, bool], _S]
    """A component's solvers, by root and by whether it optimizes.  Each is
    encoded once and then learns incrementally."""

    _span: int
    """Total magnitude of the constants assumed so far."""

    _scale: int
    """Largest coefficient magnitude assumed so far.

    With :attr:`_span` this says how far a bound the system implies can
    reach, which is where the upward probe stops and answers ``inf``.
    Crossing it only loosens the answer, so a ceiling set too low costs
    precision, never soundness.  There is no matching floor: a satisfiable
    system has some solution, so the downward probe always terminates.
    """

    _degraded: bool
    """Whether the query being answered has met an undecided call."""

    def __init__(self, timeout_ms: int = 10_000, keep_trace: bool = False):
        self.timeout_ms = timeout_ms
        self.stats = SolverStats()
        self.keep_trace = keep_trace
        self.trace = []
        self._components = Unionfind()
        self._group = {}
        self._built = {}
        self._span = 0
        self._scale = 1
        self._degraded = False

    # -- what a backend supplies -------------------------------------------

    def _new(self, optimize: bool) -> _S:
        """An empty solver, optimizing where *optimize*."""
        raise NotImplementedError

    def _encode(self, into: _S, constraint: Constraint) -> None:
        """State *constraint* into the solver *into*."""
        raise NotImplementedError

    def _check(
        self, into: _S, lits: tuple[int, ...], bound: tuple[Term, int] | None = None,
    ) -> Status:
        """Is what *into* holds satisfiable where the literals *lits* hold,
        and, given *bound* ``(t, k)``, where ``t >= k``?"""
        raise NotImplementedError

    def _value(self, into: _S, objective: Term) -> int | None:
        """*objective*'s value in the model of *into*'s last satisfiable
        check, where it has one."""
        raise NotImplementedError

    # -- components --------------------------------------------------------

    def _union(self, vs: set[int]) -> int:
        """Merge every component the variables *vs* touch into one, and
        return the root it now has.

        The side holding the most constraints survives, so the fewest move
        and its live solvers just learn what moved in.
        """
        roots = {self._components.add(v) for v in vs}
        root = max(roots, key=lambda r: len(self._group.get(r, ())))
        for other in roots - {root}:
            self._components.union(root, other)   # *root* stays the leader
            moved = self._group.pop(other, [])
            self._group.setdefault(root, []).extend(moved)
            for opt in (False, True):
                live = self._built.get((root, opt))
                if live is not None:
                    for c in moved:
                        self._encode(live, c)
                self._built.pop((other, opt), None)
        return root

    @staticmethod
    def _vars(constraint: Constraint) -> set[int]:
        out: set[int] = set()
        for t in (constraint.lhs, *constraint.rhs):
            out.update(v.index for v, _ in t.coeffs)
        return out

    @staticmethod
    def _term_vars(term: Term) -> set[int]:
        return {v.index for v, _ in term.coeffs}

    def assume(self, constraint: Constraint) -> None:
        start = time.perf_counter()
        for t in (constraint.lhs, *constraint.rhs):
            self._span += abs(t.const)
            for _, k in t.coeffs:
                self._scale = max(self._scale, abs(k))
        named = self._vars(constraint)
        # every constraint the store builds names at least one variable: each
        # of `le`/`ge`/`eq`/`le_max` takes a term the store minted
        assert named, f'constraint over no variable: {constraint}'
        root = self._union(named)
        self._group.setdefault(root, []).append(constraint)
        self.stats.constraints += 1
        # the backends are incremental: state it into the live solvers rather
        # than discarding ones that have already learned from their neighbours
        for opt in (False, True):
            live = self._built.get((root, opt))
            if live is not None:
                self._encode(live, constraint)
        self.stats.solver_time += time.perf_counter() - start

    def _solver_for_vars(self, named: set[int], *, optimize: bool) -> _S:
        """A solver holding just what can bear on the variables *named*.

        A constraint reaching none of them cannot move a bound over them, so
        leaving it out is exact, not an approximation -- and most of them
        reach nothing, so most never need encoding at all.
        """
        roots = {self._components.add(v) for v in named} if named else set()
        if len(roots) > 1:
            # An objective spanning several components names no single one,
            # so its solver would be built from scratch every time.  What is
            # asked together will be asked together again, and merging does
            # not change any answer.
            roots = {self._union(named)}
        key = (min(roots, default=-1), optimize)
        cached = self._built.get(key)
        if cached is not None:
            return cached
        s = self._new(optimize)
        for r in roots:
            for c in self._group.get(r, ()):
                self._encode(s, c)
        self._built[key] = s
        return s

    # -- queries -----------------------------------------------------------

    def maximize(
        self, objectives: Sequence[Term], assuming: frozenset[int] = frozenset(),
    ) -> list[int | float]:
        lits = tuple(sorted(assuming))
        named = {v.index for t in objectives for v, _ in t.coeffs}
        with self._query(named):
            if not self.bisect:
                # one search answers the whole query, so one solver holds
                # every objective's component
                solver = self._solver_for_vars(named, optimize=True)
                return self._maxima(solver, objectives, lits)
            # each objective alone, against its own component: merging every
            # component a query happens to name would only grow later queries
            return [
                self._maximum(
                    self._solver_for_vars(self._term_vars(t), optimize=False), t, lits)
                for t in objectives
            ]

    def reaches(
        self, objective: Term, k: int, assuming: frozenset[int] = frozenset(),
    ) -> bool:
        named = self._term_vars(objective)
        with self._query(named):
            # a check needs no optimizer, whichever way this backend maximizes
            solver = self._solver_for_vars(named, optimize=False)
            return self._reaches(solver, objective, k, tuple(sorted(assuming)))

    @contextmanager
    def _query(self, named: set[int]) -> Iterator[None]:
        """Count and time one query over the variables *named*, and trace it
        where :attr:`keep_trace`."""
        start = time.perf_counter()
        self._degraded = False
        try:
            yield
        finally:
            seconds = time.perf_counter() - start
            self.stats.queries += 1
            self.stats.degraded += self._degraded
            self.stats.solver_time += seconds
            if self.keep_trace:
                self.trace.append(QueryRecord(self._size(named), seconds))

    def _size(self, named: set[int]) -> int:
        """How many constraints the components of *named* hold: what a query
        over them saw, whether asked of one solver or of one per component."""
        roots = {self._components.add(v) for v in named}
        return sum(len(self._group.get(r, ())) for r in roots)

    def _record(self, status: Status) -> Status:
        """Note a backend call's result: an undecided one degrades the query
        it serves."""
        if status is Status.UNKNOWN:
            self._degraded = True
        return status

    def _status(
        self, into: _S, lits: tuple[int, ...], bound: tuple[Term, int] | None = None,
    ) -> Status:
        return self._record(self._check(into, lits, bound))

    def _reaches(
        self, solver: _S, objective: Term, k: int, lits: tuple[int, ...],
    ) -> bool:
        """Can the objective reach *k*?

        An undecided check reads as ``True``, which widens the bracket and
        so the answer -- the direction a backend that cannot answer must
        fail in."""
        return self._status(solver, lits, (objective, k)) is not Status.UNSAT

    def _ceiling(self, objective: Term) -> int:
        """How far a bound on *objective* can reach: the constants either
        side, scaled by the widest coefficient over them."""
        scale = self._scale
        for _, k in objective.coeffs:
            scale = max(scale, abs(k))
        return (self._span + abs(objective.const)) * scale

    def _maxima(
        self, solver: _S, objectives: Sequence[Term], lits: tuple[int, ...],
    ) -> list[int | float]:
        """Each objective's maximum from the one optimizing *solver*, where
        not :attr:`bisect`: one :meth:`_maximum` apiece."""
        return [self._maximum(solver, t, lits) for t in objectives]

    def _maximum(
        self, solver: _S, objective: Term, lits: tuple[int, ...],
    ) -> int | float:
        """*objective*'s maximum."""
        return self._bisect(solver, objective, lits)

    def _bisect(
        self, solver: _S, objective: Term, lits: tuple[int, ...],
    ) -> int | float:
        """The maximum by refutation: ``max >= k`` is one satisfiability
        question, and monotone in *k*, so a bracket and a bisection inside it
        settle it in a handful of cheap checks.

        The first check carries a model, and the objective's value there is
        one it genuinely attains, so the bracket starts from a real lower
        bound; doubling to widen keeps the search logarithmic.

        Sound because ``unsat`` is never spurious: every ``k`` it rules out is
        a genuine upper bound, so the answer is never below the true maximum.
        """

        status = self._status(solver, lits)
        if status is Status.UNSAT:
            return -math.inf
        if status is not Status.SAT:
            return math.inf

        ceiling = self._ceiling(objective)

        # start from a value the objective actually takes, so the bracket does
        # not have to be walked to from zero
        found = self._value(solver, objective)
        lo = found if found is not None else 0

        # widen upward by doubling the gap until the objective cannot reach
        step, hi = 1, lo + 1
        while self._reaches(solver, objective, hi, lits):
            lo, step = hi, step * 2
            hi = lo + step
            if lo > ceiling:
                return math.inf

        # invariant: reaches(lo), not reaches(hi); the maximum is in [lo, hi)
        while hi - lo > 1:
            mid = lo + (hi - lo) // 2
            if self._reaches(solver, objective, mid, lits):
                lo = mid
            else:
                hi = mid
        return lo
