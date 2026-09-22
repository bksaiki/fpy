"""
Solver backends for the digit-bound store.

The constraint language -- affine terms over integer variables -- and the
backend that answers questions about it.  :mod:`fpy2.analysis.digit_bound.store`
states the constraints; nothing here walks an AST or knows what a variable
means.
"""

import math
from dataclasses import dataclass
from typing import Protocol, TypeAlias

import z3

from ...utils import Unionfind

_Z3Solver: TypeAlias = 'z3.Solver | z3.Optimize'
"""Either z3 engine: both take assertions and answer ``check``, and which one
a query runs on is :attr:`Z3Solver.bisect`'s business."""

__all__ = [
    'Constraint',
    'Solver',
    'Term',
    'Z3Solver',
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


@dataclass(frozen=True)
class Constraint:
    """One constraint of the store.

    ``op`` is ``'<='``, ``'=='``, or ``'<=max'`` -- the last meaning
    ``lhs <= max(rhs)``, which is a disjunction rather than anything affine and
    the shape every upper bound through a ``max`` takes.
    """

    lhs: Term
    op: str
    rhs: tuple[Term, ...]


class Solver(Protocol):
    """Backend interface, kept small so the store is not tied to z3.

    Incremental: the store states each constraint once, as it is made, and
    asks many objectives against the accumulated system.  Re-stating the whole
    system per query is what a shared store makes expensive.
    """

    def assume(self, constraint: Constraint) -> None:
        """State *constraint*.  Constraints only ever narrow."""
        ...

    def maximize(self, objective: Term, cutoff: int | None = None) -> int | float:
        """An upper bound on the greatest value *objective* can take under
        what was assumed -- every caller may use the answer as one.

        ``math.inf`` when unbounded **or** undecided: a backend that cannot
        answer must degrade rather than raise, since unbounded is the answer
        the analysis gives without a store at all.  ``-math.inf`` when the
        constraints are unsatisfiable.

        *cutoff*, where given, says the caller will not distinguish anything
        at or above it, so the backend may stop as soon as it knows the
        maximum gets there and answer ``inf``.  That turns a search into one
        refutation; ignoring it is correct, since the exact maximum is an
        upper bound whatever the caller asked for.
        """
        ...


class Z3Solver:
    """:class:`Solver` backed by z3's ``Optimize``.

    The store is quantifier-free linear integer arithmetic over a handful of
    variables; *timeout_ms* guards against a pathological one rather than an
    expected path.
    """

    timeout_ms: int

    bisect: bool
    """Find the maximum by refuting ``objective >= k`` rather than by asking
    z3 to optimize.

    On, because the component's plain solver is already built: a capped
    :meth:`maximize` asks satisfiability questions of it, so refutation
    reuses a solver that has been encoded once and has been learning ever
    since, where `Optimize` is a second encoding of the same constraints and
    its search costs more than the dozen checks refutation spends.

    Never off in production.  The `Optimize` path stays reachable so that
    `test_both_backends_agree` can check the two against each other, which is
    the only thing that would catch `_bisect` drifting from it.
    """

    _env: dict[_Var, z3.ArithRef]

    _components: Unionfind[int]
    """Variable indices grouped into components: constraints that share no
    variable cannot bear on each other, so they never meet a solver
    together."""

    _group: dict[int, list[Constraint]]
    """Constraints by component root, kept unencoded until asked for."""

    _built: dict[tuple[int, bool], _Z3Solver]
    """A component's solvers, by root and by whether it optimizes.  A `check`
    does not need `Optimize`'s machinery and is markedly faster without it,
    and both are cheap to keep: each is encoded once and then learns
    incrementally."""

    _encoded: dict[Term, z3.ArithRef]
    """Each term's z3 form, built once.  The same terms recur across the
    constraints of one store far more often than not, and rebuilding the
    expression costs more than the check that follows it."""

    _span: int
    """Total magnitude of the constants assumed so far."""

    _scale: int
    """Largest coefficient magnitude assumed so far.

    With :attr:`_span` this says how far a bound the system implies can
    reach: its constants scaled by the widest coefficient over them.  Past
    that nothing is holding the objective down, which is where the upward
    probe stops and answers ``inf``.  Only an upper cutoff, and crossing it
    merely loosens the answer -- so a ceiling set too low costs precision,
    never soundness.  There is no matching floor: a satisfiable system has
    some solution, so the downward probe always terminates, and stopping it
    early would report a finite maximum as something below it.
    """

    def __init__(self, timeout_ms: int = 10_000, bisect: bool = True):
        self.timeout_ms = timeout_ms
        self.bisect = bisect
        self._env = {}
        self._components = Unionfind()
        self._group = {}
        self._built = {}
        self._encoded = {}
        self._span = 0
        self._scale = 1

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

    def _find(self, v: int) -> int:
        """*v*'s component, which it joins as a singleton if it has none."""
        return self._components.add(v)

    def _union(self, vs: set[int]) -> int:
        """Merge every component the variables *vs* touch into one, and
        return the root it now has.

        The side holding the most constraints survives, so the fewest of
        them move, and it keeps its live solvers -- which simply learn what
        moved in, where rebuilding would throw away everything z3 has
        inferred.
        """
        roots = {self._find(v) for v in vs}
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

    def assume(self, constraint: Constraint) -> None:
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
        # z3 is incremental: state it into the live solvers rather than
        # discarding ones that have already learned from their neighbours
        for opt in (False, True):
            live = self._built.get((root, opt))
            if live is not None:
                self._encode(live, constraint)

    def _encode(self, into: _Z3Solver, constraint: Constraint) -> None:
        lhs = self._to_z3(constraint.lhs)
        rhs = [self._to_z3(r) for r in constraint.rhs]
        if constraint.op == '==':
            into.add(lhs == rhs[0])
        elif constraint.op == '<=':
            into.add(lhs <= rhs[0])
        else:
            into.add(z3.Or([lhs <= r for r in rhs]))

    def _solver_for_vars(self, named: set[int], *, optimize: bool) -> _Z3Solver:
        """A solver holding just what can bear on the variables *named*.

        A constraint reaching none of them cannot move a bound over them, so
        leaving it out is exact, not an approximation -- and most of them
        reach nothing, so most never need encoding at all.
        """
        roots = {self._find(v) for v in named} if named else set()
        if len(roots) > 1:
            # An objective spanning several components names no single one,
            # so its solver would be built from scratch every time.  Merge
            # them instead: what is asked together will be asked together
            # again -- an `msb` and the `lsb` beside it are related by nothing
            # but the question -- and a constraint sharing no variable with
            # the objective cannot move its bound, so the answer is the same
            # either way.
            roots = {self._union(named)}
        key = (min(roots, default=-1), optimize)
        cached = self._built.get(key)
        if cached is not None:
            return cached
        s = z3.Optimize() if optimize else z3.Solver()
        s.set('timeout', self.timeout_ms)
        for r in roots:
            for c in self._group.get(r, ()):
                self._encode(s, c)
        self._built[key] = s
        return s

    def maximize(self, objective: Term, cutoff: int | None = None) -> int | float:
        e = self._to_z3(objective)
        named = {v.index for v, _ in objective.coeffs}
        if cutoff is not None:
            # one refutation settles it: reaching the cutoff answers `inf`,
            # and failing to reach it puts the maximum at `cutoff - 1` or
            # below.  Both are upper bounds, so the answer stays usable as
            # one, and `Optimize` is not needed for a `check`.
            solver = self._solver_for_vars(named, optimize=False)
            return math.inf if self._reaches(solver, e, cutoff) else cutoff - 1
        solver = self._solver_for_vars(named, optimize=not self.bisect)
        if self.bisect:
            return self._bisect(solver, e, objective)
        else:
            return self._optimize(solver, e)

    def _optimize(self, solver: _Z3Solver, e: z3.ArithRef) -> int | float:
        """The maximum from ``Optimize``: one search, run by z3."""
        solver.push()
        try:
            handle = solver.maximize(e)
            status = solver.check()
            if status == z3.unsat:
                return -math.inf
            if status != z3.sat:
                return math.inf
            bound = solver.upper(handle)
            # an unbounded objective comes back as the symbolic `oo`
            return bound.as_long() if z3.is_int_value(bound) else math.inf
        finally:
            solver.pop()

    def _reaches(self, solver: _Z3Solver, e: z3.ArithRef, k: int) -> bool:
        """Can the objective reach *k*?

        An undecided check reads as ``True``, which widens the bracket and
        so the answer -- the direction a backend that cannot answer must
        fail in."""
        return solver.check(e >= k) != z3.unsat

    def _bisect(
        self, solver: _Z3Solver, e: z3.ArithRef, objective: Term,
    ) -> int | float:
        """The maximum by refutation: ``max >= k`` is one satisfiability
        question, and monotone in *k*, so a bracket and a bisection inside it
        settle it in a handful of cheap checks -- where asking z3 to optimize
        costs far more for the same answer.

        The first check also carries a model, and the objective's value there
        is one it genuinely attains, so the bracket starts from a real lower
        bound instead of from zero.  Widening it by doubling keeps the search
        logarithmic; stepping through attained values one at a time would not.

        Sound because ``unsat`` is never spurious: every ``k`` it rules out is
        a genuine upper bound, so the answer is never below the true maximum.
        """

        status = solver.check()
        if status == z3.unsat:
            return -math.inf
        if status != z3.sat:
            return math.inf

        # how far a bound can reach: the constants either side, scaled by
        # the widest coefficient over them
        scale = self._scale
        for _, k in objective.coeffs:
            scale = max(scale, abs(k))
        ceiling = (self._span + abs(objective.const)) * scale

        # start from a value the objective actually takes, so the bracket does
        # not have to be walked to from zero
        found = solver.model().eval(e, model_completion=True)
        lo = found.as_long() if z3.is_int_value(found) else 0
        if not self._reaches(solver, e, lo):
            lo = 0          # the model said nothing usable; fall back

        # widen upward by doubling the gap until the objective cannot reach
        step, hi = 1, lo + 1
        while self._reaches(solver, e, hi):
            lo, step = hi, step * 2
            hi = lo + step
            if lo > ceiling:
                return math.inf

        # invariant: reaches(lo), not reaches(hi); the maximum is in [lo, hi)
        while hi - lo > 1:
            mid = lo + (hi - lo) // 2
            if self._reaches(solver, e, mid):
                lo = mid
            else:
                hi = mid
        return lo
