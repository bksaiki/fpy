"""
A relational store over integer exponents.

Linear constraints over integer variables, and one query: the precision two
of them bracket.  Everything stays in *exponent* space, which keeps ``2**x``
out of the constraints and the store decidable.  What the variables mean is
:mod:`infer`'s business, and how a question is answered is :mod:`solver`'s.
"""

import math
from collections.abc import Iterable, Sequence

from .solver import Constraint, Solver, Term, Z3Solver, _Var

__all__ = ['DigitBoundStore']


class DigitBoundStore:
    """A conjunction of linear constraints over exponent variables.

    Constraints only ever narrow, so omitting one is always sound: an empty
    store answers ``inf``.
    """

    _solver: Solver
    _constraints: list[Constraint]
    _n_vars: int

    _answers: dict[tuple[Term, frozenset[int]], int | float]
    """Each term's maximum under each assumption set, which only :meth:`_add`
    can change."""

    _decisions: dict[tuple, bool]
    """Each :meth:`reaches` question's answer, kept apart from
    :attr:`_answers`: a capped query returns an upper bound rather than the
    maximum, so the two must not be mistaken for each other."""

    _constrained: set[int]
    """Variables some constraint names; the rest are free, and a term naming
    one is unbounded by inspection."""

    _n_lits: int
    """How many guard literals exist."""

    _universal: set[int]
    """Literals that hold at every index or none, so a constraint they guard
    may be replayed onto an instance."""

    def __init__(self, solver: Solver | None = None):
        self._solver = solver if solver is not None else Z3Solver()
        self._constraints = []
        self._n_vars = 0
        self._answers = {}
        self._decisions = {}
        self._constrained = set()
        self._n_lits = 0
        self._universal = set()

    def _add(self, c: Constraint) -> None:
        self._constraints.append(c)
        for t in (c.lhs, *c.rhs):
            self._constrained.update(v.index for v, _ in t.coeffs)
        self._solver.assume(c)
        self._answers.clear()
        self._decisions.clear()

    @property
    def n_vars(self) -> int:
        """How many variables exist, so a caller can name a range of them."""
        return self._n_vars

    def var(self, name: str) -> Term:
        """A fresh variable, as a term."""
        v = _Var(self._n_vars, name)
        self._n_vars += 1
        return Term(((v, 1),), 0)

    def instance(
        self,
        elementwise: set[int],
        subst: dict[int, Term],
        mark: int,
        tag: str,
    ) -> int:
        """Replay every constraint over *elementwise* variables alone, from
        *mark* on, with each variable renamed by *subst*.  Returns a new mark.

        Such a constraint holds at every index of the lists it is about, so
        it holds at an index drawn from any subset -- which is how a part of
        a list gets the facts the whole one has without *sharing* a variable
        with it.  One naming a variable outside *elementwise* may be an
        aggregate (``logb(sum xs) <= msb(xs) + k``), true of the list and
        false of a part, so it is left alone.  A guarded one is replayed
        under its guard, and only where every literal in it is universal.

        *subst* grows in place: a second call for the same index set reuses
        the renaming, which is what relates two parts taken over one range.
        """
        end = len(self._constraints)
        for c in self._constraints[mark:end]:
            vs = [v for t in (c.lhs, *c.rhs) for v, _ in t.coeffs]
            if not vs or any(v.index not in elementwise for v in vs):
                continue
            if not self._universal.issuperset(c.guard):
                # a non-universal literal need not hold at every index
                continue
            for v in vs:
                if v.index not in subst:
                    subst[v.index] = self.var(v.name + tag)
            # keep the guard
            self._add(Constraint(
                c.lhs.rename(subst), c.op,
                tuple(t.rename(subst) for t in c.rhs), c.guard,
            ))
        # past the copies too: they are elementwise themselves, so a mark of
        # *end* would replay them again on the next call for this key
        return len(self._constraints)

    def literal(self, *, universal: bool = False) -> int:
        """A fresh guard literal, for a constraint that holds only where it
        does; see :meth:`maximum`.  *universal* where it speaks for every
        index at once, as "every element of `xs` is finite" does."""
        self._n_lits += 1
        if universal:
            self._universal.add(self._n_lits)
        return self._n_lits

    def le(self, lhs: Term, rhs: Term | int, *, guard: tuple[int, ...] = ()) -> None:
        """``lhs <= rhs``, only where every literal in *guard* holds."""
        self._add(Constraint(lhs, '<=', (_as_term(rhs),), guard))

    def ge(self, lhs: Term, rhs: Term | int, *, guard: tuple[int, ...] = ()) -> None:
        """``lhs >= rhs``."""
        self.le(_as_term(rhs), lhs, guard=guard)

    def eq(self, lhs: Term, rhs: Term | int) -> None:
        """``lhs == rhs``.  The walk states one-directional bounds and never
        reaches for this; it is here for a constraint set built by hand."""
        self._add(Constraint(lhs, '==', (_as_term(rhs),)))

    def le_max(
        self, lhs: Term, rhs: Iterable[Term | int], *, guard: tuple[int, ...] = (),
    ) -> None:
        """``lhs <= max(rhs)``.

        Every upper bound that goes through a `max` takes this shape, and none
        of them are affine -- the backend has to take a disjunction.  An empty
        *rhs* states nothing.
        """
        terms = tuple(_as_term(r) for r in rhs)
        if len(terms) == 1:
            self.le(lhs, terms[0], guard=guard)
        elif terms:
            self._add(Constraint(lhs, '<=max', terms, guard))

    def ge_min(
        self, lhs: Term, rhs: Iterable[Term | int], *, guard: tuple[int, ...] = (),
    ) -> None:
        """``lhs >= min(rhs)`` -- :meth:`le_max` with every sign flipped."""
        self.le_max(-lhs, [-_as_term(r) for r in rhs], guard=guard)

    def maximum(
        self, term: Term, assuming: frozenset[int] = frozenset(),
    ) -> int | float:
        """The greatest value *term* can take where the literals *assuming*
        hold; ``inf`` when unbounded.

        Memoized: the same term is asked many times over an unchanged store,
        and :meth:`_add` is the only thing that can change an answer.
        """
        key = (term, assuming)
        answer = self._answers.get(key)
        if answer is None:
            answer = self._bound(term, assuming)
            self._answers[key] = answer
        return answer

    def reaches(
        self, bounds: Sequence[tuple[Term, int]],
        assuming: frozenset[int] = frozenset(),
    ) -> bool:
        """Can any ``term >= k`` hold?

        A decision, where :meth:`maximum` is an optimisation -- the question
        to ask where the answer is only ever compared against a threshold.
        Each term is asked with its own `k` as the solver's cutoff, so the
        backend may stop at the first refutation instead of searching for a
        maximum nothing will read, and the first `True` ends the question --
        so a caller puts the term most likely to reach first.  An empty list
        reaches nothing.
        """
        if not bounds:
            return False
        key = (tuple(bounds), assuming)
        answer = self._decisions.get(key)
        if answer is None:
            answer = any(self._at_least(t, k, assuming) for t, k in bounds)
            self._decisions[key] = answer
        return answer

    def _free(self, term: Term) -> bool:
        """Whether *term* names a variable no constraint mentions, which runs
        to both infinities and so leaves *term* unbounded either way."""
        return any(v.index not in self._constrained for v, _ in term.coeffs)

    def _at_least(self, term: Term, k: int, assuming: frozenset[int]) -> bool:
        """Can *term* reach *k*?  A free variable reaches anything."""
        if self._free(term):
            return True
        return self._solver.maximize(term, k, assuming) >= k

    def _bound(self, term: Term, assuming: frozenset[int]) -> int | float:
        """*term*'s greatest value, asking the solver only where the answer
        is not already settled.

        A free term is unbounded above, which a set lookup settles where the
        solver would have to search.
        """
        if self._free(term):
            return math.inf
        return self._solver.maximize(term, assuming=assuming)

    # -- the query -------------------------------------------------------

    def prec(
        self, value: Term, grid: Term | int,
        assuming: frozenset[int] = frozenset(),
    ) -> int | float:
        """Precision of a value whose most significant digit is bounded by
        *value* and whose least significant digit sits at *grid* -- the count
        of significant digits, ``msb - lsb + 1``.

        Floors at zero: ``msb - lsb + 1`` goes negative where the grid is
        coarser than the value's whole reach, and how far below is not a
        precision.

        Zero means *no* significant digits, i.e. only zero is representable.
        That is below what a :class:`Format` admits, so a caller
        materializing one has to handle it.
        """
        return max(self.maximum(value - _as_term(grid) + 1, assuming), 0)


def _as_term(x: Term | int) -> Term:
    return x if isinstance(x, Term) else Term((), x)
