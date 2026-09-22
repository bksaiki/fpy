"""
A relational store over integer exponents.

Linear constraints over integer variables, and one query: the precision two of
them bracket.  Everything stays in *exponent* space, which is what keeps
``2**x`` out of the constraints and the whole store decidable.

What the variables mean is :mod:`fpy2.analysis.digit_bound.infer`'s business;
nothing here walks an AST.  How a question is answered is
:mod:`fpy2.analysis.digit_bound.solver`'s.
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

    _answers: dict[Term, int | float]
    """Each term's maximum, which only :meth:`_add` can change."""

    _decisions: dict[tuple, bool]
    """Each :meth:`reaches` question's answer, kept apart from
    :attr:`_answers`: a capped query returns an upper bound rather than the
    maximum, so the two must not be mistaken for each other."""

    _constrained: set[int]
    """Variables some constraint names; the rest are free, and a term naming
    one is unbounded by inspection."""

    def __init__(self, solver: Solver | None = None):
        self._solver = solver if solver is not None else Z3Solver()
        self._constraints = []
        self._n_vars = 0
        self._answers = {}
        self._decisions = {}
        self._constrained = set()

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

        Such a constraint holds at every index of the lists it is about, so it
        holds at an index drawn from any subset of them -- which is what a
        copy over fresh variables says, and is how a part of a list gets the
        facts the whole one has without *sharing* a variable with it.  One
        naming a variable outside *elementwise* may be an aggregate over the
        whole list (``logb(sum xs) <= msb(xs) + k``), true of the list and false
        of a part, so it is left alone.

        *subst* grows in place: a second call for the same index set reuses
        the renaming, which is what relates two parts taken over one range.
        """
        end = len(self._constraints)
        for c in self._constraints[mark:end]:
            vs = [v for t in (c.lhs, *c.rhs) for v, _ in t.coeffs]
            if not vs or any(v.index not in elementwise for v in vs):
                continue
            for v in vs:
                if v.index not in subst:
                    subst[v.index] = self.var(v.name + tag)
            self._add(Constraint(
                c.lhs.rename(subst), c.op,
                tuple(t.rename(subst) for t in c.rhs),
            ))
        return end

    def le(self, lhs: Term, rhs: Term | int) -> None:
        """``lhs <= rhs``."""
        self._add(Constraint(lhs, '<=', (_as_term(rhs),)))

    def ge(self, lhs: Term, rhs: Term | int) -> None:
        """``lhs >= rhs``."""
        self.le(_as_term(rhs), lhs)

    def eq(self, lhs: Term, rhs: Term | int) -> None:
        """``lhs == rhs``, the shape a definitional equality takes."""
        self._add(Constraint(lhs, '==', (_as_term(rhs),)))

    def le_max(self, lhs: Term, rhs: Iterable[Term | int]) -> None:
        """``lhs <= max(rhs)``.

        Every upper bound that goes through a `max` takes this shape, and none
        of them are affine -- the backend has to take a disjunction.  An empty
        *rhs* states nothing.
        """
        terms = tuple(_as_term(r) for r in rhs)
        if len(terms) == 1:
            self.le(lhs, terms[0])
        elif terms:
            self._add(Constraint(lhs, '<=max', terms))

    def ge_min(self, lhs: Term, rhs: Iterable[Term | int]) -> None:
        """``lhs >= min(rhs)`` -- :meth:`le_max` with every sign flipped."""
        self.le_max(-lhs, [-_as_term(r) for r in rhs])

    def maximum(self, term: Term) -> int | float:
        """The greatest value *term* can take; ``inf`` when unbounded.

        Memoized: the same term is asked many times over an unchanged store,
        and :meth:`_add` is the only thing that can change an answer.
        """
        answer = self._answers.get(term)
        if answer is None:
            answer = self._bound(term)
            self._answers[term] = answer
        return answer

    def reaches(self, bounds: Sequence[tuple[Term, int]]) -> bool:
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
        key = tuple(bounds)
        answer = self._decisions.get(key)
        if answer is None:
            answer = any(self._at_least(t, k) for t, k in bounds)
            self._decisions[key] = answer
        return answer

    def _at_least(self, term: Term, k: int) -> bool:
        """Can *term* reach *k*?  A free variable reaches anything."""
        if any(v.index not in self._constrained for v, _ in term.coeffs):
            return True
        return self._solver.maximize(term, k) >= k

    def _bound(self, term: Term) -> int | float:
        """*term*'s greatest value, asking the solver only where the answer
        is not already settled.

        A variable no constraint mentions is free, so any term naming one is
        unbounded above -- with either sign, since a free variable runs to
        both infinities.  Saying so costs a set lookup where a solver would
        have to search.
        """
        if any(v.index not in self._constrained for v, _ in term.coeffs):
            return math.inf
        return self._solver.maximize(term)

    # -- the query -------------------------------------------------------

    def prec(self, value: Term, grid: Term | int) -> int | float:
        """Precision of a value whose most significant digit is bounded by
        *value* and whose least significant digit sits at *grid* -- the count
        of significant digits, ``msb - lsb + 1``.

        A count of digits is never negative, so this floors at zero.  ``msb -
        lsb + 1`` goes negative when the grid is coarser than the value's whole
        reach, and how far below is not a precision: every value there rounds
        either to zero or to one quantum, and *which* is the rounding mode's
        business, not the store's.

        Zero means *no* significant digits, i.e. only zero is representable.
        That is below what a :class:`Format` admits -- formats guarantee ``prec
        >= 1`` -- so a caller materializing one has to handle it.
        """
        return max(self.maximum(value - _as_term(grid) + 1), 0)


def _as_term(x: Term | int) -> Term:
    return x if isinstance(x, Term) else Term((), x)
