"""
A relational store over integer exponents.

Linear constraints over integer variables, and one query: the precision two
of them bracket.  Everything stays in *exponent* space, which keeps ``2**x``
out of the constraints and the store decidable.  What the variables mean is
:mod:`infer`'s business, and how a question is answered is :mod:`solver`'s.
"""

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from .solver import Constraint, Solver, Term, Z3Solver, _Var

__all__ = ['DigitBoundStore']


@dataclass
class _Instance:
    elementwise: set[int]
    subst: dict[int, Term]
    tag: str
    done: set[int] = field(default_factory=set)
    """constraints replayed onto it"""
    grown: set[int] = field(default_factory=set)
    """variables in *subst* whose constraints have been replayed"""


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

    _each: set[int]
    """Constraints, by position, that hold at every index of the per-element
    variables they name although they name others too."""

    _instances: dict[int, _Instance]
    """Every :meth:`instance`, by its renaming's identity."""

    _copies: set[int]
    """Constraints, by position, that are replays, so never replayed again."""

    _by_var: dict[int, list[int]]
    """Constraints, by position, naming each variable; replays excepted."""

    _n_lits: int
    """How many guard literals exist."""

    def __init__(self, solver: Solver | None = None):
        self._solver = solver if solver is not None else Z3Solver()
        self._constraints = []
        self._n_vars = 0
        self._answers = {}
        self._decisions = {}
        self._constrained = set()
        self._each = set()
        self._instances = {}
        self._copies = set()
        self._by_var = {}
        self._n_lits = 0

    def _add(self, c: Constraint, each: bool = False, copy: bool = False) -> None:
        i = len(self._constraints)
        if each:
            self._each.add(i)
        if copy:
            self._copies.add(i)
        self._constraints.append(c)
        for t in (c.lhs, *c.rhs):
            self._constrained.update(v.index for v, _ in t.coeffs)
        self._solver.assume(c)
        self._answers.clear()
        self._decisions.clear()
        if copy:
            return
        vs = {v.index for t in (c.lhs, *c.rhs) for v, _ in t.coeffs}
        for v in vs:
            self._by_var.setdefault(v, []).append(i)
        for inst in self._instances.values():
            if not vs.isdisjoint(inst.subst):
                self._replay(i, inst)
                self._grow(inst)

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
        self, elementwise: set[int], subst: dict[int, Term], tag: str,
    ) -> None:
        """Replay every constraint over *elementwise* variables alone, with
        each variable renamed by *subst* -- those stated so far and those
        stated later.

        Such a constraint holds at every index of the lists it is about, so
        it holds at an index drawn from any subset -- which is how a part of
        a list gets the facts the whole one has without *sharing* a variable
        with it.  One naming a variable outside *elementwise* may be an
        aggregate (``logb(sum xs) <= msb(xs) + k``), true of the list and
        false of a part, so it is left alone -- unless stated *each*, as
        ``max(xs) >= xs`` is, and then only its per-element variables move.

        *subst* grows in place, and is the caller's to keep: one renaming per
        index set is what relates two parts taken over one range.  Only what
        is connected to *subst* is replayed -- the rest would bind variables
        nothing names -- so a caller adding to it calls again.
        """
        inst = self._instances.get(id(subst))
        if inst is None:
            inst = self._instances[id(subst)] = _Instance(elementwise, subst, tag)
        self._grow(inst)

    def _grow(self, inst: _Instance) -> None:
        """Replay what names a variable *inst* renames, to a fixpoint."""
        while new := [v for v in inst.subst if v not in inst.grown]:
            for v in new:
                inst.grown.add(v)
                for i in list(self._by_var.get(v, ())):
                    self._replay(i, inst)

    def _replay(self, i: int, inst: _Instance) -> None:
        if i in inst.done or i in self._copies:
            return
        inst.done.add(i)
        c = self._constraints[i]
        if c.guard:
            # a literal names one definition, not one per index
            return
        vs = [v for t in (c.lhs, *c.rhs) for v, _ in t.coeffs]
        moved = [v for v in vs if v.index in inst.elementwise]
        if not moved or (len(moved) < len(vs) and i not in self._each):
            return
        for v in moved:
            if v.index not in inst.subst:
                inst.subst[v.index] = self.var(v.name + inst.tag)
        self._add(Constraint(
            c.lhs.rename(inst.subst), c.op,
            tuple(t.rename(inst.subst) for t in c.rhs),
        ), copy=True)

    def literal(self) -> int:
        """A fresh guard literal, for a constraint that holds only where it
        does; see :meth:`maximum`."""
        self._n_lits += 1
        return self._n_lits

    def le(
        self, lhs: Term, rhs: Term | int, *,
        each: bool = False, guard: tuple[int, ...] = (),
    ) -> None:
        """``lhs <= rhs``; *each* where it holds at every index, see
        :meth:`instance`, and only where every literal in *guard* holds."""
        self._add(Constraint(lhs, '<=', (_as_term(rhs),), guard), each)

    def ge(
        self, lhs: Term, rhs: Term | int, *,
        each: bool = False, guard: tuple[int, ...] = (),
    ) -> None:
        """``lhs >= rhs``."""
        self.le(_as_term(rhs), lhs, each=each, guard=guard)

    def eq(self, lhs: Term, rhs: Term | int) -> None:
        """``lhs == rhs``.  The walk states one-directional bounds and never
        reaches for this; it is here for a constraint set built by hand."""
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
