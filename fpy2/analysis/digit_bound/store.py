"""
A relational store over integer exponents.

Linear constraints over integer variables, and one query: the precision two
of them bracket.  Everything stays in *exponent* space, which keeps ``2**x``
out of the constraints and the store decidable.  What the variables mean is
:mod:`infer`'s business, and how a question is answered is :mod:`solver`'s.
"""

import math
from collections.abc import Iterable, Sequence

from .solver import Constraint, Rel, Solver, Term, Z3Solver, _Var
from .spaces import IndexSpaces, Key

__all__ = ['DigitBoundStore']


class DigitBoundStore:
    """A conjunction of linear constraints over exponent variables.

    Constraints only ever narrow, so omitting one is always sound: an empty
    store answers ``inf``.
    """

    _solver: Solver
    _constraints: list[Constraint]

    _answers: dict[tuple[Term, frozenset[int]], int | float]
    """Each term's maximum under each assumption set, which :meth:`_add` and
    :meth:`imply` invalidate."""

    _decisions: dict[tuple, bool]
    """Each :meth:`reaches` question's answer, kept apart from
    :attr:`_answers`: a capped query returns an upper bound rather than the
    maximum, so the two must not be mistaken for each other."""

    _n_lits: int
    """How many guard literals exist."""

    _own: dict[tuple[int, bool], int]
    """Each variable's literal "this is not `+inf`" (`True`, :meth:`high`) or
    "not `-inf`" (`False`, :meth:`not_neg_inf`), for those that have one."""

    _owner: dict[int, tuple[int, bool]]
    """Each of those literals' variable, and whether it is the `high` one: a
    copy renaming the variable renames the literal with it."""

    _lit_keys: dict[int, frozenset[Key]]
    """Every other literal's binders, which a copy renames it over."""

    _implications: list[tuple[frozenset[int], int]]
    """Each implication's premises and conclusion, at one point; see
    :meth:`imply`."""

    _premised: dict[int, list[int]]
    """Each literal's implications, by position: those it is a premise of."""

    _imp_naming: dict[int, list[int]]
    """Each variable's implications, by position: those naming a literal of
    its own."""

    _vars: list[_Var]
    """Every variable, by index."""

    _naming: dict[int, list[int]]
    """Each variable's constraints, by position; a variable with none is free,
    and a term naming one is unbounded by inspection."""

    spaces: IndexSpaces
    """Which index spaces each variable varies over, kept with the variables so
    every walk sharing them shares it."""

    def __init__(self, solver: Solver | None = None) -> None:
        self._solver = solver if solver is not None else Z3Solver()
        self._constraints = []
        self._answers = {}
        self._decisions = {}
        self._n_lits = 0
        self._own = {}
        self._owner = {}
        self._lit_keys = {}
        self._implications = []
        self._premised = {}
        self._imp_naming = {}
        self._vars = []
        self._naming = {}
        self.spaces = IndexSpaces()

    def _add(self, c: Constraint) -> None:
        self._constraints.append(c)
        for t in (c.lhs, *c.rhs):
            for v, _ in t.coeffs:
                self._naming.setdefault(v.index, []).append(len(self._constraints) - 1)
        self._solver.assume(c)
        self._answers.clear()
        self._decisions.clear()

    def var(self, name: str) -> Term:
        """A fresh variable, as a term."""
        v = _Var(len(self._vars), name)
        self._vars.append(v)
        return Term(((v, 1),), 0)

    def instance(
        self,
        moved: set[Key],
        subst: dict[int, Term],
        lits: dict[int, int],
        done: set[int],
        tag: str,
    ) -> None:
        """Replay every constraint and implication reachable from *subst*'s
        variables through those varying with a class in *moved*, renaming those
        by *subst* (grown with a fresh copy for each one met) and sharing the
        rest.

        Every constraint holds at every index of the lists it names -- an
        aggregate's upper bound is stated only on a copy of its own -- so it
        holds at an index drawn from any subset.  One naming this instance's
        own copies is skipped: it already holds pointwise, and replaying it
        would copy the copy.  A literal is renamed with its variable -- copied
        where its literal is met first -- or, where it varies with a class in
        *moved*, to a fresh one kept in *lits*.  Skipping only weakens.

        *done* holds what was replayed and grows; another call with the same
        *subst* reaches what was stated since.
        """
        frontier = list(subst)
        own = {v.index for t in subst.values() for v, _ in t.coeffs}
        seen: dict[int, bool] = {}

        def moves(v: int) -> bool:
            # classes do not merge during a replay
            if v not in seen:
                seen[v] = bool(self.spaces.signature(v) & moved)
            return seen[v]

        def copy(v: _Var) -> None:
            if v.index not in subst:
                subst[v.index] = self.var(v.name + tag)
                frontier.append(v.index)
                own.update(x.index for x, _ in subst[v.index].coeffs)

        def rename(lit: int) -> int:
            owner = self._owner.get(lit)
            if owner is not None:
                v, high = owner
                if not moves(v):
                    return lit
                # the variable moves, so the copy reaches it through its literal
                copy(self._vars[v])
                (new, _), = subst[v].coeffs
                return self._own_literal(new.index, high)
            keys = self._lit_keys.get(lit, frozenset())
            if not any(self.spaces.find(k) in moved for k in keys):
                return lit
            if lit not in lits:
                lits[lit] = self.literal(k for k in keys if self.spaces.find(k) not in moved)
            return lits[lit]

        while frontier:
            v0 = frontier.pop()
            for i in tuple(self._naming.get(v0, ())):
                if i in done:
                    continue
                done.add(i)
                c = self._constraints[i]
                if any(v.index in own for t in (c.lhs, *c.rhs) for v, _ in t.coeffs):
                    continue
                for t in (c.lhs, *c.rhs):
                    for v, _ in t.coeffs:
                        if moves(v.index):
                            copy(v)
                self._add(Constraint(
                    c.op, c.lhs.rename(subst),
                    tuple(t.rename(subst) for t in c.rhs), tuple(map(rename, c.guard)),
                ))
            for k in tuple(self._imp_naming.get(v0, ())):
                if -1 - k in done:
                    continue
                done.add(-1 - k)
                premises, b = self._implications[k]
                if any(self._owner.get(a, (None,))[0] in own for a in (*premises, b)):
                    continue
                self.imply(map(rename, premises), rename(b))

    def literal(self, keys: Iterable[Key] = ()) -> int:
        """A fresh guard literal, for a constraint that holds only where it
        does (see :meth:`maximum`), varying with the binders *keys*: a copy
        over one of them renames it, as it does a variable.  "Every element of
        `xs` is finite" varies with what `xs` does, but not with its
        elements."""
        self._n_lits += 1
        self._lit_keys[self._n_lits] = frozenset(keys)
        return self._n_lits

    def _own_literal(self, v: int, high: bool) -> int:
        """Variable *v*'s `high` literal, or its "not `-inf`" one."""
        lit = self._own.get((v, high))
        if lit is None:
            self._n_lits += 1
            lit = self._own[v, high] = self._n_lits
            self._owner[lit] = (v, high)
        return lit

    def not_neg_inf(self, t: Term) -> tuple[int, ...]:
        """The literals saying *t* is not `-inf`, one per variable: its "not
        `-inf`" where the coefficient is positive, its `high` where negative.

        A variable may stand for `-inf` -- a zero's `logb` -- which an integer
        store represents as arbitrarily low.  So a lower bound on one holds
        only under its literal, and whatever needs *t* finite assumes these.
        """
        return tuple(self._own_literal(v.index, c < 0) for v, c in t.coeffs)

    def high(self, t: Term) -> tuple[int, ...]:
        """The literals saying *t* is not `+inf`, one per variable: its `high`
        where the coefficient is positive, its "not `-inf`" where negative.  An
        infinity or NaN has `logb` `+inf` (digit-bound.md), so an upper bound on one
        that may be holds only under these."""
        return tuple(self._own_literal(v.index, c > 0) for v, c in t.coeffs)

    def imply(self, premises: int | Iterable[int], b: int) -> None:
        """Wherever every literal in *premises* holds, *b* does, at the same
        point -- as "a sum is finite" implies "every summand is", and finite
        summands a finite sum.  An assumption set is closed under these
        (:meth:`maximum`)."""
        ps = frozenset((premises,) if isinstance(premises, int) else premises)
        if b in ps:
            return
        rule = (ps, b)
        for k in self._premised.get(min(ps), ()) if ps else ():
            if self._implications[k] == rule:
                return
        k = len(self._implications)
        self._implications.append(rule)
        for a in ps:
            self._premised.setdefault(a, []).append(k)
        for lit in (*ps, b):
            owner = self._owner.get(lit)
            if owner is not None:
                self._imp_naming.setdefault(owner[0], []).append(k)
        self._answers.clear()
        self._decisions.clear()

    def closed(self, assuming: frozenset[int]) -> frozenset[int]:
        """*assuming* and everything it implies."""
        out, todo = set(assuming), list(assuming)
        while todo:
            for k in self._premised.get(todo.pop(), ()):
                premises, b = self._implications[k]
                if b not in out and premises <= out:
                    out.add(b)
                    todo.append(b)
        return frozenset(out)

    def le(self, lhs: Term, rhs: Term | int, *, guard: tuple[int, ...] = ()) -> None:
        """``lhs <= rhs``, only where every literal in *guard* holds."""
        self._add(Constraint(Rel.LE, lhs, (_as_term(rhs),), guard))

    def ge(self, lhs: Term, rhs: Term | int, *, guard: tuple[int, ...] = ()) -> None:
        """``lhs >= rhs``."""
        self.le(_as_term(rhs), lhs, guard=guard)

    def eq(self, lhs: Term, rhs: Term | int) -> None:
        """``lhs == rhs``.  The walk states one-directional bounds and never
        reaches for this; it is here for a constraint set built by hand."""
        self._add(Constraint(Rel.EQ, lhs, (_as_term(rhs),)))

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
            self._add(Constraint(Rel.LE_MAX, lhs, terms, guard))

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

        Memoized: the same term is asked many times over an unchanged store.
        """
        assuming = self.closed(assuming)
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
        assuming = self.closed(assuming)
        key = (tuple(bounds), assuming)
        answer = self._decisions.get(key)
        if answer is None:
            answer = any(self._at_least(t, k, assuming) for t, k in bounds)
            self._decisions[key] = answer
        return answer

    def _free(self, term: Term) -> bool:
        """Whether *term* names a variable no constraint mentions, which runs
        to both infinities and so leaves *term* unbounded either way."""
        return any(v.index not in self._naming for v, _ in term.coeffs)

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
