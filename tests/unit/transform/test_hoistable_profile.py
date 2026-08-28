"""How little :class:`fpy2.transform.Hoistable` does to the corpus, pinned.

Two directions no correctness test can see:

**Too little.**  The pass never raises: a position it cannot give a statement
slot is left as it is, and `refusals` reports it.  So a regression that stopped
lowering something would still be a correct program -- just no longer hoistable.
:data:`EXPECTED_RESIDUE` is that direction.

**Too much.**  The pass is the *weak* half of normalizing a program:
:class:`~fpy2.transform.ANF` is the strong half, and requires this one to have
run.  A regression that started atomizing here would pass every other test in
this directory.  :data:`EXPECTED_GROWTH` is that direction -- and set beside
:data:`EXPECTED_ATOMIZATION_GROWTH`, it is the number that says splitting the two
was worth it: a rewrite that needs a statement slot pays the first and not the
second.
"""

import importlib

import fpy2 as fp
from fpy2.transform import ANF, CompToLoop, Hoistable
from fpy2.transform.path import walk_stmts
from tests.infra.examples import all_example_tests, all_unit_tests

EXPECTED_FUNCTIONS = 230
"""Corpus size.  A count only means something while this holds."""

_ELEMENT = "a comprehension's element runs once per iteration"
_ITERABLE = "a comprehension's iterable may read an earlier target"

EXPECTED_RESIDUE = {_ELEMENT: 20, _ITERABLE: 16}
"""Sealed positions left holding a non-atom, by reason.

Every one is a comprehension, which is the whole claim: the other three sealed
positions -- a ternary arm, a short-circuited operand, a `while` condition --
are emptied outright, and each is a miscompile recorded in
``docs/todos/backend-cpp.md`` when a backend meets one.
"""

EXPECTED_RESIDUE_AFTER_COMP_TO_LOOP = {_ITERABLE: 3, _ELEMENT: 2}
"""What is left once `CompToLoop` has run first, as a caller is told to.

Only the comprehensions that pass declines: a dependent clause list, whose
length is a sum rather than a product, so `fp.empty` has nowhere to get it.
"""

EXPECTED_STATEMENTS = 831
EXPECTED_GROWTH = 66
"""Statements this pass adds over the whole corpus -- one per lowering, plus the
second copy of each rotated condition.

The whole cost of being able to hoist anywhere.  Compare
:data:`EXPECTED_ATOMIZATION_GROWTH`, which is what `ANF` adds *on top* for its
much stronger invariant.  There is no measuring `ANF` alone any more: it requires
hoistable form and raises without it, which is the point of the split.
"""

EXPECTED_ATOMIZATION_GROWTH = 295
"""What `ANF` adds on top of hoistable form.

Context for :data:`EXPECTED_GROWTH`, not a claim about `ANF`.  Pinned so the
comparison cannot quietly stop holding.
"""


def _corpus():
    yield from all_unit_tests()
    yield from all_example_tests()
    for name in ('core', 'eft', 'vector', 'matrix'):
        mod = importlib.import_module(f'fpy2.libraries.{name}')
        for f in mod.__dict__.values():
            if isinstance(f, fp.Function):
                yield f


def _size(func) -> int:
    return sum(1 for _path, _stmt in walk_stmts(func))


def _profile():
    n = 0
    residue: dict[str, int] = {}
    after: dict[str, int] = {}
    before = grown = atomized = 0
    for f in _corpus():
        n += 1
        out = Hoistable.apply(f.ast)        # never raises; that is the claim
        for _e, why in Hoistable.refusals(out):
            residue[why] = residue.get(why, 0) + 1
        for _e, why in Hoistable.refusals(Hoistable.apply(CompToLoop.apply(f.ast))):
            after[why] = after.get(why, 0) + 1
        before += _size(f.ast)
        grown += _size(out)
        # `ANF` on the *output*, not the input: it requires this pass to have run
        atomized += _size(ANF.apply(out))
    return n, residue, after, (before, grown, atomized)


_PROFILE = _profile()
"""Measured once: the corpus is 230 functions and four passes over each."""


def test_the_pass_applies_to_the_whole_corpus():
    """Total in the sense that matters: it declines to normalize a position,
    never to accept a program."""
    n, _residue, _after, _sizes = _PROFILE
    assert n == EXPECTED_FUNCTIONS


def test_only_a_comprehension_is_left_unhoistable():
    _n, residue, _after, _sizes = _PROFILE
    assert residue == EXPECTED_RESIDUE


def test_comp_to_loop_first_leaves_only_what_it_declined():
    _n, _residue, after, _sizes = _PROFILE
    assert after == EXPECTED_RESIDUE_AFTER_COMP_TO_LOOP


def test_the_pass_stays_weak():
    """The load-bearing measurement.  Being able to hoist anywhere costs 66
    statements; the atomization stacked on top of it costs four times that, and a
    rewrite needs only the first."""
    _n, _residue, _after, (before, grown, atomized) = _PROFILE
    assert before == EXPECTED_STATEMENTS
    assert grown - before == EXPECTED_GROWTH
    assert atomized - grown == EXPECTED_ATOMIZATION_GROWTH
