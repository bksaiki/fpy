"""How little :class:`fpy2.transform.Hoistable` does to the corpus, pinned.

Two directions no correctness test can see:

**Too little.**  The pass never raises: a position it cannot give a statement
slot is left as it is, and `refusals` reports it.  So a regression that stopped
lowering something would still be a correct program -- just no longer hoistable.
:data:`EXPECTED_RESIDUE` is that direction.

**Too much.**  The pass exists to be *weaker* than :class:`~fpy2.transform.ANF`,
which establishes the same invariant by also binding every nameable
subexpression to a name.  A regression that started atomizing would pass every
other test in this directory.  :data:`EXPECTED_GROWTH` is that direction, and it
is the number that says the pass is worth having.
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
"""Statements the pass adds over the whole corpus -- one per lowering, plus the
second copy of each rotated condition.

ANF adds `EXPECTED_ANF_GROWTH` for the same invariant.  That ratio is why this
pass exists; if it ever approaches ANF's, the pass has stopped being weak and
callers should use ANF instead.
"""

EXPECTED_ANF_GROWTH = 324
"""Context for :data:`EXPECTED_GROWTH`, not a claim about ANF.  Pinned so the
comparison cannot quietly stop holding."""


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
    before = grown = anf_grown = 0
    for f in _corpus():
        n += 1
        out = Hoistable.apply(f.ast)        # never raises; that is the claim
        for _e, why in Hoistable.refusals(out):
            residue[why] = residue.get(why, 0) + 1
        for _e, why in Hoistable.refusals(Hoistable.apply(CompToLoop.apply(f.ast))):
            after[why] = after.get(why, 0) + 1
        before += _size(f.ast)
        grown += _size(out)
        anf_grown += _size(ANF.apply(f.ast))
    return n, residue, after, (before, grown, anf_grown)


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
    """The load-bearing measurement.  ANF establishes the same invariant on the
    same corpus for five times the statements."""
    _n, _residue, _after, (before, grown, anf_grown) = _PROFILE
    assert before == EXPECTED_STATEMENTS
    assert grown - before == EXPECTED_GROWTH
    assert anf_grown - before == EXPECTED_ANF_GROWTH
