"""How much of the corpus :class:`fpy2.transform.ANF` leaves un-flattened, pinned.

The pass never raises: a position it cannot give a statement slot is left as it
is, and :func:`fpy2.transform.anf.refusals` reports it.  So a regression that
stopped flattening something would be invisible to every correctness test --
the program still means the same thing, it is just no longer normal.  This is
the missing direction, and it fails in both: a change that flattens *less* than
today, and a change that flattens more without anyone deciding to.

The load-bearing assertion is :data:`EXPECTED_DANGEROUS`.  A ternary arm, a
short-circuited operand and a ``while`` condition are the three positions the cpp
emitter cannot slot -- each is a miscompile recorded in
``docs/todos/backend-cpp.md`` -- and the lowerings exist to empty them.  A
comprehension is different: the emitter gives its element the loop body it
generates and its iterable the ``for`` header, so a refusal there is a shape
this pass declines to normalize rather than a shape anything gets wrong.
"""

import importlib

import fpy2 as fp
from fpy2.transform import ANF
from tests.infra.examples import all_example_tests, all_unit_tests

EXPECTED_FUNCTIONS = 230
"""Corpus size.  An empty result only means something while this holds."""

EXPECTED_RESIDUE = {
    "a comprehension's iterable may read an earlier target": 18,
    "a comprehension's element runs once per iteration": 7,
}
"""Sealed positions left holding something that needs a place, by reason.

Every one is a comprehension.  An iterable is a list by definition, so hoisting
it would create the aggregate name the pass exists to avoid; an element runs
once per iteration, which only `CompToLoop` can give a slot.
"""

EXPECTED_DANGEROUS = 0
"""Refusals in a position no backend can slot.  Must stay zero."""

_DANGEROUS = (
    'a ternary arm is evaluated conditionally',
    'a short-circuited operand may not be evaluated',
    'a `while` condition is re-evaluated every iteration',
)


def _corpus():
    yield from all_unit_tests()
    yield from all_example_tests()
    for name in ('core', 'eft', 'vector', 'matrix'):
        mod = importlib.import_module(f'fpy2.libraries.{name}')
        for f in mod.__dict__.values():
            if isinstance(f, fp.Function):
                yield f


def _profile():
    residue: dict[str, int] = {}
    n = 0
    for f in _corpus():
        n += 1
        out = ANF.apply(f.ast)          # never raises; that is the claim
        for _e, why in ANF.refusals(out):
            residue[why] = residue.get(why, 0) + 1
    return n, residue


def test_anf_applies_to_the_whole_corpus():
    """The pass is total in the sense that matters: it declines to normalize a
    position, never to accept a program."""
    n, _ = _profile()
    assert n == EXPECTED_FUNCTIONS


def test_no_refusal_in_a_position_no_backend_can_slot():
    _, residue = _profile()
    dangerous = sum(residue.get(why, 0) for why in _DANGEROUS)
    assert dangerous == EXPECTED_DANGEROUS, (
        f'{dangerous} refusal(s) in a position the cpp emitter hoists out of; '
        f'see the three miscompiles in docs/todos/backend-cpp.md'
    )


def test_the_comprehension_residue_is_what_it_was():
    _, residue = _profile()
    assert residue == EXPECTED_RESIDUE
