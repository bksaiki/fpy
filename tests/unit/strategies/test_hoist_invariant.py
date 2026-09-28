"""Unit tests for :func:`fpy2.strategies.hoist_invariant`.

The transform itself is tested in
``tests/unit/transform/test_hoist_invariant.py``; these pin the wrapper's
surface — how it is aimed, and how it fails when aimed at nothing.
"""

import fpy2 as fp
import pytest

from fpy2.strategies import (
    TransformReferenceError,
    hoist_invariant,
    sites,
)

from ..transform.test_hoist_invariant import _body_names

@fp.fpy(ctx=fp.REAL)
def two_loops(xs: list[fp.Real]) -> fp.Real:
    n = len(xs)
    a = 0.0
    for x in xs:
        p = n + 1
        a = a + p * x
    b = 0.0
    for x in xs:
        q = n + 2
        b = b + q * x
    return a + b


@fp.fpy(ctx=fp.REAL)
def nothing_to_hoist(xs: list[fp.Real]) -> fp.Real:
    acc = 0.0
    for x in xs:
        acc = acc + x * x
    return acc


_VALUES = ([], [1.0], [1.5, -2.0, 3.25])


def _agrees(func, out) -> bool:
    return all(repr(out(v)) == repr(func(v)) for v in _VALUES)


class TestAiming:
    def test_an_index_takes_one_loop(self):
        first = hoist_invariant(two_loops, 0)
        assert _body_names(first.ast) == {'a', 'q', 'b'}
        assert _agrees(two_loops, first)

        second = hoist_invariant(two_loops, 1)
        assert _body_names(second.ast) == {'p', 'a', 'b'}
        assert _agrees(two_loops, second)


class TestFailures:
    def test_a_cursor_from_another_program(self):
        where = sites(hoist_invariant, two_loops)[0]
        with pytest.raises(TransformReferenceError, match='unrelated program'):
            hoist_invariant(nothing_to_hoist, where)
