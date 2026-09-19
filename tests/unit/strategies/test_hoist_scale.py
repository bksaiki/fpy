"""Unit tests for :func:`fpy2.strategies.hoist_scale`.

The transform itself is tested in
``tests/unit/transform/test_hoist_scale.py``; these pin the wrapper's surface —
how it is aimed, and how it fails when aimed at nothing.
"""

import fpy2 as fp
import pytest

from fpy2.function import Function
from fpy2.strategies import (
    TransformDeclined,
    TransformReferenceError,
    hoist_scale,
    refusals,
    sites,
)


@fp.fpy(ctx=fp.REAL)
def two_sums(xs: list[fp.Real], ys: list[fp.Real], k: fp.Real) -> fp.Real:
    if fp.isfinite(k):
        a = sum([(2 ** k) * x for x in xs])
        b = sum([(2 ** k) * y for y in ys])
        return a + b
    else:
        return 0.0


@fp.fpy(ctx=fp.FP32)
def rounds_between_adds(xs: list[fp.Real], k: fp.Real) -> fp.Real:
    if fp.isfinite(k):
        return sum([(2 ** k) * x for x in xs])
    else:
        return 0.0


_VALUES = ([], [1.0], [1.5, -2.0, 3.25])


def _agrees(func, out) -> bool:
    return all(
        repr(out(xs, ys, 3.0)) == repr(func(xs, ys, 3.0))
        for xs in _VALUES for ys in _VALUES
    )


def _hoisted_count(ast) -> int:
    """How many reductions are left unscaled — `sum([x for x in …])`."""
    return ' '.join(ast.format().split()).count('* sum(')


class TestTheWrapper:

    def test_it_returns_a_function_and_leaves_the_input_alone(self):
        out = hoist_scale(two_sums)
        assert isinstance(out, Function)
        assert _hoisted_count(two_sums.ast) == 0

    def test_it_rejects_a_non_function(self):
        with pytest.raises(TypeError):
            hoist_scale(two_sums.ast)

    def test_none_rewrites_every_reduction(self):
        out = hoist_scale(two_sums)
        assert _hoisted_count(out.ast) == 2
        assert _agrees(two_sums, out)


class TestAiming:

    def test_an_index_takes_one_reduction(self):
        for i in (0, 1):
            out = hoist_scale(two_sums, i)
            assert _hoisted_count(out.ast) == 1
            assert _agrees(two_sums, out)

    def test_a_cursor_takes_the_reduction_it_names(self):
        where = sites(hoist_scale, two_sums)
        assert len(where) == 2
        out = hoist_scale(two_sums, where[1])
        assert _hoisted_count(out.ast) == 1
        assert _agrees(two_sums, out)

    def test_a_reduction_that_rounds_is_no_site(self):
        assert sites(hoist_scale, rounds_between_adds) == []


class TestFailures:

    def test_an_index_past_the_end(self):
        with pytest.raises(TransformReferenceError, match='does not correspond'):
            hoist_scale(two_sums, 2)

    def test_a_cursor_naming_a_refused_reduction_says_why(self):
        (where, why), = refusals(hoist_scale, rounds_between_adds)
        assert why == 'the reduction does not round exactly'
        with pytest.raises(TransformDeclined, match='round exactly'):
            hoist_scale(rounds_between_adds, where)

    def test_a_cursor_from_another_program(self):
        where = sites(hoist_scale, two_sums)[0]
        with pytest.raises(TransformReferenceError, match='unrelated program'):
            hoist_scale(rounds_between_adds, where)
