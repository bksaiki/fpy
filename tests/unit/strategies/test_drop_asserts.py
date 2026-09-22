"""Unit tests for :func:`fpy2.strategies.drop_asserts`."""

import pytest

import fpy2 as fp

from fpy2.strategies import drop_asserts


@fp.fpy(ctx=fp.FP64)
def _recip(x: fp.Real) -> fp.Real:
    assert x != 0, 'nonzero'
    return 1 / x


@fp.fpy(ctx=fp.FP64)
def _checks_then_sums(xs: list[fp.Real]) -> fp.Real:
    s = fp.round(0)
    for x in xs:
        assert x > 0, 'positive'
    for x in xs:
        s = s + x
    return s


def test_the_assertions_are_gone():
    assert 'assert' not in drop_asserts(_recip).format()


def test_it_agrees_where_the_assertions_held():
    assert repr(drop_asserts(_recip)(4.0)) == repr(_recip(4.0))


def test_it_no_longer_aborts():
    """The documented consequence: an input that aborted now returns."""
    with pytest.raises(AssertionError):
        _recip(0.0)
    # the target's own answer for dividing by zero, divzero flag and all
    assert float(drop_asserts(_recip)(0.0)) == float('inf')


def test_a_loop_that_only_checked_is_removed():
    """FPy admits no empty block, so the loop goes with its assertion."""
    out = drop_asserts(_checks_then_sums)
    assert out.format().count('for x in xs') == 1
    assert repr(out([1.0, 2.0])) == repr(_checks_then_sums([1.0, 2.0]))


def test_it_rejects_a_non_function():
    with pytest.raises(TypeError, match='Function'):
        drop_asserts(42)
