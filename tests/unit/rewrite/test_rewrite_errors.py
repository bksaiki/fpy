"""
A user rewrite fails in the same vocabulary as a built-in strategy.

`except TransformError` has to catch a rewrite that named no place, or a
try/fallback schedule cannot mix user rewrites with the built-in operators.
"""

import pytest

import fpy2 as fp

from fpy2.rewrite import Rewrite
from fpy2.rewrite.applier import SubstitutionError
from fpy2.strategies import (
    TransformError,
    TransformReferenceError,
    unfold_special,
)


@fp.pattern
def fma_l(a, b, c):
    a * b + c


@fp.pattern
def fma_r(a, b, c):
    fp.fma(a, b, c)


@fp.pattern
def extra_var_r(a, b, c, d):
    fp.fma(a, b, c) + d


fma = Rewrite(fma_l, fma_r)


@fp.fpy
def twice(x, y, z):
    t = x * y + z
    u = z * y + x
    return t + u


@fp.fpy
def plain(x):
    return x + 1


def test_a_pattern_that_matches_nothing_is_a_bad_reference():
    with pytest.raises(TransformReferenceError, match='matches nothing'):
        fma.apply(plain)


def test_the_failure_is_catchable_as_a_transform_error():
    """The point of the phase: one `except` covers a user rewrite and a
    built-in."""
    for act in (lambda: fma.apply(plain),
                lambda: unfold_special(plain, where=3)):
        with pytest.raises(TransformError):
            act()


def test_an_index_past_the_end_says_how_many_matched():
    with pytest.raises(TransformReferenceError, match='where=5') as exc:
        fma.apply(twice, 5)
    assert 'matches 2 place(s)' in str(exc.value)
    assert 'fma_l' in str(exc.value)


def test_the_two_failures_are_distinguishable():
    """Matching nothing and naming the wrong match are different mistakes."""
    with pytest.raises(TransformReferenceError) as nothing:
        fma.apply(plain)
    with pytest.raises(TransformReferenceError) as wrong:
        fma.apply(twice, 5)
    assert 'matches nothing' in str(nothing.value)
    assert 'does not correspond' in str(wrong.value)


def test_apply_all_reports_the_same_way():
    with pytest.raises(TransformReferenceError, match='matches nothing'):
        Rewrite(fma_l, fma_r).apply(plain)


def test_a_malformed_rule_is_not_a_transform_error():
    """A replacement whose variables the pattern never binds is a bug in the
    rule, not a bad reference to a program, so a fallback schedule must not
    swallow it.  (An *unbound* name cannot even be written: the pattern's own
    syntax check rejects it.)"""
    bad = Rewrite(fma_l, extra_var_r)
    with pytest.raises(SubstitutionError):
        bad.apply(twice)
    assert not issubclass(SubstitutionError, TransformError)
