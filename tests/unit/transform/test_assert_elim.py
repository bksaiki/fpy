"""
Unit tests for :class:`fpy2.transform.AssertElim`.

The pass is a *semantic* change -- the program said to abort and the result
will not -- so the tests check that what remains is the assert-free program,
and that nothing is left in a shape FPy does not admit: there is no empty
block, so a statement whose body held only asserts has to go with them.
"""

import pytest

import fpy2 as fp
from fpy2 import Function
from fpy2.analysis import SyntaxCheck
from fpy2.transform import AssertElim


def _applied(f: Function) -> Function:
    return Function(AssertElim.apply(f.ast), runtime=f.runtime)


@fp.fpy(ctx=fp.FP64)
def plain(x: fp.Real):
    assert x > 0, 'positive'
    y = x * 2
    assert y > 0, 'still positive'
    return y


@fp.fpy(ctx=fp.FP64)
def only_assert_in_loop(xs: list[fp.Real]):
    s = fp.round(0)
    for x in xs:
        assert x > 0, 'positive'
    for x in xs:
        s = s + x
    return s


@fp.fpy(ctx=fp.FP64)
def only_assert_in_one_arm(c: bool, x: fp.Real):
    y = x
    if c:
        assert x > 0, 'positive'
    else:
        y = x * 2
    return y


def test_the_asserts_are_gone():
    assert 'assert' not in _applied(plain).format()


@pytest.mark.parametrize('x', [1.0, 3.0])
def test_meaning_is_kept_where_the_asserts_held(x):
    assert repr(_applied(plain)(x)) == repr(plain(x))


def test_it_no_longer_aborts():
    """The point of the pass: the program said to abort and this does not."""
    with pytest.raises(Exception):
        plain(-1.0)
    assert repr(_applied(plain)(-1.0)) == repr(fp.FP64.round(-2.0))


def test_a_loop_left_empty_goes_too():
    """FPy has no empty block, so the `for` cannot stay behind."""
    out = _applied(only_assert_in_loop)
    assert out.format().count('for x in xs') == 1
    assert repr(out([1.0, 2.0])) == repr(only_assert_in_loop([1.0, 2.0]))


def test_an_emptied_arm_becomes_a_one_armed_if():
    """A two-armed `if` that loses one arm cannot keep a hole."""
    out = _applied(only_assert_in_one_arm)
    SyntaxCheck.check(out.ast, ignore_unknown=True)
    for c in (True, False):
        assert repr(out(c, 3.0)) == repr(only_assert_in_one_arm(c, 3.0))


def test_a_function_with_none_is_unchanged_in_meaning():
    @fp.fpy(ctx=fp.FP64)
    def clean(x: fp.Real):
        return x * 2

    assert repr(_applied(clean)(3.0)) == repr(clean(3.0))
