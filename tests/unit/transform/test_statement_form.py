"""`StatementForm`: every comprehension and derived iterable lowered."""

import pytest

import fpy2 as fp
from fpy2 import Function
from fpy2.transform import Simplify, StatementForm


@fp.fpy(ctx=fp.FP64)
def _f(xs: list[fp.Real]) -> list[fp.Real]:
    return [x * 2.0 for x in xs]


class TestSimplify:
    def test_off_by_default(self):
        """The lowering binds its iterable, `t = xs`, which `Simplify` clears."""
        out = StatementForm.apply(_f.ast)
        assert out.format() != Simplify.apply(out).format()

    def test_on_it_is_the_simplified_form(self):
        plain = StatementForm.apply(_f.ast)
        out = StatementForm.apply(_f.ast, simplify=True)
        assert out.format() == Simplify.apply(plain).format()
        xs = [1.5, -2.0]
        assert repr(Function(out, runtime=_f.runtime)(xs)) == repr(_f(xs))


@fp.fpy
def _needs_positive_g(x: fp.Real) -> fp.Real:
    assert x > 0.0, 'g'
    return x


@fp.fpy
def _needs_positive_h(x: fp.Real) -> fp.Real:
    assert x > 0.0, 'h'
    return x


class TestOrdering:
    def test_a_left_operand_is_not_overtaken_by_a_comprehension(self):
        """`CompToLoop` hoists the comprehension above the statement, so
        without naming `g(a)` it would run `h` first, and both raise."""
        @fp.fpy
        def f(a: fp.Real, xs: list[fp.Real]) -> fp.Real:
            return _needs_positive_g(a) + len([_needs_positive_h(x) for x in xs])

        out = Function(StatementForm.apply(f.ast), runtime=f.runtime)
        args = [-1.0, [-1.0]]
        with pytest.raises(AssertionError) as before:
            f(*args)
        with pytest.raises(AssertionError) as after:
            out(*args)
        assert 'g' in str(before.value)
        assert str(after.value) == str(before.value)
