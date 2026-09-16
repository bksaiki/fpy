"""
Free-variable capture at decoration.

The decorator collects the names an FPy body may read from its defining Python
scope.  Before Python 3.12 a comprehension compiles to its own code object, so
reading one code object -- which is all :func:`inspect.getclosurevars` does --
missed every name used only inside ``[... for x in xs]``, and `SyntaxCheck`
reported it unbound.  These pin the names being found, on every version.
"""

import pytest

import fpy2 as fp
from fpy2.analysis.syntax_check import FPySyntaxError

_K = 3.0
_N = 4


class TestComprehensionScope:
    """A free variable read only inside a comprehension."""

    def test_element(self):
        @fp.fpy(ctx=fp.FP64)
        def f(xs):
            return [x + _K for x in xs]

        assert f([1.0, 2.0]) == [4.0, 5.0]

    def test_nested_call(self):
        # `fp.logb` inside a comprehension: the module alias `fp` is itself a
        # free variable, and was lost with the rest of the nested code object
        @fp.fpy(ctx=fp.FP64)
        def f(xs):
            return [max(fp.logb(x), _K) for x in xs]

        assert f([8.0, 1.0]) == [3.0, 3.0]

    def test_iterable(self):
        # the outermost iterable is evaluated in the *enclosing* scope, so this
        # arm worked already; kept so a fix cannot regress it
        @fp.fpy(ctx=fp.FP64)
        def f():
            return [x * 2 for x in range(_N)]

        assert f() == [0.0, 2.0, 4.0, 6.0]

    def test_doubly_nested(self):
        @fp.fpy(ctx=fp.FP64)
        def f(xs):
            return [[y + _K for y in xs] for _ in xs]

        assert f([1.0]) == [[4.0]]

    def test_only_use_is_inside(self):
        # the whole point: no other reference to `_K` exists to carry it
        @fp.fpy(ctx=fp.FP64)
        def f(xs):
            return sum([x * _K for x in xs])

        assert f([1.0, 2.0]) == 9.0


class TestOuterScope:
    """The cases that never depended on the walk."""

    def test_plain_reference(self):
        @fp.fpy(ctx=fp.FP64)
        def f(x):
            return x + _K

        assert f(1.0) == 4.0

    def test_closure_cell(self):
        def outer(k):
            @fp.fpy(ctx=fp.FP64)
            def inner(xs):
                return [x + k for x in xs]
            return inner

        assert outer(10.0)([1.0]) == [11.0]


def test_unbound_is_still_unbound():
    """The walk widens what resolves, so a genuine typo must still fail."""
    with pytest.raises(FPySyntaxError, match='unbound variable'):
        @fp.fpy(ctx=fp.FP64)
        def f(xs):
            return [x + _definitely_not_defined for x in xs]
