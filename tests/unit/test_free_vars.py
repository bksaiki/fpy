"""
Free-variable capture at decoration.

The decorator collects the names an FPy body may read from its defining Python
scope, including those read only inside a comprehension.
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
        # free variable
        @fp.fpy(ctx=fp.FP64)
        def f(xs):
            return [max(fp.logb(x), _K) for x in xs]

        assert f([8.0, 1.0]) == [3.0, 3.0]

    def test_iterable(self):
        @fp.fpy(ctx=fp.FP64)
        def f():
            return [x * 2 for x in range(_N)]

        assert f() == [0.0, 2.0, 4.0, 6.0]

    def test_doubly_nested(self):
        @fp.fpy(ctx=fp.FP64)
        def f(xs):
            return [[y + _K for y in xs] for _ in xs]

        assert f([1.0]) == [[4.0]]


class TestOuterScope:
    """Names read outside any comprehension."""

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
    """A genuine typo must still fail."""
    with pytest.raises(FPySyntaxError, match='unbound variable'):
        @fp.fpy(ctx=fp.FP64)
        def f(xs):
            return [x + _definitely_not_defined for x in xs]
