"""Unit tests for :func:`fpy2.strategies.split`.

The transform is tested in ``tests/unit/transform/test_split_loop.py``;
these tests pin the wrapper's factor handling and validation.
"""

import pytest

import fpy2 as fp

from fpy2.function import Function
from fpy2.strategies import simplify, split
from fpy2.transform import SplitLoopStrategy


K = 2  # captured by the str-factor fixture


@fp.fpy
def _total(xs: list[fp.Real]) -> fp.Real:
    acc = 0.0
    for x in xs:
        acc = acc + x
    return acc


@fp.fpy
def _total_k(xs: list[fp.Real]) -> fp.Real:
    acc = 0.0
    for x in xs:
        acc = acc + x * K
    return acc


_XS4 = [1.0, 2.0, 3.0, 4.0]


class TestSplit:

    def test_returns_function(self):
        out = split(_total, 2)
        assert isinstance(out, Function)
        for xs in ([], [1.0, 2.0], _XS4):
            assert _total(xs) == out(xs)

    def test_default_peel_any_length(self):
        out = split(_total, 2)
        for xs in ([1.0], [1.0, 2.0, 3.0]):
            assert _total(xs) == out(xs)

    def test_strict_asserts(self):
        out = split(_total, 2, strategy=SplitLoopStrategy.STRICT)
        with pytest.raises(AssertionError):
            out([1.0, 2.0, 3.0])

    def test_str_factor(self):
        # the factor is a variable resolved at runtime (here, the
        # captured global `K`)
        out = split(_total_k, 'K')
        assert _total_k(_XS4) == out(_XS4)

    def test_with_simplify(self):
        out = simplify(split(_total, 2))
        assert _total(_XS4) == out(_XS4)

    def test_factor_validation(self):
        with pytest.raises(ValueError):
            split(_total, 0)
        with pytest.raises(ValueError):
            split(_total, -2)
        with pytest.raises(TypeError):
            split(_total, 1.5)  # type: ignore[arg-type]

    def test_where_out_of_range(self):
        with pytest.raises(ValueError):
            split(_total, 2, 3)

    def test_type_error(self):
        with pytest.raises(TypeError):
            split(_total.ast, 2)  # type: ignore[arg-type]  # FuncDef, not Function
