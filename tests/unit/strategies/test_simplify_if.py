"""Unit tests for :func:`fpy2.strategies.simplify_if`.

The rewrite itself is covered in ``tests/unit/transform/test_simplify_if.py``.
What is asserted here is the strategy layer: a `Function` in and out, the
keyword reaching the transform, and a refusal arriving as the shared
:class:`TransformDeclined` rather than something the layer invented.
"""

import pytest

import fpy2 as fp
from fpy2 import Function
import fpy2.strategies as st
from fpy2.strategies import (
    TransformDeclined,
    TransformReferenceError,
    simplify_if,
)


@fp.fpy
def _two_armed(x: fp.Real) -> fp.Real:
    if x > 0:
        y = x * 2
    else:
        y = -x
    return y


@fp.fpy
def _guarded_read(xs: list[fp.Real], i: int) -> fp.Real:
    if i < len(xs):
        y = xs[i]
    else:
        y = 0.0
    return y


@fp.fpy
def _asserts(x: fp.Real) -> fp.Real:
    if x > 0:
        assert x > 10, 'too small'
        y = x
    else:
        y = 0.0
    return y


class TestTheStrategyLayer:
    def test_returns_a_function(self):
        assert isinstance(simplify_if(_two_armed), Function)

    @pytest.mark.parametrize('x', [3.0, -3.0, 0.0])
    def test_semantics_are_preserved(self, x):
        assert repr(simplify_if(_two_armed)(x)) == repr(_two_armed(x))


class TestTheKeywordIsForwarded:
    def test_default_hoists_a_guarded_read(self):
        assert isinstance(simplify_if(_guarded_read), Function)

    def test_strict_declines_it(self):
        with pytest.raises(TransformDeclined):
            simplify_if(_guarded_read, strict=True)


class TestRefusalsCrossTheLayer:
    @pytest.mark.parametrize('strict', [False, True])
    def test_an_unconditional_refusal_surfaces_as_transform_declined(self, strict):
        """`TransformDeclined` is the shared hierarchy, so one `except` covers
        a strategy and a raw transform alike."""
        with pytest.raises(TransformDeclined, match='assert'):
            simplify_if(_asserts, strict=strict)


@fp.fpy
def _two_ifs(x: fp.Real, y: fp.Real) -> fp.Real:
    if x > 0:
        a = 1.0
    else:
        a = 2.0
    if y > 0:
        b = 3.0
    else:
        b = 4.0
    return a + b


class TestWhereThroughTheStrategyLayer:
    def test_sites_are_reported_by_the_generic_lister(self):
        assert len(st.sites(simplify_if, _two_ifs)) == 2

    def test_refusals_are_reported_by_the_generic_lister(self):
        refused = st.refusals(simplify_if, _asserts)
        assert len(refused) == 1 and 'assert' in refused[0][1]

    @pytest.mark.parametrize('where', [0, 1])
    def test_an_index_is_forwarded(self, where):
        g = simplify_if(_two_ifs, where)
        for x in (1.0, -1.0):
            for y in (1.0, -1.0):
                assert repr(g(x, y)) == repr(_two_ifs(x, y))

    def test_a_cursor_is_forwarded(self):
        cursor = st.sites(simplify_if, _two_ifs)[1]
        g = simplify_if(_two_ifs, cursor)
        assert repr(g(1.0, 1.0)) == repr(_two_ifs(1.0, 1.0))

    def test_strict_is_still_forwarded_alongside_where(self):
        """`strict` decides what is a site, so it reaches the pass before the
        index is resolved: without it `where=0` succeeds, with it there is no
        site 0 at all."""
        assert isinstance(simplify_if(_guarded_read, 0), Function)
        with pytest.raises(TransformReferenceError, match='subscript'):
            simplify_if(_guarded_read, 0, strict=True)
