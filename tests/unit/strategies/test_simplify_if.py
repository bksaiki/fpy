"""Unit tests for :func:`fpy2.strategies.simplify_if`.

The rewrite itself is covered in ``tests/unit/transform/test_simplify_if.py``.
What is asserted here is the strategy layer: a `Function` in and out, the
keyword reaching the transform, and a refusal arriving as the shared
:class:`TransformDeclined` rather than something the layer invented.
"""

import pytest

import fpy2 as fp
from fpy2 import Function
from fpy2.strategies import TransformDeclined, simplify_if


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
