"""Unit tests for :func:`fpy2.strategies.simplify_if`.

The rewrite itself is covered in ``tests/unit/transform/test_simplify_if.py``.
What is asserted here is the strategy layer: a `Function` in and out, and the
keyword reaching the transform.
"""

import pytest

import fpy2 as fp
from fpy2 import Function
from fpy2.ast.fpyast import ReturnStmt
from fpy2.transform.cursor import stmt_sites
import fpy2.strategies as st
from fpy2.strategies import (
    TransformReferenceError,
    simplify_if,
)


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


class TestTheKeywordIsForwarded:
    def test_strict_declines_it(self):
        assert not simplify_if(_guarded_read).ast.is_equiv(_guarded_read.ast)
        assert simplify_if(_guarded_read, strict=True).ast.is_equiv(_guarded_read.ast)


class TestWhereThroughTheStrategyLayer:
    def test_refusals_are_reported_by_the_generic_lister(self):
        refused = st.refusals(simplify_if, _asserts)
        assert len(refused) == 1 and 'assert' in refused[0][1]

    def test_strict_is_still_forwarded_alongside_where(self):
        """`strict` decides what is a site, so it reaches the pass before the
        index is resolved: without it `where=0` succeeds, with it there is no
        site 0 at all."""
        assert isinstance(simplify_if(_guarded_read, 0), Function)
        with pytest.raises(TransformReferenceError, match='subscript'):
            simplify_if(_guarded_read, 0, strict=True)


@fp.fpy
def _around_an_if(x: fp.Real) -> fp.Real:
    a = x + 1.0
    if x > 0:
        b = 1.0
    else:
        b = 2.0
    return a + b


class TestCursorsForwardThroughTheLayer:
    def _ret(self):
        return stmt_sites(_around_an_if.ast, lambda s: isinstance(s, ReturnStmt))[0]

    def test_it_forwards_past_the_growth(self):
        g = simplify_if(_around_an_if)
        cursor = self._ret()
        assert g.forward(cursor).path.index > cursor.path.index

    def test_it_chains_across_two_applications(self):
        """The second pass finds no `if`, so its log is empty -- which is the
        identity, not a broken chain."""
        g = simplify_if(_around_an_if)
        h = simplify_if(g)
        assert h.forward(self._ret()).path.index == g.forward(self._ret()).path.index
