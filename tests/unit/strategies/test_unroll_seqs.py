"""Unit tests for :func:`fpy2.strategies.unroll_seqs`."""

import pytest

import fpy2 as fp

from fpy2.ast import fpyast as A
from fpy2.ast.visitor import DefaultVisitor
from fpy2.strategies import inline, unroll_seqs


def _comps(f) -> int:
    n = 0

    class _C(DefaultVisitor):
        def _visit_expr(self, e, ctx):
            nonlocal n
            if isinstance(e, A.ListComp):
                n += 1
            super()._visit_expr(e, ctx)

    _C()._visit_function(f.ast, None)
    return n


@fp.fpy
def _weighted(xs: list[fp.Real]) -> fp.Real:
    ys = [xs[i] * 2 for i in range(3)]
    return sum(ys)


@fp.fpy
def _bump(x: fp.Real) -> fp.Real:
    t = x + 1
    return t


@fp.fpy
def _calls(xs: list[fp.Real]) -> fp.Real:
    ys = [_bump(xs[i]) for i in range(3)]
    return ys[0] + ys[2]


def test_it_unrolls():
    assert _comps(unroll_seqs(_weighted)) == 0


def test_the_list_survives_for_its_uses():
    """`sum(ys)` still has a `ys` to fold."""
    assert 'ys = [' in unroll_seqs(_weighted).format()


def test_meaning_is_preserved():
    args = [1.0, 2.0, 3.0]
    assert repr(unroll_seqs(_weighted)(args)) == repr(_weighted(args))


def test_it_makes_a_call_reachable_by_inline():
    """The documented reason the strategy exists."""
    assert _comps(inline(_calls)) == 1, 'inline alone cannot reach it'
    assert _comps(unroll_seqs(_calls)) == 0
    args = [1.0, 2.0, 3.0]
    both = inline(unroll_seqs(_calls))
    assert repr(both(args)) == repr(_calls(args))


def test_the_cap_is_honoured():
    assert _comps(unroll_seqs(_weighted, cap=2)) == 1
    assert _comps(unroll_seqs(_weighted, cap=3)) == 0


def test_over_the_cap_is_not_a_refusal():
    """Declining to unroll leaves a working program, not an error."""
    args = [1.0, 2.0, 3.0]
    assert repr(unroll_seqs(_weighted, cap=2)(args)) == repr(_weighted(args))


def test_it_rejects_a_non_function():
    with pytest.raises(TypeError, match='Function'):
        unroll_seqs(42)
