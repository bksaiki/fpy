"""Unit tests for :func:`fpy2.strategies.inline`."""

import pytest

import fpy2 as fp

from fpy2.ast.visitor import DefaultVisitor
from fpy2.function import Function
from fpy2.strategies import inline


def _fpy_callees(ast) -> list[Function]:
    """Remaining calls to user-defined FPy functions, in visit order."""
    callees: list[Function] = []

    class _C(DefaultVisitor):
        def _visit_call(self, e, ctx):
            if isinstance(e.fn, Function):
                callees.append(e.fn)
            super()._visit_call(e, ctx)

    _C()._visit_function(ast, None)
    return callees


def _count_fpy_calls(ast) -> int:
    """Number of remaining calls to user-defined FPy functions."""
    return len(_fpy_callees(ast))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@fp.fpy
def _leaf(x: fp.Real) -> fp.Real:
    return x + 1


@fp.fpy
def _mid(x: fp.Real) -> fp.Real:
    return _leaf(x) * 2


@fp.fpy
def _top(x: fp.Real) -> fp.Real:
    # `_leaf` is shared: reached via `_mid` and directly.
    return _mid(x) + _leaf(x)


@fp.fpy
def _other(x: fp.Real) -> fp.Real:
    return x * 3


@fp.fpy
def _two_callees(x: fp.Real) -> fp.Real:
    return _leaf(x) + _other(x)


@fp.fpy
def _leaf_twice(x: fp.Real) -> fp.Real:
    # same callee at two sites — `where` can address them separately
    return _leaf(x) * _leaf(x + 1)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestInline:

    def test_returns_function(self):
        out = inline(_top)
        assert isinstance(out, Function)
        # the input is not mutated
        assert _count_fpy_calls(_top.ast) == 2

    def test_recursive_default_flattens(self):
        out = inline(_top)
        assert _count_fpy_calls(out.ast) == 0
        for x in (0.0, 1.5, -3.25, 10.0):
            assert _top(x) == out(x)

    def test_one_level(self):
        out = inline(_top, recursive=False)
        # `_mid`'s body still calls `_leaf`, plus the direct `_leaf` call
        # was inlined — one nested call remains.
        assert _count_fpy_calls(out.ast) == 1
        for x in (0.0, 1.5, -3.25, 10.0):
            assert _top(x) == out(x)

    def test_funcs_filter(self):
        out = inline(_two_callees, funcs=[_leaf])
        # `_leaf` inlined, `_other` untouched
        assert _count_fpy_calls(out.ast) == 1
        for x in (0.0, 1.5, -3.25, 10.0):
            assert _two_callees(x) == out(x)

    def test_funcs_filter_empty(self):
        out = inline(_two_callees, funcs=[])
        assert out.ast.is_equiv(_two_callees.ast)

    def test_no_calls_noop(self):
        out = inline(_leaf)
        assert out.ast.is_equiv(_leaf.ast)

    def test_type_errors(self):
        with pytest.raises(TypeError):
            inline(_top.ast)  # FuncDef, not Function
        with pytest.raises(TypeError):
            inline(_two_callees, funcs=[_leaf.ast])
        with pytest.raises(TypeError):
            inline(_top, 'x')


class TestInlineWhere:

    def test_where_selects_site(self):
        # `_top`'s candidate sites in visit order: `_mid(x)`, `_leaf(x)`
        out = inline(_top, 0)
        assert _fpy_callees(out.ast) == [_leaf]
        for x in (0.0, 1.5, -3.25, 10.0):
            assert _top(x) == out(x)

        out = inline(_top, 1)
        assert _fpy_callees(out.ast) == [_mid]
        for x in (0.0, 1.5, -3.25, 10.0):
            assert _top(x) == out(x)

    def test_where_same_callee_two_sites(self):
        out = inline(_leaf_twice, 1)
        assert _fpy_callees(out.ast) == [_leaf]
        for x in (0.0, 1.5, -3.25, 10.0):
            assert _leaf_twice(x) == out(x)

    def test_where_selected_site_flattens(self):
        # `recursive=True` (default): inlining the `_mid` site also
        # flattens `_mid`'s own call to `_leaf`
        out = inline(_top, 0)
        assert _count_fpy_calls(out.ast) == 1

    def test_where_with_recursive_false(self):
        # one level only: `_mid`'s body still calls `_leaf`, and the
        # direct `_leaf` site is not selected
        out = inline(_top, 0, recursive=False)
        assert _fpy_callees(out.ast) == [_leaf, _leaf]
        for x in (0.0, 1.5, -3.25, 10.0):
            assert _top(x) == out(x)

    def test_where_with_funcs(self):
        # candidates are only calls to `_other`, so index 0 is the
        # `_other` site even though a `_leaf` call precedes it
        out = inline(_two_callees, 0, funcs=[_other])
        assert _fpy_callees(out.ast) == [_leaf]
        for x in (0.0, 1.5, -3.25, 10.0):
            assert _two_callees(x) == out(x)

    def test_where_out_of_range(self):
        with pytest.raises(ValueError):
            inline(_top, 2)
        with pytest.raises(ValueError):
            inline(_top, -1)
        with pytest.raises(ValueError):
            inline(_two_callees, 1, funcs=[_other])
