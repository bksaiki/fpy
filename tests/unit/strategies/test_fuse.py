"""Unit tests for :func:`fpy2.strategies.fuse`.

The transform is tested exhaustively in
``tests/unit/transform/test_reduce_fusion.py``; these tests pin the
wrapper's behavior.
"""

import pytest

import fpy2 as fp

from fpy2.ast import AllOf, AnyOf, ListComp
from fpy2.ast.visitor import DefaultVisitor
from fpy2.function import Function
from fpy2.strategies import fuse


def _has_node(ast, node_type) -> bool:
    """True iff any reachable expression in *ast* is a *node_type*."""
    found = [False]

    class _C(DefaultVisitor):
        def _visit_expr(self, e, ctx):
            if isinstance(e, node_type):
                found[0] = True
            super()._visit_expr(e, ctx)

    _C()._visit_function(ast, None)
    return found[0]


@fp.fpy
def _any_small(xs: list[fp.Real]) -> bool:
    return any([abs(x) < 1e-6 for x in xs])


@fp.fpy
def _all_pos(xs: list[fp.Real]) -> bool:
    return all([x > 0.0 for x in xs])


@fp.fpy
def _any_direct(bs: list[bool]) -> bool:
    # not a comprehension — no fusion
    return any(bs)


_SAMPLES: tuple[list[float], ...] = (
    [1.0, 2.0, 3.0],
    [1.0, 1e-7, 3.0],
    [-1.0, 2.0],
    [],
)


class TestFuse:

    def test_any_fused(self):
        out = fuse(_any_small)
        assert isinstance(out, Function)
        assert not _has_node(out.ast, AnyOf)
        assert not _has_node(out.ast, ListComp)
        for xs in _SAMPLES:
            assert _any_small(xs) == out(xs)
        # the input is not mutated
        assert _has_node(_any_small.ast, AnyOf)

    def test_all_fused(self):
        out = fuse(_all_pos)
        assert not _has_node(out.ast, AllOf)
        assert not _has_node(out.ast, ListComp)
        for xs in _SAMPLES:
            assert _all_pos(xs) == out(xs)

    def test_non_comp_noop(self):
        out = fuse(_any_direct)
        assert out.ast.is_equiv(_any_direct.ast)

    def test_type_error(self):
        with pytest.raises(TypeError):
            fuse(_any_small.ast)  # type: ignore[arg-type]  # FuncDef, not Function
