"""
Unit tests for the bundling transforms.

`WhileBundling`, `ForBundling` and `IfBundling` each pack the variables a block
mutates into one tuple, so a loop or branch carries a single value.  The
FPCore backend requires it -- it can compile at most one mutated variable per
block -- and was until now the only thing exercising them.

The packing order is asserted **sorted** rather than merely self-consistent.
It used to come from iterating a `set`, which is consistent within a run and
differs between them: the tuple's element order and each name's index in it
both followed hash order, so the emitted program was not reproducible across
processes.  `sorted` cannot be tested by varying `PYTHONHASHSEED` in-process,
so the property is pinned directly.
"""

import pytest

import fpy2 as fp
from fpy2 import Function
from fpy2.ast.fpyast import TupleExpr, Var
from fpy2.ast.visitor import DefaultVisitor
from fpy2.transform import ForBundling, IfBundling, WhileBundling


def _first_tuple_names(ast) -> list[str]:
    """The names packed by the first tuple the pass emitted, in order."""
    found: list[list[str]] = []

    class _V(DefaultVisitor):
        def _visit_tuple_expr(self, e: TupleExpr, ctx):
            if all(isinstance(x, Var) for x in e.elts):
                found.append([str(x.name) for x in e.elts])
            super()._visit_tuple_expr(e, ctx)

    _V()._visit_function(ast, None)
    assert found, 'the pass emitted no tuple, so there is nothing to check'
    return found[0]


@fp.fpy
def while_loop(n: int):
    delta = 0.0
    beta = 1.0
    alpha = 2.0
    i = 0
    while i < n:
        delta = delta + 1.0
        beta = beta * 2.0
        alpha = alpha - 1.0
        i = i + 1
    return delta + beta + alpha


@fp.fpy
def for_loop(xs: list[fp.Real]):
    delta = 0.0
    beta = 1.0
    alpha = 2.0
    for x in xs:
        delta = delta + x
        beta = beta * x
        alpha = alpha - x
    return delta + beta + alpha


@fp.fpy
def if_stmt(c: fp.Real):
    delta = 0.0
    beta = 1.0
    alpha = 2.0
    if c > 0:
        delta = 5.0
        beta = 6.0
        alpha = 7.0
    return delta + beta + alpha


_CASES = [
    (WhileBundling, while_loop, [(0,), (1,), (4,)]),
    (ForBundling, for_loop, [([],), ([2.0],), ([2.0, 3.0, 4.0],)]),
    (IfBundling, if_stmt, [(1.0,), (-1.0,), (0.0,)]),
]
_IDS = [p.__name__ for p, _f, _a in _CASES]


class TestThePackingOrderIsDeterministic:
    """Declared source order is `delta, beta, alpha`, so a sorted packing is
    distinguishable from both source order and (in practice) hash order."""

    @pytest.mark.parametrize('pas,f,_args', _CASES, ids=_IDS)
    def test_names_are_packed_in_sorted_order(self, pas, f, _args):
        names = _first_tuple_names(pas.apply(f.ast))
        assert names == sorted(names)

    @pytest.mark.parametrize('pas,f,_args', _CASES, ids=_IDS)
    def test_the_order_is_not_merely_source_order(self, pas, f, _args):
        """Otherwise the assertion above would pass on an unsorted pass."""
        assert _first_tuple_names(pas.apply(f.ast)) != ['delta', 'beta', 'alpha']


class TestBundlingPreservesSemantics:
    @pytest.mark.parametrize('pas,f,args', _CASES, ids=_IDS)
    def test_agrees_with_the_interpreter(self, pas, f, args):
        g = Function(pas.apply(f.ast), runtime=f.runtime)
        for a in args:
            assert repr(g(*a)) == repr(f(*a))
