"""Unit tests for :func:`fpy2.strategies.hoist_invariant`.

The transform itself is tested in
``tests/unit/transform/test_hoist_invariant.py``; these pin the wrapper's
surface — how it is aimed, and how it fails when aimed at nothing.
"""

import fpy2 as fp
import pytest

from fpy2.ast import Assign, ForStmt, StmtBlock
from fpy2.ast.visitor import DefaultVisitor
from fpy2.function import Function
from fpy2.strategies import (
    TransformDeclined,
    TransformReferenceError,
    hoist_invariant,
    refusals,
    sites,
)
from fpy2.utils import NamedId


def _block_names(block: StmtBlock) -> set[str]:
    return {
        str(s.target) for s in block.stmts
        if isinstance(s, Assign) and isinstance(s.target, NamedId)
    }


def _body_names(ast) -> set[str]:
    """Names assigned directly in some loop body."""
    found: set[str] = set()

    class V(DefaultVisitor):
        def _visit_for(self, stmt: ForStmt, ctx):
            found.update(_block_names(stmt.body))
            super()._visit_for(stmt, ctx)

    V()._visit_function(ast, None)
    return found


@fp.fpy(ctx=fp.REAL)
def two_loops(xs: list[fp.Real]) -> fp.Real:
    n = len(xs)
    a = 0.0
    for x in xs:
        p = n + 1
        a = a + p * x
    b = 0.0
    for x in xs:
        q = n + 2
        b = b + q * x
    return a + b


@fp.fpy(ctx=fp.REAL)
def nothing_to_hoist(xs: list[fp.Real]) -> fp.Real:
    acc = 0.0
    for x in xs:
        acc = acc + x * x
    return acc


_VALUES = ([], [1.0], [1.5, -2.0, 3.25])


def _agrees(func, out) -> bool:
    return all(repr(out(v)) == repr(func(v)) for v in _VALUES)


class TestTheWrapper:

    def test_it_returns_a_function_and_leaves_the_input_alone(self):
        out = hoist_invariant(two_loops)
        assert isinstance(out, Function)
        assert _body_names(two_loops.ast) == {'p', 'a', 'q', 'b'}

    def test_it_rejects_a_non_function(self):
        with pytest.raises(TypeError):
            hoist_invariant(two_loops.ast)

    def test_none_hoists_out_of_every_loop(self):
        out = hoist_invariant(two_loops)
        assert _body_names(out.ast) == {'a', 'b'}
        assert {'p', 'q'} <= _block_names(out.ast.body)
        assert _agrees(two_loops, out)


class TestAiming:

    def test_an_index_takes_one_loop(self):
        first = hoist_invariant(two_loops, 0)
        assert _body_names(first.ast) == {'a', 'q', 'b'}
        assert _agrees(two_loops, first)

        second = hoist_invariant(two_loops, 1)
        assert _body_names(second.ast) == {'p', 'a', 'b'}
        assert _agrees(two_loops, second)

    def test_a_cursor_takes_the_loop_it_names(self):
        where = sites(hoist_invariant, two_loops)
        assert len(where) == 2
        out = hoist_invariant(two_loops, where[1])
        assert _body_names(out.ast) == {'p', 'a', 'b'}
        assert _agrees(two_loops, out)

    def test_a_loop_with_nothing_to_hoist_is_no_site(self):
        assert sites(hoist_invariant, nothing_to_hoist) == []


class TestFailures:

    def test_an_index_past_the_end(self):
        with pytest.raises(TransformReferenceError, match='does not correspond'):
            hoist_invariant(two_loops, 2)

    def test_a_cursor_naming_a_refused_loop_says_why(self):
        (where, why), = refusals(hoist_invariant, nothing_to_hoist)
        assert '`x` varies across iterations' in why
        with pytest.raises(TransformDeclined, match='varies across iterations'):
            hoist_invariant(nothing_to_hoist, where)

    def test_a_cursor_from_another_program(self):
        where = sites(hoist_invariant, two_loops)[0]
        with pytest.raises(TransformReferenceError, match='unrelated program'):
            hoist_invariant(nothing_to_hoist, where)
