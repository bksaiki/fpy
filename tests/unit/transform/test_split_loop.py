"""
Unit tests for the :class:`fpy2.transform.SplitLoop` transform.

The rewrite mints fresh names via a seeded ``Gensym``, whose suffixes
depend on the names already in scope — so ``is_equiv`` against a
hand-written golden AST is too brittle.  As in ``test_for_unroll``,
the tests assert:

1. **Structural shape** of the rewritten AST: the iterable is
   materialized once, all synthesized arithmetic sits under
   ``with fp.INTEGER:``, no ``ListSlice`` is emitted, and the loops
   nest as expected.
2. **Semantic equivalence** via the FPy interpreter on concrete
   inputs — including the two soundness fixtures from the audit:
   user variables that collide with the minted names, and a body
   that mutates the iterated list in place.
"""

import pytest

import fpy2 as fp

from fpy2.ast import (
    Assign, ContextStmt, ForeignVal, ForStmt, Integer, ListSlice,
)
from fpy2.ast.visitor import DefaultVisitor
from fpy2.number import INTEGER
from fpy2.transform import SplitLoop


# ----------------------------------------------------------------------
# Helpers


def _split(f: fp.Function, factor: int = 2, **kwargs) -> fp.ast.FuncDef:
    return SplitLoop.apply(f.ast, Integer(factor, None), **kwargs)


def _run(ast: fp.ast.FuncDef, fn: fp.Function, *args):
    return fn.with_ast(ast)(*args)


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


def _count_integer_blocks(ast) -> int:
    """Number of ``with fp.INTEGER:`` blocks in *ast*."""
    count = 0

    class _C(DefaultVisitor):
        def _visit_context(self, stmt: ContextStmt, ctx):
            nonlocal count
            if isinstance(stmt.ctx, ForeignVal) and stmt.ctx.val is INTEGER:
                count += 1
            super()._visit_context(stmt, ctx)

    _C()._visit_function(ast, None)
    return count


def _count_fors(ast) -> int:
    count = 0

    class _C(DefaultVisitor):
        def _visit_for(self, stmt, ctx):
            nonlocal count
            count += 1
            super()._visit_for(stmt, ctx)

    _C()._visit_function(ast, None)
    return count


# ----------------------------------------------------------------------
# Fixtures


@fp.fpy
def _total(xs: list[fp.Real]) -> fp.Real:
    acc = 0.0
    for x in xs:
        acc = acc + x
    return acc


@fp.fpy
def _collides(xs: list[fp.Real]) -> fp.Real:
    # user variables that collide with the transform's minted names
    t = 100.0
    i = 10.0
    j = 1.0
    acc = 0.0
    for x in xs:
        acc = acc + x * t + i + j
    return acc


@fp.fpy
def _mutates(xs: list[fp.Real]) -> fp.Real:
    # the body writes the iterated list: later reads must observe it
    acc = 0.0
    for x in xs:
        acc = acc + x
        xs[1] = 99.0
    return acc


@fp.fpy
def _pairs(ps: list[tuple[fp.Real, fp.Real]]) -> fp.Real:
    acc = 0.0
    for a, b in ps:
        acc = acc + a * b
    return acc


@fp.fpy
def _nested(xss: list[list[fp.Real]]) -> fp.Real:
    acc = 0.0
    for xs in xss:
        for x in xs:
            acc = acc + x
    return acc


_XS4 = [1.0, 2.0, 3.0, 4.0]


# ----------------------------------------------------------------------
# Structural shape


class TestShape:

    def test_no_slice(self):
        out = _split(_total)
        assert not _has_node(out, ListSlice)

    def test_integer_blocks(self):
        # one prelude (factor/len/assert), one chunk bound
        out = _split(_total)
        assert _count_integer_blocks(out) == 2

    def test_nested_loops(self):
        out = _split(_total)
        assert _count_fors(out) == 2

    def test_materialized_once(self):
        # exactly one bare Assign precedes the INTEGER prelude
        out = _split(_total)
        kinds = [type(s) for s in out.body.stmts]
        assert kinds[:4] == [Assign, Assign, ContextStmt, ForStmt]

    def test_input_not_mutated(self):
        _split(_total)
        assert _count_fors(_total.ast) == 1


# ----------------------------------------------------------------------
# Semantic equivalence


class TestEquivalence:

    def test_divisible_lengths(self):
        out = _split(_total)
        for xs in ([], [1.0, 2.0], _XS4, [0.5] * 8):
            assert _total(xs) == _run(out, _total, xs)

    def test_factor_one_and_whole(self):
        for factor in (1, 4):
            out = _split(_total, factor)
            assert _total(_XS4) == _run(out, _total, _XS4)

    def test_strict_non_divisible_asserts(self):
        out = _split(_total)
        with pytest.raises(AssertionError):
            _run(out, _total, [1.0, 2.0, 3.0])

    def test_name_collisions(self):
        out = _split(_collides)
        assert _collides(_XS4) == _run(out, _collides, _XS4)

    def test_body_mutates_iterable(self):
        out = _split(_mutates)
        assert _mutates(_XS4) == _run(out, _mutates, _XS4)

    def test_tuple_target(self):
        out = _split(_pairs)
        ps = [(1.0, 2.0), (3.0, 4.0)]
        assert _pairs(ps) == _run(out, _pairs, ps)


# ----------------------------------------------------------------------
# `where` targeting and validation


class TestWhere:

    def test_where_selects_loop(self):
        xss = [[1.0, 2.0], [3.0, 4.0]]
        for w in (0, 1):
            out = _split(_nested, 2, where=w)
            # exactly one loop split: original 2 loops become 3
            assert _count_fors(out) == 3
            assert _nested(xss) == _run(out, _nested, xss)

    def test_where_out_of_range(self):
        with pytest.raises(ValueError):
            _split(_total, 2, where=1)
        with pytest.raises(ValueError):
            _split(_total, 2, where=-1)

    def test_type_errors(self):
        with pytest.raises(TypeError):
            SplitLoop.apply(_total, Integer(2, None))  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            SplitLoop.apply(_total.ast, 2)  # type: ignore[arg-type]  # int, not Expr
        with pytest.raises(TypeError):
            _split(_total, 2, where='x')  # type: ignore[arg-type]
