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
   inputs — including two soundness fixtures: user variables that
   collide with the minted names, and a body that mutates the
   iterated list in place.
"""

import pytest

import fpy2 as fp

from fpy2.ast import (
    Assign, ContextStmt, ForeignVal, ForStmt, Integer, ListSlice,
)
from fpy2.ast.visitor import DefaultVisitor
from fpy2.number import INTEGER
from fpy2.transform import SplitLoop, SplitLoopStrategy

_BOTH = (SplitLoopStrategy.STRICT, SplitLoopStrategy.PEEL)


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
def _mutates_late(xs: list[fp.Real]) -> fp.Real:
    # writes the *last* cell — for factor 2 and odd length, written in
    # the chunked prefix and read by the residual loop
    acc = 0.0
    for x in xs:
        acc = acc + x
        xs[2] = 99.0
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

    @pytest.mark.parametrize('strategy', _BOTH)
    def test_no_slice(self, strategy):
        out = _split(_total, strategy=strategy)
        assert not _has_node(out, ListSlice)

    @pytest.mark.parametrize('strategy', _BOTH)
    def test_integer_blocks(self, strategy):
        # one prelude (factor/len/check), one chunk bound; the PEEL
        # residual loop indexes with plain variables and needs none
        out = _split(_total, strategy=strategy)
        assert _count_integer_blocks(out) == 2

    def test_nested_loops(self):
        # STRICT: chunk loop + inner; PEEL adds the residual loop
        assert _count_fors(_split(_total, strategy=SplitLoopStrategy.STRICT)) == 2
        assert _count_fors(_split(_total, strategy=SplitLoopStrategy.PEEL)) == 3

    @pytest.mark.parametrize('strategy', _BOTH)
    def test_materialized_once(self, strategy):
        # user's `acc`, then the single materialize Assign, then the
        # INTEGER prelude
        out = _split(_total, strategy=strategy)
        kinds = [type(s) for s in out.body.stmts]
        assert kinds[:4] == [Assign, Assign, ContextStmt, ForStmt]

    def test_input_not_mutated(self):
        _split(_total)
        assert _count_fors(_total.ast) == 1


# ----------------------------------------------------------------------
# Semantic equivalence


@pytest.mark.parametrize('strategy', _BOTH)
class TestEquivalence:

    def test_divisible_lengths(self, strategy):
        out = _split(_total, strategy=strategy)
        for xs in ([], [1.0, 2.0], _XS4, [0.5] * 8):
            assert _total(xs) == _run(out, _total, xs)

    def test_factor_one_and_whole(self, strategy):
        for factor in (1, 4):
            out = _split(_total, factor, strategy=strategy)
            assert _total(_XS4) == _run(out, _total, _XS4)

    def test_name_collisions(self, strategy):
        out = _split(_collides, strategy=strategy)
        assert _collides(_XS4) == _run(out, _collides, _XS4)

    def test_body_mutates_iterable(self, strategy):
        out = _split(_mutates, strategy=strategy)
        assert _mutates(_XS4) == _run(out, _mutates, _XS4)

    def test_tuple_target(self, strategy):
        out = _split(_pairs, strategy=strategy)
        ps = [(1.0, 2.0), (3.0, 4.0)]
        assert _pairs(ps) == _run(out, _pairs, ps)


class TestRemainder:

    def test_strict_non_divisible_asserts(self):
        out = _split(_total, strategy=SplitLoopStrategy.STRICT)
        with pytest.raises(AssertionError):
            _run(out, _total, [1.0, 2.0, 3.0])

    def test_peel_any_length(self):
        out = _split(_total, strategy=SplitLoopStrategy.PEEL)
        for xs in ([], [1.0], [1.0, 2.0, 3.0], [0.5] * 7):
            assert _total(xs) == _run(out, _total, xs)

    def test_peel_factor_exceeds_length(self):
        out = _split(_total, 8, strategy=SplitLoopStrategy.PEEL)
        assert _total(_XS4) == _run(out, _total, _XS4)

    def test_peel_mutation_reaches_residual(self):
        # `xs[2]` is written in the chunked prefix; for length 3 and
        # factor 2 the residual loop reads exactly that cell
        out = _split(_mutates_late, strategy=SplitLoopStrategy.PEEL)
        for xs in ([1.0, 2.0, 3.0], _XS4):
            assert _mutates_late(list(xs)) == _run(out, _mutates_late, list(xs))

    def test_peel_tuple_target(self):
        # the residual loop rebuilds its own copy of the tuple binding
        out = _split(_pairs, strategy=SplitLoopStrategy.PEEL)
        ps = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]
        assert _pairs(ps) == _run(out, _pairs, ps)


# ----------------------------------------------------------------------
# Static-size discharge


@fp.fpy
def _range8() -> fp.Real:
    x = 0.0
    for i in range(8):
        x = x + i
    return x


@fp.fpy
def _range7() -> fp.Real:
    x = 0.0
    for i in range(7):
        x = x + i
    return x


@fp.fpy
def _range0() -> fp.Real:
    x = 0.0
    for i in range(0):
        x = x + i
    return x


class TestStaticSize:
    """A statically-known length plus a literal factor discharges the
    remainder handling at compile time: no ``len``, no ``fmod``, and
    empty regions are dropped."""

    def test_peel_divisible_no_residual(self):
        out = _split(_range8, 2, strategy=SplitLoopStrategy.PEEL)
        txt = out.format()
        assert 'len(' not in txt and 'fmod(' not in txt
        assert txt.count('for ') == 2   # chunk + inner; no residual
        assert _range8() == _run(out, _range8)

    def test_peel_non_divisible_constant_residual(self):
        out = _split(_range7, 2, strategy=SplitLoopStrategy.PEEL)
        txt = out.format()
        assert 'len(' not in txt and 'fmod(' not in txt
        assert txt.count('for ') == 3   # chunk + inner + residual
        assert _range7() == _run(out, _range7)

    def test_peel_factor_exceeds_length(self):
        # m = 0: no chunked region at all, only the residual loop
        out = _split(_range7, 8, strategy=SplitLoopStrategy.PEEL)
        txt = out.format()
        assert 'len(' not in txt and 'fmod(' not in txt
        assert txt.count('for ') == 1
        assert _range7() == _run(out, _range7)

    def test_peel_empty(self):
        out = _split(_range0, 2, strategy=SplitLoopStrategy.PEEL)
        assert _count_fors(out) == 0
        assert _range0() == _run(out, _range0)

    def test_strict_divisible_no_runtime_check(self):
        out = _split(_range8, 2, strategy=SplitLoopStrategy.STRICT)
        txt = out.format()
        assert 'len(' not in txt and 'fmod(' not in txt and 'assert' not in txt
        assert _range8() == _run(out, _range8)

    def test_strict_indivisible_raises(self):
        with pytest.raises(ValueError):
            _split(_range7, 2, strategy=SplitLoopStrategy.STRICT)

    def test_unknown_length_keeps_runtime_check(self):
        # a list parameter has no statically-known length
        out = _split(_total, 2, strategy=SplitLoopStrategy.PEEL)
        txt = out.format()
        assert 'len(' in txt and 'fmod(' in txt


# ----------------------------------------------------------------------
# `where` targeting and validation


class TestWhere:

    def test_where_selects_loop(self):
        # STRICT replaces the selected loop with 2 loops.  PEEL replaces
        # it with 3 and duplicates its body into the residual loop, so
        # splitting the *outer* loop of the nest re-emits the inner one
        # (3 + 2 copies of the body's loop = 5).
        xss = [[1.0, 2.0], [3.0, 4.0]]
        for w, expect_peel in ((0, 5), (1, 4)):
            out = _split(_nested, 2, where=w, strategy=SplitLoopStrategy.STRICT)
            assert _count_fors(out) == 3
            assert _nested(xss) == _run(out, _nested, xss)

            out = _split(_nested, 2, where=w, strategy=SplitLoopStrategy.PEEL)
            assert _count_fors(out) == expect_peel
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


# ----------------------------------------------------------------------
# Factor validation


@fp.fpy
def _total_k(xs: list[fp.Real], k: fp.Real) -> fp.Real:
    # `k` is referenced only by the factor `Var` the transform injects
    acc = 0.0
    for x in xs:
        acc = acc + x
    return acc


class TestFactorValidation:

    def test_non_positive_literal_raises(self):
        for bad in (0, -2):
            with pytest.raises(ValueError):
                _split(_total, bad)

    @pytest.mark.parametrize('strategy', _BOTH)
    def test_non_positive_runtime_factor_asserts(self, strategy):
        # a runtime factor < 1 would silently skip iterations
        # (`range(0, n, f)` is empty) — the emitted assert rejects it
        from fpy2.ast import NamedId, Var
        out = SplitLoop.apply(
            _total_k.ast, Var(NamedId('k'), None), strategy=strategy
        )
        assert _total_k(_XS4, 2.0) == _run(out, _total_k, _XS4, 2.0)
        for bad in (0.0, -2.0):
            with pytest.raises(AssertionError):
                _run(out, _total_k, _XS4, bad)
