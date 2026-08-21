"""Unit tests for :func:`fpy2.strategies.insert_round`.

The transform itself is tested exhaustively in
``tests/unit/transform/test_round_insert.py``; these tests pin the wrapper's
behavior, its aiming, and its composition with ``elim_round``.
"""

import pytest

import fpy2 as fp
from fpy2.analysis import PartialEval
from fpy2.ast import ContextStmt
from fpy2.ast.visitor import DefaultVisitor
from fpy2.number import REAL
from fpy2.strategies import (
    TransformDeclined,
    TransformReferenceError,
    elim_round,
    insert_round,
    monomorphize,
    simplify,
    sites,
)
from fpy2.types import RealType


def _count_real_blocks(ast) -> int:
    """Number of blocks in *ast* that round exactly, however written."""
    eval_info = PartialEval.apply(ast)
    count = 0

    class _C(DefaultVisitor):
        def _visit_context(self, stmt: ContextStmt, ctx):
            nonlocal count
            if eval_info.by_expr.get(stmt.ctx) is REAL:
                count += 1
            super()._visit_context(stmt, ctx)

    _C()._visit_function(ast, None)
    return count


def _count_blocks(ast) -> int:
    """Number of context blocks in *ast*, of any context."""
    count = 0

    class _C(DefaultVisitor):
        def _visit_context(self, stmt: ContextStmt, ctx):
            nonlocal count
            count += 1
            super()._visit_context(stmt, ctx)

    _C()._visit_function(ast, None)
    return count


@fp.fpy(ctx=fp.FP64)
def _prod3(x: fp.Real, y: fp.Real, z: fp.Real) -> fp.Real:
    return x * y * z


def _hoisted():
    """``_prod3`` with FP32 arguments and its exact multiply hoisted to REAL."""
    pinned = monomorphize(_prod3, fp.FP64, [RealType(fp.FP32)] * 3)
    return elim_round(pinned)


_SAMPLE = [(1.5, 2.5, 3.5), (0.1, -0.25, 4.0), (-3.0, 0.0, 1.0)]


class TestInsertRound:
    def test_gives_the_exact_block_a_format(self):
        hoisted = _hoisted()
        assert _count_real_blocks(hoisted.ast) == 1
        out = insert_round(hoisted, fp.FP64)
        assert _count_real_blocks(out.ast) == 0

    def test_agrees_with_the_hoisted_program(self):
        hoisted = _hoisted()
        out = insert_round(hoisted, fp.FP64)
        for xyz in _SAMPLE:
            assert out(*xyz) == hoisted(*xyz)

    def test_does_not_mutate_the_input(self):
        hoisted = _hoisted()
        out = insert_round(hoisted, fp.FP64)
        assert out is not hoisted
        assert _count_real_blocks(hoisted.ast) == 1

    def test_is_idempotent(self):
        once = insert_round(_hoisted(), fp.FP64)
        twice = insert_round(once, fp.FP64)
        assert twice.ast.is_equiv(once.ast)

    def test_declines_a_format_too_narrow(self):
        with pytest.raises(TransformDeclined, match='not representable'):
            insert_round(_hoisted(), fp.FP32, where=0)

    def test_skips_a_narrow_format_without_a_where(self):
        hoisted = _hoisted()
        assert insert_round(hoisted, fp.FP32).ast.is_equiv(hoisted.ast)

    def test_composes_with_simplify(self):
        hoisted = _hoisted()
        out = simplify(insert_round(hoisted, fp.FP64))
        for xyz in _SAMPLE:
            assert out(*xyz) == hoisted(*xyz)

    def test_rejects_non_function(self):
        with pytest.raises(TypeError):
            insert_round(_hoisted().ast, fp.FP64)  # type: ignore[arg-type]

    def test_rejects_non_context(self):
        with pytest.raises(TypeError):
            insert_round(_hoisted(), fp.FP64.format())  # type: ignore[arg-type]


class TestWhere:
    @staticmethod
    def _two_blocks():
        @fp.fpy(ctx=fp.FP64)
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            with fp.REAL:
                a = x * y
            with fp.REAL:
                b = x * y
            return a + b

        return monomorphize(f, fp.FP64, [RealType(fp.FP32)] * 2)

    def test_an_index_aims_one_block(self):
        f = self._two_blocks()
        assert _count_real_blocks(insert_round(f, fp.FP64, where=0).ast) == 1
        assert _count_real_blocks(insert_round(f, fp.FP64, where=1).ast) == 1
        assert _count_real_blocks(insert_round(f, fp.FP64).ast) == 0

    def test_a_cursor_aims_the_same_as_its_index(self):
        f = self._two_blocks()
        for i, cursor in enumerate(sites(insert_round, f)):
            expect = insert_round(f, fp.FP64, where=i).format()
            assert insert_round(f, fp.FP64, where=cursor).format() == expect

    def test_a_where_naming_nothing(self):
        with pytest.raises(TransformReferenceError):
            insert_round(_hoisted(), fp.FP64, where=7)

    def test_a_cursor_of_an_unrelated_program(self):
        other = sites(insert_round, self._two_blocks())[0]
        with pytest.raises(TransformReferenceError, match='unrelated program'):
            insert_round(_hoisted(), fp.FP64, where=other)


class TestRoundTrip:
    """The headline property: the two operators are inverses on what
    ``elim_round`` emits."""

    def test_elim_then_insert_recovers_the_program(self):
        pinned = monomorphize(_prod3, fp.FP64, [RealType(fp.FP32)] * 3)
        out = insert_round(elim_round(pinned), fp.FP64)
        assert _count_real_blocks(out.ast) == 0
        for xyz in _SAMPLE:
            assert out(*xyz) == pinned(*xyz)

    def test_an_unbounded_scope_leaves_nothing_to_insert(self):
        """The other side of the asymmetry: under an unbounded scope
        ``elim_round``'s strictly-tighter guard declines to hoist, so this
        operator has no site.  The pair is a no-op rather than an inverse."""

        @fp.fpy(ctx=fp.INTEGER)
        def add2(x: fp.Real, y: fp.Real) -> fp.Real:
            return x + y

        pinned = monomorphize(add2, fp.INTEGER, [RealType(fp.INTEGER)] * 2)
        hoisted = elim_round(pinned)
        assert hoisted.ast.is_equiv(pinned.ast)
        assert sites(insert_round, hoisted) == []

    def test_alternating_the_two_does_not_converge(self):
        """Neither operator's guard bounds the composition: each round trip
        wraps the operation in one more block, because ``elim_round`` hoists
        into a *nested* REAL block rather than replacing the scope it found.
        A recipe that alternates them needs explicit fuel."""
        f = monomorphize(_prod3, fp.FP64, [RealType(fp.FP32)] * 3)
        depths = []
        for i in range(4):
            f = elim_round(f) if i % 2 == 0 else insert_round(f, fp.FP64)
            depths.append(_count_blocks(f.ast))
        assert depths == sorted(depths) and depths[0] < depths[-1]
        # still correct at every step, just larger
        for xyz in _SAMPLE:
            assert f(*xyz) == _prod3(*xyz)
