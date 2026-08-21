"""Unit tests for :func:`fpy2.strategies.insert_round`.

The transform itself is tested exhaustively in
``tests/unit/transform/test_round_insert.py``; these tests pin the wrapper's
behavior, its aiming, and its composition with ``elim_round``.
"""

import itertools

import pytest

import fpy2 as fp
from fpy2.ast import ContextStmt, ForeignVal
from fpy2.ast.visitor import DefaultVisitor
from fpy2.strategies import (
    ExprCursor,
    TransformDeclined,
    TransformReferenceError,
    elim_round,
    insert_round,
    monomorphize,
    simplify,
    sites,
)
from fpy2.types import RealType


def _blocks(ast, ctx=None) -> int:
    """Context blocks in *ast* -- all of them, or just those written as a
    `ForeignVal` of *ctx*."""
    count = 0

    class _C(DefaultVisitor):
        def _visit_context(self, stmt: ContextStmt, c):
            nonlocal count
            if ctx is None or (
                isinstance(stmt.ctx, ForeignVal) and stmt.ctx.val == ctx
            ):
                count += 1
            super()._visit_context(stmt, c)

    _C()._visit_function(ast, None)
    return count


@fp.fpy(ctx=fp.FP64)
def _sum_of_squares(x: fp.Real, y: fp.Real) -> fp.Real:
    with fp.REAL:
        t = (x * x) + (y * y)
    return t


@fp.fpy(ctx=fp.FP64)
def _prod3(x: fp.Real, y: fp.Real, z: fp.Real) -> fp.Real:
    return x * y * z


def _pinned(func, n: int):
    """*func* with its `n` arguments pinned to FP32 under an FP64 context."""
    return monomorphize(func, fp.FP64, [RealType(fp.FP32)] * n)


# exactly FP32-representable, so the argument pin is honoured
_SAMPLE = [1.5, -3.0, 0.5, 2.25, 7.5, 0.25]


def _agree(a, b, arity: int) -> bool:
    return all(
        a(*args) == b(*args)
        for args in itertools.product(_SAMPLE, repeat=arity)
    )


class TestInsertRound:
    def test_returns_a_function_that_agrees(self):
        f = _pinned(_sum_of_squares, 2)
        out = insert_round(f, fp.FP64)
        assert out is not f
        assert _blocks(out.ast, fp.FP64) == 2
        assert _agree(out, f, 2)

    def test_composes_with_simplify(self):
        f = _pinned(_sum_of_squares, 2)
        out = simplify(insert_round(f, fp.FP64))
        assert _agree(out, f, 2)

    def test_rejects_non_function(self):
        with pytest.raises(TypeError):
            insert_round(_pinned(_sum_of_squares, 2).ast, fp.FP64)  # type: ignore[arg-type]

    def test_rejects_non_context(self):
        f = _pinned(_sum_of_squares, 2)
        with pytest.raises(TypeError):
            insert_round(f, fp.FP64.format())  # type: ignore[arg-type]


class TestWhere:
    def test_an_index_aims_one_operation(self):
        f = _pinned(_sum_of_squares, 2)
        for i in (0, 1):
            out = insert_round(f, fp.FP64, where=i)
            assert _blocks(out.ast, fp.FP64) == 1
            assert _agree(out, f, 2)

    def test_a_where_naming_nothing(self):
        with pytest.raises(TransformReferenceError):
            insert_round(_pinned(_sum_of_squares, 2), fp.FP64, where=99)

    def test_a_cursor_of_an_unrelated_program(self):
        # `_prod3`'s body is under FP64, so it has no exact operation until
        # `elim_round` hoists one
        other = sites(insert_round, elim_round(_pinned(_prod3, 3)), ctx=fp.FP64)[0]
        with pytest.raises(TransformReferenceError):
            insert_round(_pinned(_sum_of_squares, 2), fp.FP64, where=other)


class TestRoundTrip:
    """The headline property: the two operators are inverses on what
    ``elim_round`` emits."""

    def test_elim_then_insert_recovers_the_program(self):
        pinned = _pinned(_prod3, 3)
        out = insert_round(elim_round(pinned), fp.FP64)
        assert _agree(out, pinned, 3)

    def test_an_unbounded_scope_leaves_nothing_to_insert(self):
        """The other side of the asymmetry: under an unbounded scope
        ``elim_round``'s strictly-tighter guard declines to hoist, so this
        operator finds nothing it can prove."""

        @fp.fpy(ctx=fp.INTEGER)
        def add2(x: fp.Real, y: fp.Real) -> fp.Real:
            return x + y

        pinned = monomorphize(add2, fp.INTEGER, [RealType(fp.INTEGER)] * 2)
        hoisted = elim_round(pinned)
        assert hoisted.ast.is_equiv(pinned.ast)
        assert insert_round(hoisted, fp.INTEGER).ast.is_equiv(pinned.ast)

    def test_alternating_the_two_does_not_converge(self):
        """Neither operator's guard bounds the composition: each round trip
        wraps the operation in one more block, because both hoist into a
        *nested* block rather than replacing the scope they found.  A recipe
        that alternates them needs explicit fuel."""
        f = _pinned(_prod3, 3)
        depths = []
        for i in range(4):
            f = elim_round(f) if i % 2 == 0 else insert_round(f, fp.FP64)
            depths.append(_blocks(f.ast))
        assert depths == sorted(depths) and depths[0] < depths[-1]
        assert _agree(f, _pinned(_prod3, 3), 3)


class TestCursorForwarding:
    def test_a_cursor_survives_an_earlier_insertion(self):
        """The per-site workflow: list once, then round them one at a time.
        This needs the pass to claim `exprs_preserved`, which it did not at
        first, so every expression cursor failed to forward."""

        @fp.fpy(ctx=fp.FP64)
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            with fp.REAL:
                t = x * x
                s = y * y
            return t + s

        f0 = monomorphize(f, fp.FP64, [RealType(fp.FP32)] * 2)
        cursors = sites(insert_round, f0, ctx=fp.FP64)
        assert len(cursors) == 2

        f1 = insert_round(f0, fp.FP64, where=cursors[0])
        # a cursor of `f0`, forwarded across the rewrite that produced `f1`
        f2 = insert_round(f1, fp.FP64, where=cursors[1])
        assert _blocks(f2.ast, fp.FP64) == 2
        assert _agree(f2, f0, 2)
