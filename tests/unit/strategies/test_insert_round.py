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


def _target_blocks(ast, ctx) -> int:
    """Number of blocks in *ast* written as a `ForeignVal` of *ctx*."""
    count = 0

    class _C(DefaultVisitor):
        def _visit_context(self, stmt: ContextStmt, c):
            nonlocal count
            if isinstance(stmt.ctx, ForeignVal) and stmt.ctx.val == ctx:
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
    def test_gives_each_verifying_operation_a_format(self):
        f = _pinned(_sum_of_squares, 2)
        assert _target_blocks(f.ast, fp.FP64) == 0
        out = insert_round(f, fp.FP64)
        # both multiplies verify; the exact sum of two 48-digit products does not
        assert _target_blocks(out.ast, fp.FP64) == 2
        assert _agree(out, f, 2)

    def test_does_not_mutate_the_input(self):
        f = _pinned(_sum_of_squares, 2)
        out = insert_round(f, fp.FP64)
        assert out is not f
        assert _target_blocks(f.ast, fp.FP64) == 0

    def test_is_idempotent(self):
        once = insert_round(_pinned(_sum_of_squares, 2), fp.FP64)
        twice = insert_round(once, fp.FP64)
        assert twice.ast.is_equiv(once.ast)

    def test_declines_a_format_too_narrow(self):
        f = _pinned(_sum_of_squares, 2)
        with pytest.raises(TransformDeclined, match='not representable'):
            insert_round(f, fp.FP16, where=1)

    def test_skips_a_narrow_format_without_a_where(self):
        f = _pinned(_sum_of_squares, 2)
        assert insert_round(f, fp.FP16).ast.is_equiv(f.ast)

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
    def test_lists_operations(self):
        f = _pinned(_sum_of_squares, 2)
        found = sites(insert_round, f)
        assert all(isinstance(c, ExprCursor) for c in found)
        assert [type(c.resolve()).__name__ for c in found] == ['Add', 'Mul', 'Mul']

    def test_an_index_aims_one_operation(self):
        f = _pinned(_sum_of_squares, 2)
        for i in (1, 2):
            out = insert_round(f, fp.FP64, where=i)
            assert _target_blocks(out.ast, fp.FP64) == 1
            assert _agree(out, f, 2)

    def test_a_cursor_aims_the_same_as_its_index(self):
        f = _pinned(_sum_of_squares, 2)
        for i, cursor in enumerate(sites(insert_round, f)):
            try:
                expect = insert_round(f, fp.FP64, where=i).format()
            except TransformDeclined:
                with pytest.raises(TransformDeclined):
                    insert_round(f, fp.FP64, where=cursor)
                continue
            assert insert_round(f, fp.FP64, where=cursor).format() == expect

    def test_a_where_naming_nothing(self):
        with pytest.raises(TransformReferenceError):
            insert_round(_pinned(_sum_of_squares, 2), fp.FP64, where=99)

    def test_a_cursor_of_an_unrelated_program(self):
        # `_prod3`'s body is under FP64, so it has no exact operation until
        # `elim_round` hoists one
        other = sites(insert_round, elim_round(_pinned(_prod3, 3)))[0]
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
            depths.append(_count_blocks(f.ast))
        assert depths == sorted(depths) and depths[0] < depths[-1]
        assert _agree(f, _pinned(_prod3, 3), 3)


def _count_blocks(ast) -> int:
    """Number of context blocks in *ast*, of any context."""
    count = 0

    class _C(DefaultVisitor):
        def _visit_context(self, stmt: ContextStmt, c):
            nonlocal count
            count += 1
            super()._visit_context(stmt, c)

    _C()._visit_function(ast, None)
    return count
