"""
Unit tests for the :class:`fpy2.transform.RoundInsert` transform.

The rewrite only swaps a block's context expression, so the shape is easy to
read off: these tests assert

1. **Structural shape**: the ``fp.REAL`` block's context becomes the target.
2. **Declines**: each refusal path, by the reason it gives.
3. **Semantic equivalence** via the interpreter, since an inserted rounding
   that is proven an identity must change no result.
4. **Site listing and aiming**, including that listing is syntactic — a block
   the rewrite will refuse is still counted by a ``where`` index.
"""

import pytest

import fpy2 as fp
from fpy2.analysis import PartialEval
from fpy2.ast.fpyast import ContextStmt, FuncDef
from fpy2.ast.visitor import DefaultVisitor
from fpy2.function import Function
from fpy2.number import REAL
from fpy2.transform import (
    Monomorphize,
    RoundElim,
    RoundInsert,
    TransformDeclined,
    TransformReferenceError,
)
from fpy2.types import RealType

# ----------------------------------------------------------------------
# Helpers


def _ctxs(ast: FuncDef) -> list[object]:
    """The context of every block, outermost first.

    Resolved rather than read off the node: a source-written ``with fp.REAL:``
    is an ``Attribute`` while the one :class:`RoundElim` emits is a
    ``ForeignVal``, and both must compare equal here.
    """
    eval_info = PartialEval.apply(ast)
    found: list[object] = []

    class _C(DefaultVisitor):
        def _visit_context(self, stmt: ContextStmt, ctx):
            found.append(eval_info.by_expr.get(stmt.ctx))
            super()._visit_context(stmt, ctx)

    _C()._visit_function(ast, None)
    return found


@fp.fpy(ctx=fp.FP64)
def _prod3(x: fp.Real, y: fp.Real, z: fp.Real) -> fp.Real:
    return x * y * z


def _fp32_args(func, n: int) -> FuncDef:
    """*func* with its `n` arguments pinned to FP32 under an FP64 context."""
    return Monomorphize.apply(func.ast, fp.FP64, [RealType(fp.FP32)] * n)


def _hoisted() -> FuncDef:
    """``_prod3`` with FP32 arguments and its exact multiply hoisted to REAL."""
    return RoundElim.apply(_fp32_args(_prod3, 3))


_SAMPLE = [(1.5, 2.5, 3.5), (0.1, -0.25, 4.0), (-3.0, 0.0, 1.0)]


# ----------------------------------------------------------------------
# The rewrite


class TestRoundInsert:
    def test_gives_the_exact_block_a_format(self):
        ast = _hoisted()
        assert _ctxs(ast) == [REAL]
        out = RoundInsert.apply(ast, fp.FP64)
        assert _ctxs(out) == [fp.FP64]

    def test_preserves_the_body(self):
        out = RoundInsert.apply(_hoisted(), fp.FP64)
        block = out.body.stmts[0]
        assert isinstance(block, ContextStmt)
        assert len(block.body.stmts) == 1

    def test_does_not_mutate_the_input(self):
        ast = _hoisted()
        RoundInsert.apply(ast, fp.FP64)
        assert _ctxs(ast) == [REAL]

    def test_is_idempotent(self):
        once = RoundInsert.apply(_hoisted(), fp.FP64)
        # the block is no longer REAL, so the second pass declines it
        assert RoundInsert.apply(once, fp.FP64).is_equiv(once)

    def test_agrees_with_the_hoisted_program(self):
        ast = _hoisted()
        before = Function(ast, runtime=_prod3.runtime)
        after = Function(RoundInsert.apply(ast, fp.FP64), runtime=_prod3.runtime)
        for xyz in _SAMPLE:
            assert before(*xyz) == after(*xyz)


# ----------------------------------------------------------------------
# Declines


class TestDeclines:
    @pytest.mark.parametrize('ctx', [fp.FP32, fp.FP16])
    def test_a_format_too_narrow(self, ctx):
        with pytest.raises(TransformDeclined, match='not representable'):
            RoundInsert.apply(_hoisted(), ctx, where=0)

    def test_a_narrow_format_is_skipped_without_a_where(self):
        ast = _hoisted()
        assert RoundInsert.apply(ast, fp.FP32).is_equiv(ast)

    def test_a_block_that_already_rounds(self):
        @fp.fpy(ctx=fp.FP64)
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            with fp.FP32:
                a = x * y
            return a

        with pytest.raises(TransformDeclined, match='no rounding to insert'):
            RoundInsert.apply(f.ast, fp.FP64, where=0)

    def test_a_block_of_several_assignments(self):
        @fp.fpy(ctx=fp.FP64)
        def f(x: fp.Real, y: fp.Real, z: fp.Real) -> fp.Real:
            with fp.REAL:
                b = x * y
                c = b * z
            return c

        with pytest.raises(TransformDeclined, match='more than one assignment'):
            RoundInsert.apply(f.ast, fp.FP64, where=0)

    def test_a_stochastic_target(self):
        stochastic = fp.IEEEContext(8, 32, num_randbits=4)
        with pytest.raises(TransformDeclined, match='stochastically'):
            RoundInsert.apply(_hoisted(), stochastic, where=0)

    def test_an_unbounded_operand(self):
        # no argument formats, so inference cannot bound the product
        @fp.fpy(ctx=fp.FP64)
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            with fp.REAL:
                t = x * y
            return t

        with pytest.raises(TransformDeclined):
            RoundInsert.apply(f.ast, fp.FP64, where=0)

    def test_rejects_a_non_context(self):
        with pytest.raises(TypeError):
            RoundInsert.apply(_hoisted(), fp.FP64.format())  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# Sites and aiming


class TestSites:
    def test_lists_syntactically(self):
        """A block the rewrite refuses is still a site."""

        @fp.fpy(ctx=fp.FP64)
        def f(x: fp.Real, y: fp.Real, z: fp.Real) -> fp.Real:
            with fp.FP32:
                a = x * y
            with fp.REAL:
                b = x * y
                c = b * z
            return a + c

        assert len(RoundInsert.sites(f.ast)) == 2

    def test_a_where_index_aims_one_block(self):
        @fp.fpy(ctx=fp.FP64)
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            with fp.REAL:
                a = x * y
            with fp.REAL:
                b = x * y
            return a + b

        ast = _fp32_args(f, 2)
        assert _ctxs(RoundInsert.apply(ast, fp.FP64, where=0)) == [fp.FP64, REAL]
        assert _ctxs(RoundInsert.apply(ast, fp.FP64, where=1)) == [REAL, fp.FP64]
        assert _ctxs(RoundInsert.apply(ast, fp.FP64)) == [fp.FP64, fp.FP64]

    def test_a_cursor_aims_the_same_as_its_index(self):
        ast = _hoisted()
        cursor = RoundInsert.sites(ast)[0]
        by_index = RoundInsert.apply(ast, fp.FP64, where=0)
        by_cursor = RoundInsert.apply(ast, fp.FP64, where=cursor)
        assert by_cursor.is_equiv(by_index)

    def test_a_where_out_of_range(self):
        with pytest.raises(TransformReferenceError, match='candidate site'):
            RoundInsert.apply(_hoisted(), fp.FP64, where=7)

    def test_a_where_of_the_wrong_type(self):
        with pytest.raises(TypeError):
            RoundInsert.apply(_hoisted(), fp.FP64, where='0')  # type: ignore[arg-type]
