"""
Unit tests for the :class:`fpy2.transform.RoundInsert` transform.

The candidates are *operations*, and the rewrite mints fresh ``_tN`` names via
``Gensym``, so comparing against a hand-written golden AST is brittle.  These
tests assert

1. **Structural shape**: the aimed operation lands alone in a block of the
   target format, and the rest of its statement stays exact.
2. **Independence**: operations may be given formats one at a time, including
   one whose result a later exact operation reads — the inserted rounding is an
   identity, so it changes no value.
3. **Declines**: each refusal path, by the reason it gives.
4. **Semantic equivalence** via the interpreter, on inputs that honour the
   argument formats: an inserted rounding proven an identity must change no
   result.
"""

import itertools

import pytest

import fpy2 as fp
from fpy2.ast.fpyast import ContextStmt, ForeignVal, FuncDef
from fpy2.ast.visitor import DefaultVisitor
from fpy2.function import Function
from fpy2.transform import (
    ExprCursor,
    Monomorphize,
    RoundElim,
    RoundInsert,
    TransformDeclined,
    TransformReferenceError,
)
from fpy2.types import RealType

# ----------------------------------------------------------------------
# Helpers


def _target_blocks(ast: FuncDef, ctx) -> int:
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


def _fp32_args(func, n: int) -> FuncDef:
    """*func* with its `n` arguments pinned to FP32 under an FP64 context."""
    return Monomorphize.apply(func.ast, fp.FP64, [RealType(fp.FP32)] * n)


# exactly FP32-representable, so the argument pin is honoured; feeding a value
# the pin does not cover is out of contract, and so is the rewrite's reasoning
_SAMPLE = [1.5, -3.0, 0.5, 2.25, 7.5, 0.25, 1024.0, 0.0]


def _agree(a: FuncDef, b: FuncDef, runtime, arity: int) -> bool:
    fa: Function = Function(a, runtime=runtime)
    fb: Function = Function(b, runtime=runtime)
    return all(
        fa(*args) == fb(*args)
        for args in itertools.product(_SAMPLE, repeat=arity)
    )


@fp.fpy(ctx=fp.FP64)
def _sum_of_squares(x: fp.Real, y: fp.Real) -> fp.Real:
    with fp.REAL:
        t = (x * x) + (y * y)
    return t


@fp.fpy(ctx=fp.FP64)
def _independent(x: fp.Real, y: fp.Real) -> fp.Real:
    with fp.REAL:
        t = x * x
        s = y * y
    return t + s


@fp.fpy(ctx=fp.FP64)
def _dependent(x: fp.Real, y: fp.Real) -> fp.Real:
    with fp.REAL:
        t = x * x
        s = t * y
    return s


# ----------------------------------------------------------------------
# Sites are operations


class TestSites:
    def test_lists_operations_not_blocks(self):
        ast = _fp32_args(_sum_of_squares, 2)
        found = RoundInsert.sites(ast)
        assert all(isinstance(c, ExprCursor) for c in found)
        # the add, then each multiply: outermost first
        assert [type(c.resolve()).__name__ for c in found] == ['Add', 'Mul', 'Mul']

    def test_an_operation_that_already_rounds_is_not_a_site(self):
        """The block-sited version listed both `fp.round` blocks and declined
        every one.  Only the add, which is under the function's own exact
        scope, is a candidate now."""

        @fp.fpy(ctx=fp.REAL)
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            with fp.FP16:
                p = fp.round(x)
            with fp.FP16:
                q = fp.round(y)
            return p + q

        found = RoundInsert.sites(f.ast)
        assert [type(c.resolve()).__name__ for c in found] == ['Add']

    def test_a_cursor_aims_the_same_as_its_index(self):
        ast = _fp32_args(_sum_of_squares, 2)
        for i, cursor in enumerate(RoundInsert.sites(ast)):
            try:
                expect = RoundInsert.apply(ast, fp.FP64, where=i)
            except TransformDeclined:
                with pytest.raises(TransformDeclined):
                    RoundInsert.apply(ast, fp.FP64, where=cursor)
                continue
            assert RoundInsert.apply(ast, fp.FP64, where=cursor).is_equiv(expect)


# ----------------------------------------------------------------------
# The rewrite


class TestRoundInsert:
    def test_aims_one_operation_inside_a_statement(self):
        ast = _fp32_args(_sum_of_squares, 2)
        out = RoundInsert.apply(ast, fp.FP64, where=1)   # the first multiply
        assert _target_blocks(out, fp.FP64) == 1
        assert _agree(ast, out, _sum_of_squares.runtime, 2)

    def test_rewrites_every_verifying_operation(self):
        ast = _fp32_args(_sum_of_squares, 2)
        out = RoundInsert.apply(ast, fp.FP64)
        # both multiplies fit FP64; the exact sum of two 48-digit products
        # does not, so the add is left alone
        assert _target_blocks(out, fp.FP64) == 2
        assert _agree(ast, out, _sum_of_squares.runtime, 2)

    def test_independent_operations_are_aimable_one_at_a_time(self):
        ast = _fp32_args(_independent, 2)
        assert len(RoundInsert.sites(ast)) == 2
        for i in (0, 1):
            out = RoundInsert.apply(ast, fp.FP64, where=i)
            assert _target_blocks(out, fp.FP64) == 1
            assert _agree(ast, out, _independent.runtime, 2)

    def test_rounding_a_dependency_preserves_the_reader(self):
        """The property that makes per-operation aiming safe: the insertion is
        an identity, so an operation reading the result sees what it would
        have seen."""
        ast = _fp32_args(_dependent, 2)
        out = RoundInsert.apply(ast, fp.FP64, where=0)   # `t = x * x`
        assert _target_blocks(out, fp.FP64) == 1
        assert _agree(ast, out, _dependent.runtime, 2)

    def test_does_not_mutate_the_input(self):
        ast = _fp32_args(_sum_of_squares, 2)
        RoundInsert.apply(ast, fp.FP64)
        assert _target_blocks(ast, fp.FP64) == 0

    def test_is_idempotent(self):
        once = RoundInsert.apply(_fp32_args(_sum_of_squares, 2), fp.FP64)
        assert RoundInsert.apply(once, fp.FP64).is_equiv(once)

    def test_inverts_round_elim(self):
        pinned = _fp32_args(_sum_of_squares, 2)
        out = RoundInsert.apply(RoundElim.apply(pinned), fp.FP64)
        assert _agree(pinned, out, _sum_of_squares.runtime, 2)


# ----------------------------------------------------------------------
# Declines


class TestDeclines:
    def test_an_operation_too_wide_for_the_target(self):
        ast = _fp32_args(_sum_of_squares, 2)
        with pytest.raises(TransformDeclined, match='not representable'):
            RoundInsert.apply(ast, fp.FP64, where=0)   # the add

    @pytest.mark.parametrize('ctx', [fp.FP32, fp.FP16])
    def test_a_format_too_narrow(self, ctx):
        ast = _fp32_args(_sum_of_squares, 2)
        with pytest.raises(TransformDeclined, match='not representable'):
            RoundInsert.apply(ast, ctx, where=1)

    def test_a_narrow_format_is_skipped_without_a_where(self):
        ast = _fp32_args(_sum_of_squares, 2)
        assert RoundInsert.apply(ast, fp.FP16).is_equiv(ast)

    def test_a_stochastic_target(self):
        ast = _fp32_args(_sum_of_squares, 2)
        stochastic = fp.IEEEContext(8, 32, num_randbits=4)
        with pytest.raises(TransformDeclined, match='stochastically'):
            RoundInsert.apply(ast, stochastic, where=1)

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
        ast = _fp32_args(_sum_of_squares, 2)
        with pytest.raises(TypeError):
            RoundInsert.apply(ast, fp.FP64.format())  # type: ignore[arg-type]

    def test_a_where_out_of_range(self):
        ast = _fp32_args(_sum_of_squares, 2)
        with pytest.raises(TransformReferenceError, match='candidate site'):
            RoundInsert.apply(ast, fp.FP64, where=99)

    def test_a_where_of_the_wrong_type(self):
        ast = _fp32_args(_sum_of_squares, 2)
        with pytest.raises(TypeError):
            RoundInsert.apply(ast, fp.FP64, where='0')  # type: ignore[arg-type]
