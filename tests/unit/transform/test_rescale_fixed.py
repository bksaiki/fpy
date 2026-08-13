"""
Unit tests for the :class:`fpy2.transform.RescaleFixed` transform.

The rewrite mints fresh ``_tN`` names via ``Gensym``, so comparing
against a hand-written golden AST is brittle.  These tests assert:

1. **Structural shape**: the block's context lands at scale zero, and
   the round is wrapped by two ``with fp.REAL:`` scalings.
2. **Negative checks**: blocks the rewrite must not touch compare via
   ``is_equiv`` against the original AST.
3. **Semantic equivalence** via the interpreter, since scaling by a
   power of two commutes with fixed-point rounding exactly — including
   at the wrap/saturate boundary and for every rounding mode.
"""

import fpy2 as fp
import pytest

from fpy2.ast.fpyast import Call, ContextStmt, ForeignVal, FuncDef
from fpy2.ast.visitor import DefaultVisitor
from fpy2.number import REAL, OverflowMode, RoundingMode
from fpy2.transform import RescaleFixed


# ----------------------------------------------------------------------
# Helpers


def _blocks(ast: FuncDef) -> list[ContextStmt]:
    """Every ``ContextStmt`` in *ast*, outermost first."""
    found: list[ContextStmt] = []

    class _C(DefaultVisitor):
        def _visit_context(self, stmt: ContextStmt, ctx):
            found.append(stmt)
            super()._visit_context(stmt, ctx)

    _C()._visit_function(ast, None)
    return found


def _fixed_scales(ast: FuncDef) -> list[int]:
    """The scale of every ``FixedContext`` block in *ast*."""
    scales: list[int] = []
    for stmt in _blocks(ast):
        e = stmt.ctx
        if isinstance(e, Call) and e.fn is fp.FixedContext:
            scales.append(e.args[1].val)
        elif isinstance(e, ForeignVal) and isinstance(e.val, fp.FixedContext):
            scales.append(e.val.scale)
    return scales


def _real_blocks(ast: FuncDef) -> int:
    """The number of ``with fp.REAL:`` blocks in *ast*."""
    return sum(
        1 for s in _blocks(ast)
        if isinstance(s.ctx, ForeignVal) and s.ctx.val is REAL
    )


def _eval(ast: FuncDef, fn: fp.Function, *args):
    """Run *ast* through the interpreter using *fn*'s env."""
    return fn.with_ast(ast)(*args)


def _same(a, b) -> bool:
    """Bit-exact comparison that also matches NaN to NaN."""
    if a.isnan or b.isnan:
        return a.isnan and b.isnan
    if a.isinf or b.isinf:
        return a.isinf and b.isinf and a.s == b.s
    return a.as_rational() == b.as_rational() and a.s == b.s


def _quantizer(ctx: fp.FixedContext) -> fp.Function:
    @fp.fpy(ctx=fp.REAL)
    def q(x):
        with ctx:
            xq = fp.round(x)
        return xq
    return q


# ----------------------------------------------------------------------
# The rewrite


class TestRescale:
    """A quantization block moves to scale zero, with exact scalings
    around the round."""

    def test_shape(self):
        @fp.fpy(ctx=fp.REAL)
        def f(a):
            with fp.FixedContext(True, -16, 32):
                aq = fp.round(a)
            return aq

        out = RescaleFixed.apply(f.ast)
        assert _fixed_scales(out) == [0]
        assert _real_blocks(out) == 2
        assert _same(_eval(out, f, 0.1), f(0.1))

    def test_keeps_the_written_call_form(self):
        """The context stays a ``FixedContext(...)`` call, so the
        rewritten program reads like the original."""

        @fp.fpy(ctx=fp.REAL)
        def f(a):
            with fp.FixedContext(True, -8, 16):
                aq = fp.round(a)
            return aq

        block = _blocks(RescaleFixed.apply(f.ast))[0]
        assert isinstance(block.ctx, Call) and block.ctx.fn is fp.FixedContext

    def test_positive_scale(self):
        """A scale above zero shifts down instead of up."""

        @fp.fpy(ctx=fp.REAL)
        def f(a):
            with fp.FixedContext(True, 4, 16):
                aq = fp.round(a)
            return aq

        out = RescaleFixed.apply(f.ast)
        assert _fixed_scales(out) == [0]
        for x in (0.0, 1.0, 100.0, -37.5, 1e4):
            assert _same(_eval(out, f, x), f(x)), x

    def test_several_rounds_in_one_block(self):
        @fp.fpy(ctx=fp.REAL)
        def f(a, b):
            with fp.FixedContext(True, -16, 32):
                aq = fp.round(a)
                bq = fp.round(b)
            with fp.FP64:
                s = aq + bq
            return s

        out = RescaleFixed.apply(f.ast)
        assert _fixed_scales(out) == [0]
        assert _real_blocks(out) == 4
        assert _same(_eval(out, f, 0.1, 0.2), f(0.1, 0.2))

    def test_chained_rounds(self):
        """A round of an earlier output still sees the original
        magnitude, since each round scales in and back out."""

        @fp.fpy(ctx=fp.REAL)
        def f(a):
            with fp.FixedContext(True, -16, 32):
                aq = fp.round(a)
                bq = fp.round(aq)
            return bq

        out = RescaleFixed.apply(f.ast)
        assert _same(_eval(out, f, 0.1), f(0.1))

    def test_cast(self):
        @fp.fpy(ctx=fp.REAL)
        def f(a):
            with fp.FixedContext(True, -4, 16):
                aq = fp.cast(a)
            return aq

        out = RescaleFixed.apply(f.ast)
        assert _fixed_scales(out) == [0]
        assert _same(_eval(out, f, 0.25), f(0.25))

    def test_inside_a_loop(self):
        @fp.fpy(ctx=fp.REAL)
        def f(A):
            acc = 0
            for a in A:
                with fp.FixedContext(True, -16, 32):
                    aq = fp.round(a)
                with fp.FP64:
                    acc += aq
            return acc

        out = RescaleFixed.apply(f.ast)
        assert _fixed_scales(out) == [0]
        A = [0.1, 0.25, -3.5, 1e-6, 7.0]
        assert _same(_eval(out, f, A), f(A))


# ----------------------------------------------------------------------
# Blocks the rewrite must leave alone


class TestUnchanged:

    def test_already_integral(self):
        @fp.fpy(ctx=fp.REAL)
        def f(a):
            with fp.FixedContext(True, 0, 32):
                aq = fp.round(a)
            return aq

        assert RescaleFixed.apply(f.ast).is_equiv(f.ast)

    def test_not_fixed_point(self):
        @fp.fpy(ctx=fp.REAL)
        def f(a):
            with fp.FP64:
                aq = fp.round(a)
            return aq

        assert RescaleFixed.apply(f.ast).is_equiv(f.ast)

    def test_arithmetic_in_body(self):
        """Scaling does not commute with a product of scaled values."""

        @fp.fpy(ctx=fp.REAL)
        def f(a, b):
            with fp.FixedContext(True, -16, 32):
                p = a * b
            return p

        assert RescaleFixed.apply(f.ast).is_equiv(f.ast)

    def test_round_of_an_expression(self):
        """The argument must be a variable: an inner expression would
        round under the block's context before the shift applies."""

        @fp.fpy(ctx=fp.REAL)
        def f(a, b):
            with fp.FixedContext(True, -16, 32):
                aq = fp.round(a + b)
            return aq

        assert RescaleFixed.apply(f.ast).is_equiv(f.ast)

    def test_bound_context(self):
        """``with C as c:`` exposes the context to the body as a value,
        which the rescaled context would change."""

        @fp.fpy(ctx=fp.REAL)
        def f(a):
            with fp.FixedContext(True, -16, 32) as c:
                aq = fp.round(a)
            return aq

        assert RescaleFixed.apply(f.ast).is_equiv(f.ast)

    def test_nested_block_in_body(self):
        @fp.fpy(ctx=fp.REAL)
        def f(a):
            with fp.FixedContext(True, -16, 32):
                with fp.FP64:
                    aq = fp.round(a)
            return aq

        assert RescaleFixed.apply(f.ast).is_equiv(f.ast)


# ----------------------------------------------------------------------
# Semantic equivalence


class TestEquivalence:
    """Rounding commutes with a power-of-two shift exactly, so the
    rewrite must be bit-exact for every format parameter."""

    @pytest.mark.parametrize('signed', [True, False])
    @pytest.mark.parametrize('scale', [-16, -4, -1, 1, 3])
    @pytest.mark.parametrize('nbits', [8, 32])
    def test_formats(self, signed, scale, nbits):
        ctx = fp.FixedContext(signed, scale, nbits)
        f = _quantizer(ctx)
        out = RescaleFixed.apply(f.ast)
        assert _fixed_scales(out) == [0]

        hi = 2.0 ** (nbits + scale)
        xs = [0.0, 2.0 ** scale, 1.0, hi / 3, hi, 2 * hi, 1e-9]
        if signed:
            xs += [-x for x in xs]
        for x in xs:
            assert _same(_eval(out, f, x), f(x)), (signed, scale, nbits, x)

    @pytest.mark.parametrize('rm', list(RoundingMode))
    @pytest.mark.parametrize('overflow', [OverflowMode.WRAP, OverflowMode.SATURATE])
    def test_rounding_and_overflow_modes(self, rm, overflow):
        ctx = fp.FixedContext(True, -8, 16, rm, overflow)
        f = _quantizer(ctx)
        out = RescaleFixed.apply(f.ast)
        for x in (0.1, -0.1, 2.0 ** -9, 127.9961, 128.0, 1e5, -1e5, 0.0, -0.0):
            assert _same(_eval(out, f, x), f(x)), (rm, overflow, x)
