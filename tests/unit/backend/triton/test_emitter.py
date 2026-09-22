"""Scalar expression emission: `fpy2.backend.triton.emitter`.

The cast discipline is what these pin.  Triton types `fp16 op fp16` as fp16,
so an operand is cast *into* the storage its signature wants before the
operation -- never after it, which would widen an already-rounded value.
"""

import pytest

import fpy2 as fp
from fpy2 import Module
from fpy2.ast.fpyast import Add, BinaryOp, Div, Mul, Sqrt, UnaryOp
from fpy2.ast.visitor import DefaultVisitor
from fpy2.backend.triton.emitter import (
    TritonEmitError,
    emit_block,
    emit_expr,
)
from fpy2.transform import Specialize
from fpy2.types import RealType

FP16 = fp.IEEEContext(5, 16)


class _Find(DefaultVisitor):
    def __init__(self, want):
        super().__init__()
        self.want = want
        self.out = []

    def _visit_binaryop(self, e, ctx):
        if isinstance(e, self.want):
            self.out.append(e)
        return super()._visit_binaryop(e, ctx)

    def _visit_unaryop(self, e, ctx):
        if isinstance(e, self.want):
            self.out.append(e)
        return super()._visit_unaryop(e, ctx)


def _find(func, kind):
    v = _Find(kind)
    v._visit_function(func.ast, None)
    assert v.out, f'no {kind.__name__} in the program'
    return v.out[0]


def _spec(func, *ctxs, ctx=None):
    m = Module()
    m.add(func, ctx=ctx, arg_types=[RealType(c) for c in ctxs])
    return Specialize.apply(m, size_key=True).get(func.name).func


class TestCastDiscipline:
    """`exploration/triton/kernels.py` measured the trap at 2000/2000."""

    def test_an_exact_product_widens_its_operands(self):
        """The `dot_exact` spelling: cast *then* multiply."""
        @fp.fpy(ctx=fp.REAL)
        def prod(x: fp.Real, y: fp.Real):
            p = x * y
            with fp.FP32:
                return p + p

        f = _spec(prod, FP16, FP16, ctx=fp.REAL)
        assert emit_expr(_find(f, Mul), f.ast) == \
            '(x.to(tl.float32) * y.to(tl.float32))'

    def test_the_trap_spelling_is_never_emitted(self):
        """`(x * y).to(tl.float32)` multiplies in fp16 and widens the
        *rounded* product.  Nothing may produce it."""
        @fp.fpy(ctx=fp.REAL)
        def prod(x: fp.Real, y: fp.Real):
            p = x * y
            with fp.FP32:
                return p + p

        f = _spec(prod, FP16, FP16, ctx=fp.REAL)
        out = emit_expr(_find(f, Mul), f.ast)
        assert not out.startswith('(x * y)')

    def test_same_storage_needs_no_cast(self):
        @fp.fpy(ctx=fp.FP32)
        def add(x: fp.Real, y: fp.Real):
            return x + y

        f = _spec(add, fp.FP32, fp.FP32, ctx=fp.FP32)
        assert emit_expr(_find(f, Add), f.ast) == '(x + y)'


class TestSpelling:
    def test_infix(self):
        @fp.fpy(ctx=fp.FP32)
        def add(x: fp.Real, y: fp.Real):
            return x + y

        f = _spec(add, fp.FP32, fp.FP32, ctx=fp.FP32)
        assert emit_expr(_find(f, Add), f.ast) == '(x + y)'

    def test_call(self):
        """`/` and `sqrt` are the correctly-rounded variants, not the fast
        ones -- the table names `tl.div_rn` and `tl.sqrt_rn`."""
        @fp.fpy(ctx=fp.FP32)
        def div(x: fp.Real, y: fp.Real):
            return x / y

        f = _spec(div, fp.FP32, fp.FP32, ctx=fp.FP32)
        assert emit_expr(_find(f, Div), f.ast) == 'tl.div_rn(x, y)'


class TestRefusals:
    """Every omission in the op table is a refusal, not a gap."""

    def test_an_absent_operation(self):
        """No transcendental is in the table: none is correctly rounded."""
        @fp.fpy(ctx=fp.FP32)
        def f(x: fp.Real):
            return fp.exp(x)

        g = _spec(f, fp.FP32, ctx=fp.FP32)
        from fpy2.ast.fpyast import Exp
        with pytest.raises(TritonEmitError, match='no signatures for op'):
            emit_expr(_find(g, Exp), g.ast)

    def test_division_at_fp16(self):
        """Triton types `fp16 / fp16` as fp32, so the result is a double
        rounding rather than FP16's division."""
        @fp.fpy(ctx=FP16)
        def f(x: fp.Real, y: fp.Real):
            return x / y

        g = _spec(f, FP16, FP16, ctx=FP16)
        with pytest.raises(TritonEmitError, match='no matching signature'):
            emit_expr(_find(g, Div), g.ast)

    def test_a_non_expr_is_a_type_error(self):
        @fp.fpy(ctx=fp.FP32)
        def f(x: fp.Real):
            return x

        with pytest.raises(TypeError, match='Expr'):
            emit_expr(42, f.ast)


class TestBlock:
    def test_a_straight_line_body(self):
        @fp.fpy(ctx=fp.FP32)
        def f(x: fp.Real, y: fp.Real):
            a = x + y
            b = a * a
            return b

        g = _spec(f, fp.FP32, fp.FP32, ctx=fp.FP32)
        assert emit_block(g.ast.body, g.ast) == (
            'a = (x + y)\n'
            'b = (a * a)\n'
            'return b'
        )


class TestSequentialLoops:
    """A loop `why_not_tileable` declined stays sequential per lane, which is
    `tl.static_range` -- the shape `kernels.dot_exact` uses for its fold."""

    def test_a_proven_count_emits_static_range(self):
        @fp.fpy(ctx=fp.FP32)
        def fold(x: fp.Real):
            acc = fp.round(0)
            for k in range(8):
                acc = acc + x
            return acc

        g = _spec(fold, fp.FP32, ctx=fp.FP32)
        assert emit_block(g.ast.body, g.ast) == (
            'acc = 0\n'
            'for k in tl.static_range(8):\n'
            '    acc = (acc + x)\n'
            'return acc'
        )

    def test_an_unproven_count_is_refused(self):
        """A foreign constant is not folded until `ConstFold` runs, and
        `tl.static_range` needs the count as a `constexpr`."""
        width = 8

        @fp.fpy(ctx=fp.FP32)
        def fold(x: fp.Real):
            acc = fp.round(0)
            for k in range(width):
                acc = acc + x
            return acc

        g = _spec(fold, fp.FP32, ctx=fp.FP32)
        with pytest.raises(TritonEmitError, match='compile-time trip count'):
            emit_block(g.ast.body, g.ast)

    def test_const_folding_makes_it_emittable(self):
        """The refusal is the pipeline's to fix, not the emitter's."""
        from fpy2.transform import ConstFold
        width = 8

        @fp.fpy(ctx=fp.FP32)
        def fold(x: fp.Real):
            acc = fp.round(0)
            for k in range(width):
                acc = acc + x
            return acc

        g = _spec(fold, fp.FP32, ctx=fp.FP32)
        folded = ConstFold.apply(g.ast)
        assert 'tl.static_range(8)' in emit_block(folded.body, folded)


class TestSelect:
    def test_if_expr_is_tl_where(self):
        @fp.fpy(ctx=fp.FP32)
        def sel(c: bool, x: fp.Real, y: fp.Real):
            return x if c else y

        m = Module()
        m.add(sel, ctx=fp.FP32,
              arg_types=[None, RealType(fp.FP32), RealType(fp.FP32)])
        g = Specialize.apply(m, size_key=True).get('sel').func
        assert emit_block(g.ast.body, g.ast) == 'return tl.where(c, x, y)'


class TestContextStatements:
    def test_a_with_emits_nothing_of_its_own(self):
        """A context change is a change of storage, which the dispatch reads
        per expression."""
        @fp.fpy(ctx=fp.REAL)
        def f(x: fp.Real, y: fp.Real):
            p = x * y
            with fp.FP32:
                return p + p

        g = _spec(f, FP16, FP16, ctx=fp.REAL)
        out = emit_block(g.ast.body, g.ast)
        assert 'with' not in out
        assert out == (
            'p = (x.to(tl.float32) * y.to(tl.float32))\n'
            'return (p + p)'
        )
