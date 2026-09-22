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
from fpy2.types import ListType, RealType

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


_R32 = RealType(fp.FP32)
_INT = RealType(fp.INTEGER)


def _emit(func, argt, ctx=fp.FP32):
    m = Module()
    m.add(func, ctx=ctx, arg_types=argt)
    g = Specialize.apply(m, size_key=True).get(func.name).func
    return emit_block(g.ast.body, g.ast)


class TestMemory:
    def test_a_flat_load(self):
        @fp.fpy(ctx=fp.FP32)
        def f(xs: list[fp.Real], i: fp.Real):
            return xs[i]

        assert _emit(f, [ListType(_R32, 8), _INT]) == 'return tl.load(xs_ptr + i)'

    def test_a_nested_subscript_flattens_row_major(self):
        @fp.fpy(ctx=fp.FP32)
        def f(xss: list[list[fp.Real]], r: fp.Real, k: fp.Real):
            return xss[r][k]

        out = _emit(f, [ListType(ListType(_R32, 8), 4), _INT, _INT])
        assert out == 'return tl.load(xss_ptr + r * 8 + k)'

    @pytest.mark.parametrize('rows,cols', [(4, 8), (3, 5), (1, 2), (7, 1)])
    def test_the_offset_agrees_with_row_major_flattening(self, rows, cols):
        """The differential for the indexing: evaluate the emitted offset for
        every cell and compare against the flat index."""
        @fp.fpy(ctx=fp.FP32)
        def f(xss: list[list[fp.Real]], r: fp.Real, k: fp.Real):
            return xss[r][k]

        out = _emit(f, [ListType(ListType(_R32, cols), rows), _INT, _INT])
        expr = out[out.index('xss_ptr + ') + len('xss_ptr + '):].rstrip(')')
        for r in range(rows):
            for k in range(cols):
                got = eval(expr, {}, {'r': r, 'k': k})
                assert got == r * cols + k, (r, k, expr)

    def test_a_store(self):
        @fp.fpy(ctx=fp.FP32)
        def f(xs: list[fp.Real], out: list[fp.Real], i: fp.Real):
            out[i] = xs[i]
            return out

        emitted = _emit(f, [ListType(_R32, 8), ListType(_R32, 8), _INT])
        assert emitted.splitlines()[0] == \
            'tl.store(out_ptr + i, tl.load(xs_ptr + i))'

    def test_an_unproven_length_is_refused(self):
        """A kernel argument is a flat pointer, so an unproven length has no
        offset arithmetic to emit."""
        @fp.fpy(ctx=fp.FP32)
        def f(xs: list[fp.Real], i: fp.Real):
            return xs[i]

        with pytest.raises(TritonEmitError, match='no proven length'):
            _emit(f, [ListType(_R32), _INT])


class TestMask:
    def test_a_guard_becomes_the_mask_on_both_ends(self):
        """`tile_loops` emits `if j < n` around an element write.  That is not
        a branch: it is the `mask=` of every access under it."""
        @fp.fpy(ctx=fp.FP32)
        def f(xs: list[fp.Real], out: list[fp.Real], j: fp.Real, n: fp.Real):
            if j < n:
                out[j] = xs[j]
            return out

        emitted = _emit(
            f, [ListType(_R32, 8), ListType(_R32, 8), _INT, _INT])
        first = emitted.splitlines()[0]
        assert 'if' not in emitted
        assert first == (
            'tl.store(out_ptr + j, '
            'tl.load(xs_ptr + j, mask=(j < n), other=0.0), mask=(j < n))'
        )


class TestLiteralCast:
    def test_a_numeric_literal_is_parenthesized(self):
        """`2.to(...)` lexes as `2.` then `to` -- a different program."""
        @fp.fpy(ctx=fp.FP32)
        def f(xs: list[fp.Real], i: fp.Real):
            return xs[i] * 2

        out = _emit(f, [ListType(_R32, 8), _INT])
        assert '(2).to(tl.float32)' in out
        assert '2.to(' not in out

    def test_everything_emitted_is_parseable_python(self):
        """The emitter's output has to lex, whatever else it is."""
        import ast as pyast

        @fp.fpy(ctx=fp.FP32)
        def f(xs: list[fp.Real], out: list[fp.Real], j: fp.Real, n: fp.Real):
            if j < n:
                out[j] = xs[j] * 2 + xs[j]
            return out

        emitted = _emit(
            f, [ListType(_R32, 8), ListType(_R32, 8), _INT, _INT])
        pyast.parse(emitted)


class TestNamedRefusals:
    """Dispatch is `Visitor`'s, so a node this backend cannot spell has its
    own refusal naming the construct -- not a catch-all."""

    def test_a_comprehension_names_itself(self):
        @fp.fpy(ctx=fp.FP32)
        def f(xs: list[fp.Real]):
            return [x * 2 for x in xs]

        with pytest.raises(TritonEmitError, match='comprehension'):
            _emit(f, [ListType(_R32, 8)])

    def test_a_tuple_names_itself(self):
        @fp.fpy(ctx=fp.FP32)
        def f(x: fp.Real):
            return (x, x)

        with pytest.raises(TritonEmitError, match='tuple'):
            _emit(f, [_R32])

    def test_an_assert_names_itself(self):
        """A kernel cannot raise."""
        @fp.fpy(ctx=fp.FP32)
        def f(x: fp.Real):
            assert x > 0, 'positive'
            return x

        with pytest.raises(TritonEmitError, match='cannot raise'):
            _emit(f, [_R32])

    def test_every_abstract_visit_method_is_implemented(self):
        """`Visitor` is an ABC, so a node kind added to the AST breaks this
        backend's build rather than falling into a catch-all."""
        import inspect
        from fpy2.ast.visitor import Visitor
        from fpy2.backend.triton.emitter import _Emitter

        abstract = {
            n for n, m in inspect.getmembers(Visitor, inspect.isfunction)
            if getattr(m, '__isabstractmethod__', False)
        }
        assert abstract, 'expected Visitor to declare abstract methods'
        assert not (abstract - set(dir(_Emitter)))
        assert not getattr(_Emitter, '__abstractmethods__', frozenset())


class TestKernelBody:
    """End to end: the batched dot product, through the whole pipeline, comes
    out shaped like `exploration/triton/kernels.py`'s `dot_exact`."""

    @staticmethod
    def _pipeline():
        from fpy2.backend.triton import normalize_module, tile_loops
        FP16 = fp.IEEEContext(5, 16)

        @fp.fpy(ctx=fp.REAL)
        def batched_dot(xss: list[list[fp.Real]], yss: list[list[fp.Real]],
                        out: list[fp.Real], BLOCK: fp.Real):
            for r in range(len(xss)):
                acc = fp.round(0)
                for k in range(8):
                    with fp.FP32:
                        acc = acc + xss[r][k] * yss[r][k]
                out[r] = acc
            return out

        m = Module()
        m.add(batched_dot, ctx=fp.REAL, arg_types=[
            ListType(ListType(RealType(FP16), 8), 4),
            ListType(ListType(RealType(FP16), 8), 4),
            ListType(RealType(fp.FP32), 4),
            RealType(fp.INTEGER)])
        g = Specialize.apply(m, size_key=True).get('batched_dot').func
        m2 = Module()
        m2.add(g)
        n = normalize_module(m2).get(g.name).func
        r = tile_loops(n.ast, 'BLOCK')
        return emit_block(r.func.body, r.func, r.tiled, drop_asserts=True)

    def test_the_grid_and_tile(self):
        out = self._pipeline()
        assert 'tl.program_id(0) * ' in out
        assert 'tl.arange(0, ' in out

    def test_the_refused_fold_stays_sequential(self):
        """`why_not_tileable` declines the `K` loop because the accumulation
        rounds, which is why `dot_exact` keeps it per-lane."""
        assert 'tl.static_range(8)' in self._pipeline()

    def test_both_operands_are_widened_before_the_product(self):
        """The trap, measured at 2000/2000: an fp16 product must not be
        computed in fp16 and widened after."""
        out = self._pipeline()
        assert out.count('.to(tl.float32)') == 2
        assert ').to(tl.float32) * ' not in out.replace(
            'other=0.0).to(tl.float32) * ', '')

    def test_the_mask_reaches_every_access(self):
        out = self._pipeline()
        assert out.count('mask=') == 3       # two loads and the store
        assert 'if ' not in out

    def test_it_parses(self):
        import ast as pyast
        pyast.parse(self._pipeline())


class TestKernel:
    """The whole `@triton.jit` function."""

    @staticmethod
    def _kernel(ctx, elt):
        from fpy2.backend.triton import (
            emit_kernel, normalize_module, tile_loops,
        )

        @fp.fpy(ctx=ctx)
        def dot(xss: list[list[fp.Real]], yss: list[list[fp.Real]],
                out: list[fp.Real], BLOCK: fp.Real):
            for r in range(len(xss)):
                acc = fp.round(0)
                for k in range(8):
                    with fp.FP32:
                        acc = acc + xss[r][k] * yss[r][k]
                out[r] = acc
            return out

        m = Module()
        m.add(dot, ctx=ctx, arg_types=[
            ListType(ListType(RealType(elt), 8), 4),
            ListType(ListType(RealType(elt), 8), 4),
            ListType(RealType(fp.FP32), 4), RealType(fp.INTEGER)])
        g = Specialize.apply(m, size_key=True).get('dot').func
        m2 = Module()
        m2.add(g)
        n = normalize_module(m2).get(g.name).func
        r = tile_loops(n.ast, 'BLOCK')
        return emit_kernel(r.func, r.tiled, block='BLOCK', drop_asserts=True)

    def test_the_signature(self):
        """A list becomes a pointer, the tile width a `constexpr`."""
        k = self._kernel(fp.REAL, fp.IEEEContext(5, 16))
        assert k.params == (
            'xss_ptr', 'yss_ptr', 'out_ptr', 'BLOCK: tl.constexpr')
        assert k.source.startswith('@triton.jit\ndef dot(')

    def test_a_kernel_does_not_return(self):
        """It writes through its pointers, which is why the program it comes
        from takes its output as an argument."""
        k = self._kernel(fp.REAL, fp.IEEEContext(5, 16))
        assert 'return' not in k.source

    def test_it_parses(self):
        import ast as pyast
        pyast.parse(self._kernel(fp.REAL, fp.IEEEContext(5, 16)).source)

    def test_fusion_is_derived_not_pinned(self):
        """Against the hardware audit in `exploration/triton/`: an FP16-in
        program is unchanged by fusion (0/2000 either way), an all-FP32 one
        differs under it (590/2000).  So the first may fuse and the second
        may not, and the flag has to say so without being told."""
        assert self._kernel(fp.REAL, fp.IEEEContext(5, 16)).enable_fp_fusion
        assert not self._kernel(fp.FP32, fp.FP32).enable_fp_fusion
