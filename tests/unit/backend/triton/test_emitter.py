"""Scalar expression emission: `fpy2.backend.triton.emitter`.

The cast discipline is what these pin.  Triton types `fp16 op fp16` as fp16,
so an operand is cast *into* the storage its signature wants before the
operation -- never after it, which would widen an already-rounded value.
"""

import ast as pyast
import inspect
import re

import pytest

import fpy2 as fp
from fpy2 import Function, Module
from fpy2.ast.fpyast import Add, Div, Exp, Expr, If1Stmt, Mul
from fpy2.ast.visitor import Visitor
from fpy2.backend.triton import KernelSource, TritonCompiler
from fpy2.backend.triton.emitter import (
    TritonEmitError,
    _Emitter,
    emit_block,
    emit_expr,
)
from fpy2.number import Context
from fpy2.transform import Specialize
from fpy2.transform.path import walk_exprs
from fpy2.types import ListType, RealType, Type
from fpy2.utils import NamedId

FP16 = fp.IEEEContext(5, 16)
_R32 = RealType(fp.FP32)
_INT = RealType(fp.INTEGER)


def _spec(func: Function, argt: list[Type | None], ctx: Context = fp.FP32) -> Function:
    m = Module()
    m.add(func, ctx=ctx, arg_types=argt)
    return Specialize.apply(m, size_key=True).get(func.name).func


def _find(func: Function, kind: type[Expr]) -> Expr:
    """The first *kind* node in *func*."""
    return next(e for _, e in walk_exprs(func.ast) if isinstance(e, kind))


def _emit(
    func: Function, argt: list[Type | None], ctx: Context = fp.FP32,
    *, guard: bool = False, drop_asserts: bool = False,
) -> str:
    """*guard* names the body's top-level `if1`s as tile guards."""
    g = _spec(func, argt, ctx)
    guards = [s for s in g.ast.body.stmts if isinstance(s, If1Stmt)]
    return emit_block(g.ast.body, g.ast, guards=guards if guard else (),
                      drop_asserts=drop_asserts)


def _flat(func: Function, n_out: int = 8, ctx: Context = fp.FP32) -> KernelSource:
    """*func* compiled over eight `f32` inputs and *n_out* outputs."""
    return TritonCompiler(drop_asserts=True).compile(
        func, ctx=ctx, arg_types=[ListType(_R32, 8), ListType(_R32, n_out), _INT])


def _rows(
    func: Function, n_in: int, n_out: int,
    elt: Context = fp.FP32, out: Context = fp.FP32,
) -> KernelSource:
    """*func* compiled over a runtime count of rows."""
    rows = NamedId('rows')
    return TritonCompiler(drop_asserts=True).compile(func, ctx=fp.REAL, arg_types=[
        ListType(ListType(RealType(elt), n_in), rows),
        ListType(ListType(RealType(out), n_out), rows), _INT])


@fp.fpy(ctx=fp.REAL)
def _prod(x: fp.Real, y: fp.Real):
    p = x * y
    with fp.FP32:
        return p + p


class TestCastDiscipline:
    def test_an_exact_product_widens_its_operands(self):
        """Cast *then* multiply: an fp16 product is never rounded in fp16 and
        widened after."""
        f = _spec(_prod, [RealType(FP16)] * 2, fp.REAL)
        assert emit_expr(_find(f, Mul), f.ast) == \
            '(x.to(tl.float32) * y.to(tl.float32))'

    def test_same_storage_needs_no_cast(self):
        @fp.fpy(ctx=fp.FP32)
        def add(x: fp.Real, y: fp.Real):
            return x + y

        f = _spec(add, [_R32, _R32])
        assert emit_expr(_find(f, Add), f.ast) == '(x + y)'


class TestSpelling:
    def test_call(self):
        """`/` and `sqrt` are the correctly-rounded variants, not the fast
        ones -- the table names `tl.div_rn` and `tl.sqrt_rn`."""
        @fp.fpy(ctx=fp.FP32)
        def div(x: fp.Real, y: fp.Real):
            return x / y

        f = _spec(div, [_R32, _R32])
        assert emit_expr(_find(f, Div), f.ast) == 'tl.div_rn(x, y)'


class TestRefusals:
    """Every omission in the op table is a refusal, not a gap."""

    def test_an_absent_operation(self):
        """No transcendental is in the table: none is correctly rounded."""
        @fp.fpy(ctx=fp.FP32)
        def f(x: fp.Real):
            return fp.exp(x)

        g = _spec(f, [_R32])
        with pytest.raises(TritonEmitError, match='no signatures for op'):
            emit_expr(_find(g, Exp), g.ast)

    def test_division_at_fp16(self):
        """Triton types `fp16 / fp16` as fp32, so the result is a double
        rounding rather than FP16's division."""
        @fp.fpy(ctx=FP16)
        def f(x: fp.Real, y: fp.Real):
            return x / y

        g = _spec(f, [RealType(FP16)] * 2, FP16)
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

        assert _emit(f, [_R32, _R32]) == (
            'a = (x + y)\n'
            'b = (a * a)\n'
            'return b'
        )


class TestSequentialLoops:
    """A loop `why_not_tileable` declined stays sequential per lane, which is
    `tl.static_range` when its count is proven."""

    def test_a_proven_count_emits_static_range(self):
        @fp.fpy(ctx=fp.FP32)
        def fold(x: fp.Real):
            acc = fp.round(0)
            for _k in range(8):
                acc = acc + x
            return acc

        assert _emit(fold, [_R32]) == (
            'acc = 0.0\n'
            'for _k in tl.static_range(8):\n'
            '    acc = (acc + x)\n'
            'return acc'
        )

    def test_a_foreign_constant_is_resolved_by_the_size_analysis(self):
        """`ArraySizeInfer` proves the iterable's length through the free
        variable, and `static_trip_count` reads it."""
        width = 8

        @fp.fpy(ctx=fp.FP32)
        def fold(x: fp.Real):
            acc = fp.round(0)
            for _k in range(width):
                acc = acc + x
            return acc

        assert 'tl.static_range(8)' in _emit(fold, [_R32])

    def test_a_runtime_count_carrying_a_scalar(self):
        """A loop at runtime carries each value at one type.  Outside a row
        tile a literal start is one: Triton types it by the body."""
        @fp.fpy(ctx=fp.FP32)
        def fold(x: fp.Real, n: fp.Real):
            acc = fp.round(0)
            for _k in range(n):
                acc = acc + x
            return acc

        out = _emit(fold, [_R32, _INT])
        assert 'for _k in range(n):' in out
        assert 'acc = (acc + x)' in out


class TestSelect:
    def test_if_expr_is_tl_where(self):
        @fp.fpy(ctx=fp.FP32)
        def sel(c: bool, x: fp.Real, y: fp.Real):
            return x if c else y

        assert _emit(sel, [None, _R32, _R32]) == 'return tl.where(c, x, y)'


class TestContextStatements:
    def test_a_with_emits_nothing_of_its_own(self):
        """A context change is a change of storage, which the dispatch reads
        per expression."""
        assert _emit(_prod, [RealType(FP16)] * 2, fp.REAL) == (
            'p = (x.to(tl.float32) * y.to(tl.float32))\n'
            'return (p + p)'
        )


class TestMemory:
    def test_a_flat_load(self):
        @fp.fpy(ctx=fp.FP32)
        def f(xs: list[fp.Real], i: fp.Real):
            return xs[i]

        assert _emit(f, [ListType(_R32, 8), _INT]) == \
            '__t0 = tl.load(xs_ptr + i)\nreturn __t0'

    def test_a_nested_subscript_flattens_row_major(self):
        @fp.fpy(ctx=fp.FP32)
        def f(xss: list[list[fp.Real]], r: fp.Real, k: fp.Real):
            return xss[r][k]

        out = _emit(f, [ListType(ListType(_R32, 8), 4), _INT, _INT])
        assert out == '__t0 = tl.load(xss_ptr + r * 8 + k)\nreturn __t0'

    @pytest.mark.parametrize('rows,cols', [(4, 8), (3, 5), (1, 2), (7, 1)])
    def test_the_offset_agrees_with_row_major_flattening(self, rows, cols):
        """The differential for the indexing: evaluate the emitted offset for
        every cell and compare against the flat index."""
        @fp.fpy(ctx=fp.FP32)
        def f(xss: list[list[fp.Real]], r: fp.Real, k: fp.Real):
            return xss[r][k]

        out = _emit(f, [ListType(ListType(_R32, cols), rows), _INT, _INT])
        load = out.splitlines()[0]
        expr = load[load.index('xss_ptr + ') + len('xss_ptr + '):].rstrip(')')
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
        assert emitted.splitlines()[:2] == [
            '__t0 = tl.load(xs_ptr + i)', 'tl.store(out_ptr + i, __t0)']

    def test_a_ragged_length_is_refused(self):
        """A kernel argument is a flat pointer, so a row length neither proven
        nor named -- the rows may differ -- has no stride to emit."""
        @fp.fpy(ctx=fp.FP32)
        def f(xss: list[list[fp.Real]], i: fp.Real):
            return xss[i][i]

        with pytest.raises(TritonEmitError, match='no proven length'):
            _emit(f, [ListType(ListType(_R32)), _INT])


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
            f, [ListType(_R32, 8), ListType(_R32, 8), _INT, _INT], guard=True)
        assert 'if' not in emitted
        assert emitted.splitlines()[:2] == [
            '__t0 = tl.load(xs_ptr + j, mask=(j < n), other=0.0)',
            'tl.store(out_ptr + j, __t0, mask=(j < n))',
        ]


class TestLiteralCast:
    def test_a_numeric_literal_is_retyped_not_cast(self):
        """Two traps in one.  `2.to(...)` lexes as `2.` then `to`, and
        parenthesizing it to `(2).to(...)` only moves the problem: that is
        valid Python and Triton rejects it, because a Python scalar is a
        `constexpr` rather than a tile -- *"'int' object has no attribute
        'to'"*.  Writing the literal in the target's own spelling avoids the
        conversion entirely."""
        @fp.fpy(ctx=fp.FP32)
        def f(xs: list[fp.Real], i: fp.Real):
            return xs[i] * 2

        out = _emit(f, [ListType(_R32, 8), _INT])
        assert '2.0' in out
        assert '.to(' not in out

    def test_everything_emitted_is_parseable_python(self):
        """The emitter's output has to lex, whatever else it is."""
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
        abstract = {
            n for n, m in inspect.getmembers(Visitor, inspect.isfunction)
            if getattr(m, '__isabstractmethod__', False)
        }
        assert abstract, 'expected Visitor to declare abstract methods'
        assert not (abstract - set(dir(_Emitter)))
        assert not getattr(_Emitter, '__abstractmethods__', frozenset())


class TestSelectOps:
    def test_max_propagates_nan(self):
        """FPy follows IEEE 754-2019 `maximum`, where a NaN operand
        propagates; Triton's default is `PropagateNan.NONE`, which is
        `maximumNumber` and returns the *other* operand."""
        @fp.fpy(ctx=fp.FP32)
        def f(x: fp.Real, y: fp.Real):
            return max(x, y)

        assert _emit(f, [_R32, _R32]) == (
            'return tl.maximum(x, y, propagate_nan=tl.PropagateNan.ALL)'
        )

    def test_min_too(self):
        @fp.fpy(ctx=fp.FP32)
        def f(x: fp.Real, y: fp.Real):
            return min(x, y)

        assert _emit(f, [_R32, _R32]) == (
            'return tl.minimum(x, y, propagate_nan=tl.PropagateNan.ALL)'
        )

    def test_an_nary_max_folds_pairwise(self):
        """Sound because `max` is associative *and* exact -- it returns an
        operand rather than computing one, so no grouping rounds
        differently."""
        @fp.fpy(ctx=fp.FP32)
        def f(x: fp.Real, y: fp.Real, z: fp.Real):
            return max(x, y, z)

        assert _emit(f, [_R32, _R32, _R32]).count('tl.maximum(') == 2


class TestDestructuring:
    """`a, b = (x, y)`.  Triton has no tuple *value* to bind, so the binding
    comes apart into one assignment per element."""

    def test_a_literal_tuple_comes_apart(self):
        @fp.fpy(ctx=fp.FP32)
        def f(x: fp.Real, y: fp.Real):
            a, b = (x + y, x - y)
            return a * b

        assert _emit(f, [_R32, _R32]) == (
            'a = (x + y)\n'
            'b = (x - y)\n'
            'return (a * b)'
        )

    def test_a_swap_goes_through_temporaries(self):
        """The elements are simultaneous and sequential assignment is not:
        `a = b; b = a` turns a swap into a copy."""
        @fp.fpy(ctx=fp.FP32)
        def f(x: fp.Real, y: fp.Real):
            a, b = (x, y)
            a, b = (b, a)
            return a - b

        out = _emit(f, [_R32, _R32])
        assert out == (
            'a = x\n'
            'b = y\n'
            '_t0 = b\n'
            '_t1 = a\n'
            'a = _t0\n'
            'b = _t1\n'
            'return (a - b)'
        )

    def test_no_temporaries_when_nothing_is_read_back(self):
        @fp.fpy(ctx=fp.FP32)
        def f(x: fp.Real, y: fp.Real):
            a, b = (x, y)
            return a + b

        assert '_t' not in _emit(f, [_R32, _R32])

    def test_an_underscore_binds_nothing(self):
        @fp.fpy(ctx=fp.FP32)
        def f(x: fp.Real, y: fp.Real):
            _, b = (x, y)
            return b

        out = _emit(f, [_R32, _R32])
        assert out == 'b = y\nreturn b'

    def test_binding_a_tuple_to_a_name_is_refused_first(self):
        """Destructuring a *name* never arises: binding the tuple fails
        first, because a tuple has no storage to hold it.  So the only
        destructuring that reaches this emitter is of a literal."""
        @fp.fpy(ctx=fp.FP32)
        def f(x: fp.Real, y: fp.Real):
            t = (x, y)
            a, b = t
            return a + b

        with pytest.raises(TritonEmitError, match='tuple has no Triton storage'):
            _emit(f, [_R32, _R32])


class TestLoopBinding:
    """`tl.static_range` yields an *index*.  That is what the loop variable
    means only when the iterable is a `range`."""

    def test_a_range_binds_the_index(self):
        @fp.fpy(ctx=fp.FP32)
        def f(xs: list[fp.Real]):
            acc = fp.round(0)
            for i in range(4):
                acc = acc + xs[i]
            return acc

        assert 'for i in tl.static_range(4):' in _emit(f, [ListType(_R32, 4)])

    def test_iterating_a_list_is_refused(self):
        """`for x in xs` binds an *element*, and emitting the index in its
        place maxes against 0, 1, 2 rather than against the values -- a silent
        miscompile, and the reason a proven trip count is not on its own
        enough to emit a loop."""
        @fp.fpy(ctx=fp.FP32)
        def f(xs: list[fp.Real]):
            acc = fp.round(0)
            for x in xs:
                acc = acc + x
            return acc

        with pytest.raises(TritonEmitError, match='binds an element'):
            _emit(f, [ListType(_R32, 4)])

    def test_iterating_a_slice_is_refused(self):
        @fp.fpy(ctx=fp.FP32)
        def f(xs: list[fp.Real]):
            acc = fp.round(0)
            for x in xs[1:]:
                acc = acc + x
            return acc

        with pytest.raises(TritonEmitError, match='binds an element'):
            _emit(f, [ListType(_R32, 4)])


class TestPredicates:
    """Triton has no `isnan`, `isinf` or `isfinite`, so each is written from
    comparisons.  All three are exact -- they read the value rather than
    computing one -- and each turns on a NaN comparing unequal to everything."""

    def test_isnan(self):
        @fp.fpy(ctx=fp.FP32)
        def f(x: fp.Real):
            return fp.isnan(x)

        assert _emit(f, [_R32]) == 'return (x != x)'

    def test_isinf_is_false_for_nan(self):
        """`|nan| == inf` is false because the comparison is, which is the
        answer FPy gives."""
        @fp.fpy(ctx=fp.FP32)
        def f(x: fp.Real):
            return fp.isinf(x)

        assert _emit(f, [_R32]) == "return (tl.abs(x) == float('inf'))"

    def test_isfinite_is_false_for_nan(self):
        @fp.fpy(ctx=fp.FP32)
        def f(x: fp.Real):
            return fp.isfinite(x)

        assert _emit(f, [_R32]) == "return (tl.abs(x) < float('inf'))"

    def test_signbit_reads_the_sign_bit(self):
        """No *float* comparison separates `-0.0` from `0.0`, so this reads
        the bit: a bitcast to the same-width integer, tested for negative."""
        @fp.fpy(ctx=fp.FP32)
        def f(x: fp.Real):
            return fp.signbit(x)

        assert _emit(f, [_R32]) == 'return (x.to(tl.int32, bitcast=True) < 0)'

    def test_signbit_of_an_integer_is_direct(self):
        """There is no `-0` in an integer, so nothing to bitcast around."""
        @fp.fpy(ctx=fp.INTEGER)
        def f(x: fp.Real):
            return fp.signbit(x)

        assert _emit(f, [RealType(fp.INTEGER)]) == 'return (x < 0)'

    def test_logb_of_an_integer_is_refused(self):
        """There is no exponent field to read.  The result is guarded too --
        `logb(0)` is `-inf`, which an integer cannot hold -- but the operand
        is what an integer context reaches first."""
        @fp.fpy(ctx=fp.INTEGER)
        def f(x: fp.Real):
            return fp.logb(x)

        with pytest.raises(TritonEmitError, match='has none to read'):
            _emit(f, [RealType(fp.INTEGER)])


class TestLiteralLists:
    """A list literal stops existing: it becomes that many values.  A list the
    program fills is a tile instead (`TestLanes`)."""

    def test_a_literal_list(self):
        @fp.fpy(ctx=fp.FP32)
        def f(x: fp.Real, y: fp.Real):
            t = [x, y, x + y]
            return t[2] - t[0]

        assert _emit(f, [_R32, _R32]) == (
            't_0 = x\n'
            't_1 = y\n'
            't_2 = (x + y)\n'
            'return (t_2 - t_0)'
        )

    def test_a_slice_of_memory_is_an_address_not_values(self):
        """A slice of something in memory is the same list at an offset."""
        @fp.fpy(ctx=fp.FP32)
        def f(A: list[fp.Real]):
            w = A[1:3]
            return w[0] + w[1]

        out = _emit(f, [ListType(_R32, 4)])
        assert out == ('__t0 = tl.load(A_ptr + 1)\n__t1 = tl.load(A_ptr + 1 + 1)\n'
                       'return (__t0 + __t1)')

    def test_a_slice_survives_a_dynamic_index(self):
        """The loop stays rolled and the index need not be constant, because
        there is an address to add it to."""
        @fp.fpy(ctx=fp.FP32)
        def f(A: list[fp.Real]):
            w = A[1:5]
            s = fp.round(0)
            for j in range(4):
                s = s + w[j]
            return s

        out = _emit(f, [ListType(_R32, 8)])
        assert 'for j in tl.static_range(4):' in out
        assert 'tl.load(A_ptr + j + 1)' in out

    def test_len_of_a_literal_list(self):
        @fp.fpy(ctx=fp.INTEGER)
        def f(x: fp.Real):
            p = [x, x, x]
            return fp.round(len(p))

        assert 'return 3' in _emit(f, [RealType(fp.INTEGER)], ctx=fp.INTEGER)

    def test_a_dynamic_index_selects(self):
        """There is no addressable local array, so the index picks among the
        elements; like a load, it is taken to be in range."""
        @fp.fpy(ctx=fp.FP32)
        def f(x: fp.Real, y: fp.Real, i: fp.Real):
            p = [x, y, x + y]
            return p[i]

        src = _emit(f, [_R32, _R32, _INT])
        assert 'tl.where(i == 0, p_0, tl.where(i == 1, p_1, p_2))' in src

    def test_aliasing_a_sequence_copies_its_elements(self):
        """There is no sequence to point at, so `q = p` re-binds the values."""
        @fp.fpy(ctx=fp.FP32)
        def f(x: fp.Real, y: fp.Real):
            p = [x * x, y]
            q = p
            return q[0]

        out = _emit(f, [_R32, _R32])
        assert 'q_0 = p_0' in out
        assert 'return q_0' in out

    def test_an_out_of_range_constant_index_is_refused(self):
        @fp.fpy(ctx=fp.FP32)
        def f(x: fp.Real):
            t = [x, x]
            return t[5]

        with pytest.raises(TritonEmitError, match='outside a sequence'):
            _emit(f, [_R32])


class TestBranch:
    """An `if` that is not a tile guard is flattened: each arm runs on every
    lane under its mask, and what it merges is chosen by `tl.where`."""

    def test_two_arms_merge_by_where(self):
        @fp.fpy(ctx=fp.FP32)
        def f(x: fp.Real):
            if x < 0:
                y = -x
            else:
                y = x * 2
            return y

        assert _emit(f, [_R32]) == (
            '__t0 = (x < 0)\n'
            'y = (-x)\n'
            '__t1 = y\n'
            'y = (x * 2.0)\n'
            'y = tl.where(__t0, __t1, y)\n'
            'return y'
        )

    def test_one_arm_merges_with_the_value_before(self):
        @fp.fpy(ctx=fp.FP32)
        def f(x: fp.Real):
            y = x
            if x < 0:
                y = -x
            return y

        out = _emit(f, [_R32]).splitlines()
        assert out[-2] == 'y = tl.where(__t0, __t2, y)'
        # restored before the merge reads it
        assert out[-3] == 'y = __t1'

    def test_the_second_arm_reads_what_the_first_overwrote(self):
        @fp.fpy(ctx=fp.FP32)
        def f(x: fp.Real, z: fp.Real):
            y = x
            if x < 0:
                y = z
                w = y * 2
            else:
                w = y * 3
            return w

        out = _emit(f, [_R32, _R32])
        assert out.index('y = __t1') < out.index('w = (y * 3.0)')

    def test_nested_branches_compose_the_mask(self):
        @fp.fpy(ctx=fp.FP32)
        def f(xs: list[fp.Real], out: list[fp.Real], j: fp.Real, n: fp.Real):
            if j < n:
                if xs[j] < 0:
                    out[j] = xs[j]
            return out

        out = _emit(
            f, [ListType(_R32, 8), ListType(_R32, 8), _INT, _INT], guard=True)
        assert 'mask=((j < n) & __t1)' in out

    def test_a_store_in_an_arm_carries_its_mask(self):
        @fp.fpy(ctx=fp.FP32)
        def f(out: list[fp.Real], x: fp.Real):
            if x < 0:
                out[0] = x
            else:
                out[0] = -x
            return out

        out = _emit(f, [ListType(_R32, 8), _R32])
        assert 'tl.store(out_ptr + 0, x, mask=__t0)' in out
        assert 'tl.store(out_ptr + 0, (-x), mask=(~__t0))' in out
        # the list behind the pointer merges by its stores alone
        assert 'tl.where' not in out

    def test_a_load_in_an_arm_carries_its_mask(self):
        @fp.fpy(ctx=fp.FP32)
        def f(xs: list[fp.Real], i: fp.Real):
            y = 0
            if i < 8:
                y = xs[i]
            return y

        assert 'tl.load(xs_ptr + i, mask=__t0, other=0.0)' in _emit(
            f, [ListType(_R32, 8), _INT])

    def test_a_narrower_arm_is_widened_where_it_is_assigned(self):
        """Into the storage of the name's class, as a C++ declaration would,
        so the merge itself needs no cast."""
        @fp.fpy(ctx=fp.REAL)
        def f(x: fp.Real):
            if x < 0:
                with FP16:
                    y = fp.round(x)
            else:
                y = x
            return y

        out = _emit(f, [_R32], ctx=fp.REAL)
        assert 'y = x.to(tl.float16).to(tl.float32)' in out
        assert 'y = tl.where(__t0, __t1, y)' in out

    def test_a_return_in_an_arm_is_refused(self):
        @fp.fpy(ctx=fp.FP32)
        def f(x: fp.Real):
            if x < 0:
                return -x
            return x

        with pytest.raises(TritonEmitError, match='`return` in a branch'):
            _emit(f, [_R32])

    def test_a_literal_list_chosen_by_a_branch_is_refused(self):
        @fp.fpy(ctx=fp.FP32)
        def f(x: fp.Real):
            ys = [x, x]
            if x < 0:
                ys[0] = -x
            return ys[0]

        with pytest.raises(TritonEmitError, match='list chosen by a branch'):
            _emit(f, [_R32])


@fp.fpy(ctx=fp.REAL)
def _narrow(x: fp.Real):
    with FP16:
        y = fp.cast(x)
    return y


class TestExactCast:
    """`fp.cast` asserts its result is exact.  The cpp backend checks that at
    runtime; a kernel cannot, so it is treated as an `assert` is."""

    def test_an_unproven_cast_is_refused(self):
        with pytest.raises(TritonEmitError, match='asserts its result is exact'):
            _emit(_narrow, [_R32], ctx=fp.REAL)

    def test_a_proven_cast_emits_nothing(self):
        assert _emit(_narrow, [RealType(FP16)], ctx=fp.REAL) == (
            'y = x\nreturn y')

    def test_dropping_asserts_drops_the_check(self):
        assert 'x.to(tl.float16)' in _emit(_narrow, [_R32], fp.REAL, drop_asserts=True)


class TestLiteralSpelling:
    def test_an_integer_past_int64_is_spelled_as_a_float(self):
        """Triton refuses an integer literal no `int64` holds."""
        @fp.fpy(ctx=fp.FP64)
        def f(x: fp.Real):
            return x < 340282366920938463463374607431768211456

        assert '3.402823669209385e+38' in _emit(f, [RealType(fp.FP64)])

    def test_a_negative_zero_keeps_its_sign(self):
        """Triton folds a `-0.0` constant to `+0.0`."""
        @fp.fpy(ctx=fp.FP32)
        def f(x: fp.Real):
            return -0.0 if x < 0 else x

        assert '(-tl.zeros((), ' in _emit(f, [_R32])


def test_a_float_held_exponent_is_cast_for_ldexp():
    """`libdevice.ldexp` takes an `int32`; a `logb` result is held as a float
    for its specials, and every finite value of it is an integer."""
    @fp.fpy(ctx=fp.REAL)
    def f(x: fp.Real):
        e = fp.logb(x)
        return 2 ** -e * x

    assert '.to(tl.int32))' in _emit(f, [RealType(fp.FP64)], ctx=fp.REAL)


class TestLoopCarried:
    def test_a_range_keeps_its_start_and_step(self):
        """`tl.static_range` counts from zero, so the target is where the
        count lands: `range(0, 8, 4)` is 0 and 4, not 0 and 1."""
        @fp.fpy(ctx=fp.FP32)
        def f(xs: list[fp.Real]):
            acc = fp.round(0)
            for i in range(2, 8, 4):
                acc = acc + xs[i]
            return acc

        out = _emit(f, [ListType(_R32, 8)])
        assert 'in tl.static_range(2):' in out
        assert 'i = 2 + __t0 * 4' in out

    def test_a_carried_value_is_held_in_its_class(self):
        """Triton declares nothing, so a value narrower than the phi joining
        it with what each iteration leaves is widened where it is assigned."""
        @fp.fpy(ctx=fp.REAL)
        def f(c: fp.Real):
            d = c
            for _ in range(4):
                with fp.REAL:
                    d = 2 ** -64 * d
            return d

        out = _emit(f, [_R32], ctx=fp.REAL)
        assert 'd = c.to(tl.float64)' in out


def test_an_ldexp_scales_in_the_products_storage():
    """`ldexp` computes in its argument's type, and `2 ** -200 * x` needs a
    wider one than `x`'s."""
    @fp.fpy(ctx=fp.REAL)
    def f(x: fp.Real):
        return 2 ** -200 * x

    assert 'libdevice.ldexp(x.to(tl.float64), ' in _emit(f, [_R32], ctx=fp.REAL)


def test_a_lane_invariant_address_is_one_scalar_load():
    """Under the tile's guard alone, an address the same on every lane is one
    scalar load; a lane's own is a vector load."""
    @fp.fpy(ctx=fp.FP32)
    def f(xs: list[fp.Real], out: list[fp.Real], BLOCK: fp.Real):
        for i in range(len(xs)):
            out[i] = xs[i] + xs[0]
        return out

    src = _flat(f).source
    assert 'tl.zeros_like(' not in src
    assert 'tl.load(xs_ptr + 0)' in src
    assert 'tl.load(xs_ptr + i, mask=' in src


def test_a_lane_invariant_address_under_a_branch_is_broadcast():
    """A branch can guard an address on every lane at once, so under one the
    load keeps the branch's mask, broadcast to the tile."""
    @fp.fpy(ctx=fp.FP32)
    def f(xs: list[fp.Real], out: list[fp.Real], BLOCK: fp.Real):
        for i in range(len(xs)):
            if xs[i] > 0:
                y = xs[0]
            else:
                y = xs[i]
            out[i] = y
        return out

    assert 'tl.load(xs_ptr + 0 + tl.zeros_like(j), mask=' in _flat(f).source


class TestTryWiden:
    """Under `REAL`, any width holding the operands and the exact result will
    do; the result's own need not hold an operand."""

    def test_a_result_narrower_than_an_operand(self):
        """`z * 0` is ±0, stored `f16`; `z` is a 22-bit `f32`."""
        @fp.fpy(ctx=fp.REAL)
        def f(a: fp.Real, b: fp.Real):
            with fp.FP32:
                z = a * b
            with fp.REAL:
                y = z * 0
            return y

        out = _emit(f, [RealType(FP16), RealType(FP16)], ctx=fp.REAL)
        assert '(z * 0.0).to(tl.float16)' in out

    def test_a_literal_wider_than_the_result(self):
        """The branch bounds `2^139 * t` to `f32`, where the literal needs
        `f64`: computed at `f64`, then narrowed exactly."""
        @fp.fpy(ctx=fp.REAL)
        def f(t: fp.Real):
            y = t
            if abs(t) < fp.rational(1, 85070591730234615865843651857942052864):
                with fp.REAL:
                    y = 696898287454081973172991196020261297061888 * t
            return y

        out = _emit(f, [_R32], ctx=fp.REAL)
        assert 'tl.float64' in out and '.to(tl.float32)' in out


@fp.fpy(ctx=fp.REAL)
def _exact_sum(x: fp.Real, y: fp.Real):
    with fp.REAL:
        s = sum([x, y])
    with fp.FP64:
        t = fp.round(s)
    return t


def test_an_exact_sum_no_storage_holds_is_refused():
    """`sum([x, y])` over two `f64`s needs about 2100 bits exactly.  A name
    has one storage, as a declaration has one type, so it is refused rather
    than computed in the operands' and rounded."""
    with pytest.raises(TritonEmitError, match='no storage holds its elements'):
        _emit(_exact_sum, [RealType(fp.FP64)] * 2, fp.REAL)


def test_an_exact_sum_folds_in_its_own_storage():
    """Two `f16`s sum exactly in about 41 bits: every partial sum is in the
    sum's `f64`, and folding in the elements' own `f16` would round."""
    out = _emit(_exact_sum, [RealType(FP16)] * 2, fp.REAL)
    assert '(x.to(tl.float64) + y.to(tl.float64))' in out


def test_an_unproven_length_is_a_parameter():
    """Its offsets, loops and masks read it, and the launcher is told which
    argument's dimension it is."""
    @fp.fpy(ctx=fp.FP32)
    def f(xss: list[list[fp.Real]], out: list[list[fp.Real]], BLOCK: fp.Real):
        for i in range(len(out)):
            row = out[i]
            for j in range(len(row)):
                row[j] = xss[i][j] * 2
        return out

    m, n = NamedId('m'), NamedId('n')
    t = ListType(ListType(_R32, n), m)
    src = TritonCompiler(drop_asserts=True).compile(
        f, ctx=fp.FP32, arg_types=[t, t, _INT],
    )
    assert src.params[-2:] == ('xss_n0', 'xss_n1')
    assert src.sizes == (('xss_n0', 0, 0), ('xss_n1', 0, 1))
    assert (src.grid_extent, src.grid_outer) == ('xss_n1', 'xss_n0')
    assert 'i = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]' in src.source
    assert 'xss_ptr + i * xss_n1 + j' in src.source


def test_a_load_is_masked_along_the_axes_its_address_varies():
    """Under a tile of rows, a load of `xs[i]` is a column, masked by the
    rows' guard alone; `ys[j]` a row, by the columns'."""
    @fp.fpy(ctx=fp.FP32)
    def f(xs: list[fp.Real], ys: list[fp.Real], out: list[list[fp.Real]], BLOCK: fp.Real):
        for i in range(len(out)):
            row = out[i]
            for j in range(len(row)):
                row[j] = xs[i] * ys[j]
        return out

    m, n = NamedId('m'), NamedId('n')
    src = TritonCompiler(drop_asserts=True).compile(f, ctx=fp.FP32, arg_types=[
        ListType(_R32, m), ListType(_R32, n), ListType(ListType(_R32, n), m), _INT])
    rows = re.search(r'(__t\d+) = \(i < \w+\)', src.source)
    assert rows is not None
    assert f'tl.load(xs_ptr + i, mask={rows[1]}, other=0.0)' in src.source
    assert re.search(r'tl\.load\(ys_ptr \+ j, mask=\(j\d+ < t\d+\), other=0\.0\)', src.source)


class TestLanes:
    """Under `lanes`, a loop over a list's elements is one `[rows, lanes]`
    operation, not one per element."""

    @staticmethod
    def _source(n: int) -> str:
        @fp.fpy(ctx=fp.REAL)
        def f(xss: list[list[fp.Real]], out: list[list[fp.Real]], BLOCK: fp.Real):
            for j in range(len(out)):
                xs = xss[j]
                row = out[j]
                with fp.FP32:
                    ys = [x * 2 for x in xs]
                for k in range(len(row)):
                    row[k] = ys[k]
            return out

        return _rows(f, n, n).source

    def test_one_load_and_one_store_across_the_lanes(self):
        src = self._source(8)
        assert src.count('tl.load(') == 1
        assert src.count('tl.store(') == 1
        assert 'tl.arange(0, 8)[None, :]' in src
        assert 'xss_ptr + j[:, None] * 8 + ' in src

    def test_a_tail_is_one_more_per_element(self):
        src = self._source(9)
        assert src.count('tl.load(') == 2
        assert 'ys_t0 = ' in src


def test_a_rounding_sum_over_wider_elements_is_refused():
    """Each partial sum rounds under `FP16`, which an add in the elements'
    wider storage does not do."""
    @fp.fpy(ctx=fp.REAL)
    def f(x: fp.Real, y: fp.Real):
        ys = [x * 2, y * 2]
        with fp.FP16:
            return sum(ys)

    with pytest.raises(TritonEmitError, match='rounds each partial sum'):
        _emit(f, [RealType(FP16)] * 2)


def test_an_unaligned_slice_of_a_tile_is_refused():
    """A slice becomes a row of the reshaped tile only at a multiple of its
    width; anything else would need a gather."""
    @fp.fpy(ctx=fp.REAL)
    def f(xs: list[fp.Real], out: list[fp.Real], BLOCK: fp.Real):
        ys = [x * 2 for x in xs]
        zs = ys[1:3]
        out[0] = zs[0]
        return out

    with pytest.raises(TritonEmitError, match='starts at a multiple'):
        _flat(f, n_out=1, ctx=fp.REAL)


def test_a_slice_into_the_tail_is_refused():
    """`ys[4:6]` of six is past the tile's four lanes, in its tail."""
    @fp.fpy(ctx=fp.REAL)
    def f(xss: list[list[fp.Real]], out: list[list[fp.Real]], BLOCK: fp.Real):
        for j in range(len(out)):
            xs = xss[j]
            row = out[j]
            ys = [x * 2 for x in xs]
            ws = ys[4:6]
            row[0] = max(ws)
        return out

    with pytest.raises(TritonEmitError, match='proven to lie in its first 4'):
        _rows(f, 6, 1)


class TestGather:
    """A read of a tile at a lane-varying index."""

    def test_an_interleaved_part_splits(self):
        """`a + s * k` with `s` a power of two over the whole tile is a
        reshape and a split per bit of `a`."""
        from .programs import interleaved
        src = _rows(interleaved, 8, 4, elt=FP16).source
        assert src.count('tl.split(tl.reshape(') == 2
        assert 'tl.sum(' not in src

    def test_any_other_index_is_a_one_hot_sum(self):
        from .programs import reversed_row
        src = _rows(reversed_row, 4, 4, elt=FP16).source
        assert 'tl.sum(tl.where((tl.arange(0, 4)[None, None, :] == ' in src

    def test_an_index_past_the_tile_is_refused(self):
        @fp.fpy(ctx=fp.REAL)
        def f(xss: list[list[fp.Real]], out: list[list[fp.Real]], BLOCK: fp.Real):
            for j in range(len(out)):
                xs = xss[j]
                row = out[j]
                ys = [x * 2 for x in xs]
                for k in range(4):
                    row[k] = ys[k + 1]
            return out

        with pytest.raises(TritonEmitError, match='not proven within its 4'):
            _rows(f, 4, 4, elt=FP16)


def test_a_list_rebound_under_a_branch_is_refused():
    """A tile merges by masked writes; a new one under the branch would
    replace it on every row."""
    @fp.fpy(ctx=fp.REAL)
    def f(xss: list[list[fp.Real]], out: list[list[fp.Real]], BLOCK: fp.Real):
        for j in range(len(out)):
            xs = xss[j]
            row = out[j]
            ys = [x * 2 for x in xs]
            if xs[0] > 0:
                ys = [x * 3 for x in xs]
            for k in range(len(row)):
                row[k] = ys[k]
        return out

    with pytest.raises(TritonEmitError, match='rebound under a branch'):
        _rows(f, 4, 4)


def test_an_exact_sum_over_a_tile_no_storage_holds_is_refused():
    """Folding in the elements' `f64` rounds each partial sum."""
    @fp.fpy(ctx=fp.REAL)
    def f(xss: list[list[fp.Real]], out: list[list[fp.Real]], BLOCK: fp.Real):
        for j in range(len(out)):
            xs = xss[j]
            row = out[j]
            ys = [x * 1 for x in xs]
            row[0] = 1.0 if sum(ys) > 0 else 0.0
        return out

    with pytest.raises(TritonEmitError, match='no storage holds its elements'):
        _rows(f, 3, 1, elt=fp.FP64, out=fp.FP64)


def test_a_store_that_rounds_is_refused():
    """`tl.store` converts to the pointer's `f16`, which rounds an `f32`."""
    @fp.fpy(ctx=fp.REAL)
    def f(xss: list[list[fp.Real]], out: list[list[fp.Real]], BLOCK: fp.Real):
        for j in range(len(out)):
            xs = xss[j]
            row = out[j]
            with fp.FP32:
                row[0] = xs[0] * 1.5
        return out

    with pytest.raises(TritonEmitError, match='would round'):
        _rows(f, 1, 1, out=FP16)


def test_a_store_that_overflows_is_refused():
    """Every `f32` value times `2 ** 200` has `f32`'s precision but not its
    range, so `tl.store` would overflow it."""
    @fp.fpy(ctx=fp.REAL)
    def f(xss: list[list[fp.Real]], out: list[list[fp.Real]], BLOCK: fp.Real):
        for j in range(len(out)):
            xs = xss[j]
            row = out[j]
            row[0] = xs[0] * 2 ** 200
        return out

    with pytest.raises(TritonEmitError, match='would round'):
        _rows(f, 1, 1)


def test_a_reduction_over_a_row_in_memory_is_refused():
    """A row of an argument is an address, not elements to fold."""
    @fp.fpy(ctx=fp.REAL)
    def f(xss: list[list[fp.Real]], out: list[list[fp.Real]], BLOCK: fp.Real):
        for j in range(len(out)):
            xs = xss[j]
            row = out[j]
            with fp.FP32:
                row[0] = sum(xs)
        return out

    with pytest.raises(TritonEmitError, match='folds over a list held'):
        _rows(f, 4, 1)


@pytest.mark.parametrize('rm, halves', [
    (fp.RM.RTZ, True), (fp.RM.RNE, True), (fp.RM.RNA, True),
    (fp.RM.RTN, False), (fp.RM.RTP, False),
])
def test_a_scale_in_stays_in_its_operands_storage(rm: fp.RM, halves: bool) -> None:
    """Where the round sends what underflows to zero, the scale-in is two
    exact multiplies in `fp32`; floor and ceil do not, so it stays in
    `fp64`."""
    from .programs import aligned_sum
    src = TritonCompiler(drop_asserts=True, unfold=TritonCompiler.UnfoldMode.ROUNDINGS).compile(
        aligned_sum(rm), ctx=fp.REAL,
        arg_types=[ListType(ListType(_R32, 4), NamedId('n')),
                   ListType(RealType(fp.FP64), NamedId('n')), RealType(fp.INTEGER)])
    assert ('bitcast=True) * ((' in src.source) is halves
    assert ('x.to(tl.float64)' in src.source) is not halves


def test_the_scale_out_leaves_the_sum() -> None:
    """`HoistScale`: every term is scaled by one power of two, so the sum
    is, once."""
    from .programs import aligned_sum
    src = TritonCompiler(drop_asserts=True).compile(
        aligned_sum(fp.RM.RTZ), ctx=fp.REAL,
        arg_types=[ListType(ListType(_R32, 4), NamedId('n')),
                   ListType(RealType(fp.FP64), NamedId('n')), RealType(fp.INTEGER)])
    total = re.search(r'(__t\d+) = tl\.sum\(', src.source)
    assert total is not None
    assert f' * {total[1]})' in src.source


def test_a_power_of_two_is_its_bits() -> None:
    from .programs import pow2_finite, pow2_logb
    n = NamedId('n')
    src = TritonCompiler(drop_asserts=True).compile(pow2_finite, ctx=fp.REAL, arg_types=[
        ListType(RealType(fp.SINT8), n), ListType(RealType(fp.FP64), n), _INT])
    assert '<< 52).to(tl.float64, bitcast=True)' in src.source
    assert "float('nan')" not in src.source
    # an exponent that may be infinite or NaN has those selected apart
    src = TritonCompiler(drop_asserts=True).compile(pow2_logb, ctx=fp.REAL, arg_types=[
        ListType(RealType(FP16), n), ListType(_R32, n), _INT])
    assert "float('nan'), tl.where(" in src.source


_REUSE_ARGS = (ListType(_R32, 8), ListType(_R32, 8), _INT, _INT)


class TestLoadReuse:
    """A load serves a later read of the same element whose mask implies its
    own, until a store, a reassignment of a name in the address, or the edge
    of a loop body."""

    def test_under_a_narrower_mask(self):
        @fp.fpy(ctx=fp.FP32)
        def f(xs: list[fp.Real], out: list[fp.Real], j: fp.Real, n: fp.Real):
            if j < n:
                if xs[j] < 0:
                    out[j] = xs[j]
            return out

        assert _emit(f, list(_REUSE_ARGS), guard=True).count('tl.load(') == 1

    def test_not_across_a_store(self):
        @fp.fpy(ctx=fp.FP32)
        def f(xs: list[fp.Real], out: list[fp.Real], j: fp.Real, n: fp.Real):
            a = xs[j]
            xs[j] = a * 2
            out[j] = xs[j]
            return out

        assert _emit(f, list(_REUSE_ARGS)).count('tl.load(') == 2

    def test_not_after_its_index_is_reassigned(self):
        @fp.fpy(ctx=fp.FP32)
        def f(xs: list[fp.Real], out: list[fp.Real], j: fp.Real, n: fp.Real):
            t = j
            a = xs[t]
            t = n
            out[j] = a + xs[t]
            return out

        assert _emit(f, list(_REUSE_ARGS)).count('tl.load(') == 2

    def test_not_into_a_loop_body(self):
        @fp.fpy(ctx=fp.FP32)
        def f(xs: list[fp.Real], out: list[fp.Real], j: fp.Real, n: fp.Real):
            a = xs[0]
            for _k in range(4):
                a = a + xs[0]
            out[j] = a
            return out

        assert _emit(f, list(_REUSE_ARGS)).count('tl.load(') == 2


class TestLogb:
    """`logb` answers infinities and NaN only where they may occur, and a
    zero or subnormal only where no `max` clamps it away."""

    @staticmethod
    def _logb(f: Function, ctx: Context) -> str:
        from fpy2.utils import NamedId
        n = NamedId('n')
        src = TritonCompiler(drop_asserts=True).compile(
            f, ctx=fp.REAL,
            arg_types=[ListType(RealType(ctx), n), ListType(RealType(ctx), n), RealType(fp.INTEGER)])
        return src.source

    def test_clamped_at_the_least_exponent_it_is_the_field(self):
        from .programs import logb_clamped
        src = self._logb(logb_clamped(-14), FP16)
        assert '6.103515625e-05' not in src          # no subnormal rescale
        assert "float('-inf')" not in src            # no zero
        assert "float('nan')" in src                 # nothing proves `x` finite

    def test_clamped_below_the_least_exponent_it_is_not(self):
        from .programs import logb_clamped
        assert '6.103515625e-05' in self._logb(logb_clamped(-20), FP16)

    def test_proven_finite_it_needs_no_special(self):
        from .programs import logb_guarded
        src = self._logb(logb_guarded(-126), fp.FP32)
        assert "float('nan'), tl.where" not in src
        assert "== float('inf')" not in src

    def test_proven_finite_unclamped_it_keeps_zero_and_subnormals(self):
        from .programs import logb_finite
        src = self._logb(logb_finite, fp.FP32)
        assert "float('-inf')" in src and '1.1754943508222875e-38' in src
        assert "== float('inf')" not in src


class TestSkippedArm:
    """A long arm runs only where some live row takes it; a short one stays
    flattened, where the reduction would cost more than it saves."""

    @staticmethod
    def _src(f: Function) -> str:
        from fpy2.utils import NamedId
        n = NamedId('n')
        return TritonCompiler(drop_asserts=True).compile(
            f, ctx=fp.REAL,
            arg_types=[ListType(ListType(_R32, 4), n), ListType(_R32, n), RealType(fp.INTEGER)]).source

    def test_a_long_arm_is_skipped(self):
        from .programs import rare_arm
        assert 'if tl.max(' in self._src(rare_arm)

    def test_a_short_arm_is_not(self):
        from .programs import short_arm
        assert 'if tl.max(' not in self._src(short_arm)
