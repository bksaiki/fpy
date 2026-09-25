"""Expect tests: the exact Triton source the compiler emits.

These need no GPU; `test_launch.py` runs the kernels.  Pinning the whole text
matters: an fp16 product computed in fp16 and widened afterwards differs from
the right kernel by one cast's position, not by any substring worth grepping.
When one fails, read the diff: is the new text a better kernel or a broken one?
"""

import re

import pytest

import fpy2 as fp
from fpy2.backend.triton import TritonCompiler, TritonEmitError
from fpy2.types import ListType, RealType

from .programs import (
    K,
    any_all,
    batched_dot,
    logb,
    nan_inf,
    round_to_int,
    row_bound,
    scaled,
    signbit,
)

FP16 = fp.IEEEContext(5, 16)


@fp.fpy(ctx=fp.FP32)
def _scale(xs: list[fp.Real], out: list[fp.Real], BLOCK: fp.Real):
    """A map: every element independent, so the whole loop tiles."""
    for i in range(len(xs)):
        out[i] = xs[i] * xs[i]
    return out


_DOT = '''\
@triton.jit
def batched_dot(xss_ptr, yss_ptr, out_ptr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK
    j = i + tl.arange(0, BLOCK)
    r = j
    acc = 0.0
    for k in tl.static_range(8):
        __t0 = tl.load(xss_ptr + r * 8 + k, mask=(j < 4), other=0.0)
        __t1 = tl.load(yss_ptr + r * 8 + k, mask=(j < 4), other=0.0)
        acc = (acc + (__t0.to(tl.float32) * __t1.to(tl.float32)))
    tl.store(out_ptr + r, acc, mask=(j < 4))'''


def test_the_batched_dot_product():
    """Every design decision in this backend is visible in these ten lines:
    the grid and tile from `tile_loops`, the guard as a `mask=` rather than a
    branch, the `K` fold left sequential because its accumulation rounds, and
    both fp16 operands widened *before* the product rather than after.
    """
    src = TritonCompiler(drop_asserts=True).compile(
        batched_dot, ctx=fp.REAL, arg_types=[
            ListType(ListType(RealType(FP16), K), 4),
            ListType(ListType(RealType(FP16), K), 4),
            ListType(RealType(fp.FP32), 4),
            RealType(fp.INTEGER)])
    assert src.source == _DOT
    assert src.grid_extent == 4
    assert src.enable_fp_fusion is True


def test_a_map_over_one_dimension():
    """No fold, so nothing is refused and nothing stays sequential."""
    src = TritonCompiler(drop_asserts=True).compile(
        _scale, ctx=fp.FP32, arg_types=[
            ListType(RealType(fp.FP32), 6),
            ListType(RealType(fp.FP32), 6),
            RealType(fp.INTEGER)])
    assert src.source == '''\
@triton.jit
def _scale(xs_ptr, out_ptr, BLOCK: tl.constexpr):
    i6 = tl.program_id(0) * BLOCK
    j = i6 + tl.arange(0, BLOCK)
    i = j
    __t0 = tl.load(xs_ptr + i, mask=(j < 6), other=0.0)
    tl.store(out_ptr + i, (__t0 * __t0), mask=(j < 6))'''
    assert src.grid_extent == 6


def test_fp32_throughout_disables_fusion():
    """The same program at FP32: the product rounds, so contracting it into
    an `fma` is observable and the launcher must not."""
    src = TritonCompiler(drop_asserts=True).compile(
        batched_dot, ctx=fp.FP32, arg_types=[
            ListType(ListType(RealType(fp.FP32), K), 4),
            ListType(ListType(RealType(fp.FP32), K), 4),
            ListType(RealType(fp.FP32), 4),
            RealType(fp.INTEGER)])
    assert src.enable_fp_fusion is False
    # and with nothing to widen, no cast is emitted at all
    assert '.to(' not in src.source


@fp.fpy(ctx=fp.FP32)
def _reduce(xs: list[fp.Real], ys: list[fp.Real], out: list[fp.Real],
            BLOCK: fp.Real):
    """`max`/`min`/`sum` over a literal list."""
    for i in range(len(xs)):
        row = [xs[i], ys[i], 2.0]
        out[i] = max(row) - min(row) + sum(row)
    return out


@fp.fpy(ctx=fp.FP32)
def _empty(out: list[fp.Real], BLOCK: fp.Real):
    for i in range(len(out)):
        row = []
        out[i] = sum(row)
    return out


@fp.fpy(ctx=fp.FP32)
def _empty_max(out: list[fp.Real], BLOCK: fp.Real):
    for i in range(len(out)):
        row = []
        out[i] = max(row)
    return out


def test_reductions_over_a_literal_list_fold_its_elements():
    """`propagate_nan` carries FPy's IEEE 754-2019 `maximum`; Triton's default
    is `maximumNumber`, which returns the *other* operand.  `sum` folds left,
    as FPy's does, in the storage of its context."""
    src = TritonCompiler(drop_asserts=True).compile(
        _reduce, ctx=fp.FP32, arg_types=[
            ListType(RealType(fp.FP32), 8),
            ListType(RealType(fp.FP32), 8),
            ListType(RealType(fp.FP32), 8),
            RealType(fp.INTEGER)])
    assert src.source.splitlines()[-1] == (
        '    tl.store(out_ptr + i, ((tl.maximum(tl.maximum(row_0, row_1, '
        'propagate_nan=tl.PropagateNan.ALL), row_2, '
        'propagate_nan=tl.PropagateNan.ALL) - tl.minimum(tl.minimum(row_0, '
        'row_1, propagate_nan=tl.PropagateNan.ALL), row_2, '
        'propagate_nan=tl.PropagateNan.ALL)) + ((row_0.to(tl.float32) + '
        'row_1.to(tl.float32)) + row_2.to(tl.float32))), mask=(j < 8))')


def test_an_empty_sum_is_the_literal_zero():
    """Nothing is allocated: the length is proven zero, so the fold has a
    compile-time value and a scalar literal broadcasts over the tile."""
    src = TritonCompiler(drop_asserts=True).compile(
        _empty, ctx=fp.FP32,
        arg_types=[ListType(RealType(fp.FP32), 6), RealType(fp.INTEGER)])
    assert src.source.splitlines()[-1] == (
        '    tl.store(out_ptr + i, 0.0, mask=(j < 6))')


def test_an_empty_max_is_refused():
    """FPy raises `ValueError`, so there is no value to emit."""
    with pytest.raises(TritonEmitError, match='empty sequence has no value'):
        TritonCompiler(drop_asserts=True).compile(
            _empty_max, ctx=fp.FP32,
            arg_types=[ListType(RealType(fp.FP32), 6), RealType(fp.INTEGER)])


@fp.fpy(ctx=fp.FP32)
def _zipped(xs: list[fp.Real], ys: list[fp.Real], out: list[fp.Real],
            BLOCK: fp.Real):
    """A `zip` in the inner loop, which binds a tuple."""
    for i in range(len(out)):
        acc = fp.round(0)
        for x, y in zip(xs, ys):
            acc = acc + x * y
        out[i] = acc
    return out


def test_a_zip_is_eliminated_before_emission():
    """`ZipElim` is in the pipeline, not left to the caller.

    A `zip` binds a tuple and the emitter has no tuple storage, so without
    the pass this refuses with *a destructuring loop target has no Triton
    spelling*.  Rewritten to an indexed loop it is ordinary subscripts.
    """
    src = TritonCompiler(drop_asserts=True).compile(
        _zipped, ctx=fp.FP32, arg_types=[
            ListType(RealType(fp.FP32), K),
            ListType(RealType(fp.FP32), K),
            ListType(RealType(fp.FP32), 4),
            RealType(fp.INTEGER)])
    assert src.source == '''\
@triton.jit
def _zipped(xs_ptr, ys_ptr, out_ptr, BLOCK: tl.constexpr):
    i11 = tl.program_id(0) * BLOCK
    j = i11 + tl.arange(0, BLOCK)
    i = j
    acc = 0.0
    for _i in tl.static_range(8):
        __t0 = tl.load(xs_ptr + _i)
        x = __t0
        __t1 = tl.load(ys_ptr + _i)
        y = __t1
        acc = (acc + (x * y))
    tl.store(out_ptr + i, acc, mask=(j < 4))'''


def test_a_row_bound_to_a_name_flattens_to_one_load():
    """`xs = xss[r]` is not emitted: `xs[k]` is one load off `xss_ptr`, so the
    kernel is `_DOT`'s."""
    src = TritonCompiler(drop_asserts=True).compile(
        row_bound, ctx=fp.REAL, arg_types=[
            ListType(ListType(RealType(FP16), K), 4),
            ListType(ListType(RealType(FP16), K), 4),
            ListType(RealType(fp.FP32), 4),
            RealType(fp.INTEGER)])
    assert src.source == _DOT.replace('def batched_dot(', 'def row_bound(')


@fp.fpy(ctx=fp.FP32)
def _store_row(xss: list[list[fp.Real]], oss: list[list[fp.Real]],
               BLOCK: fp.Real):
    """A row bound to a name and then *stored* through."""
    for r in range(len(oss)):
        o = oss[r]
        xs = xss[r]
        for k in range(4):
            o[k] = xs[k] * 2.0
    return oss


def test_a_store_through_a_row_resolves_the_same_way():
    """Load and store resolve a named row alike: no `o_ptr` is emitted."""
    src = TritonCompiler(drop_asserts=True).compile(
        _store_row, ctx=fp.FP32, arg_types=[
            ListType(ListType(RealType(fp.FP32), 4), 4),
            ListType(ListType(RealType(fp.FP32), 4), 4),
            RealType(fp.INTEGER)])
    assert src.source.splitlines()[-2:] == [
        '    __t1 = tl.load(xss_ptr + r[:, None] * 4 + k, mask=__t0[:, None], other=0.0)',
        '    tl.store(oss_ptr + r[:, None] * 4 + k, (__t1 * 2.0), mask=__t0[:, None])']


@fp.fpy(ctx=fp.FP32)
def _triple(x: fp.Real) -> fp.Real:
    """A callee, so the comprehension below holds a call."""
    t = x * 3.0
    return t


@fp.fpy(ctx=fp.FP32)
def _comp_of_calls(xss: list[list[fp.Real]], out: list[fp.Real],
                   BLOCK: fp.Real):
    """A comprehension whose elements are calls."""
    for r in range(len(out)):
        ys = [_triple(xss[r][k]) for k in range(3)]
        out[r] = ys[0] + ys[1] + ys[2]
    return out


def test_a_comprehension_of_calls_compiles():
    """`FuncInline` cannot reach a call inside a comprehension, so without
    `StatementForm` ahead of it the call survives and the normal form is never
    reached.  As a loop, the call is a statement of its own."""
    src = TritonCompiler(drop_asserts=True).compile(
        _comp_of_calls, ctx=fp.FP32, arg_types=[
            ListType(ListType(RealType(fp.FP32), 3), 4),
            ListType(RealType(fp.FP32), 4),
            RealType(fp.INTEGER)])
    assert 'tl.store' in src.source
    # the callee's body once across a tile of two, once for the tail
    assert src.source.count('* 3.0') == 2


@fp.fpy(ctx=fp.FP32)
def _lazy_comp(xss: list[list[fp.Real]], out: list[fp.Real], BLOCK: fp.Real):
    """A comprehension in an `if` expression's arm."""
    for r in range(len(out)):
        out[r] = max([xss[r][k] for k in range(3)]) if xss[r][0] > 0.0 else 0.0
    return out


def test_a_comprehension_in_a_lazy_position_is_a_loop_too():
    """`Hoistable` turns the `IfExpr` into a statement, which gives the
    comprehension in its arm a slot, and the loop runs under the arm's mask."""
    src = TritonCompiler(drop_asserts=True).compile(
        _lazy_comp, ctx=fp.FP32, arg_types=[
            ListType(ListType(RealType(fp.FP32), 3), 4),
            ListType(RealType(fp.FP32), 4),
            RealType(fp.INTEGER)])
    assert re.search(r'tl\.max\(\w+, axis=1\)', src.source)
    assert src.source.count('tl.maximum') == 1, 'the tail folds in once'


@fp.fpy(ctx=fp.FP32)
def _empty_any(out: list[fp.Real], BLOCK: fp.Real):
    for i in range(len(out)):
        flags = []
        out[i] = 1.0 if any(flags) else 0.0
    return out


def test_any_and_all_fold_with_bitwise_connectives():
    """`&` / `|`, not Python's keywords, which short-circuit and return an
    operand rather than a tile -- the same reason `and` / `or` are spelled
    that way."""
    src = TritonCompiler(drop_asserts=True).compile(
        any_all, ctx=fp.FP32, arg_types=[
            ListType(ListType(RealType(fp.FP32), 3), 6),
            ListType(RealType(fp.FP32), 6),
            RealType(fp.INTEGER)])
    # across the tile's lanes, then its tail
    assert '& flags_t0)' in src.source
    assert '| flags_t0)' in src.source


def test_an_empty_any_is_its_identity():
    """`any([])` is False, as the interpreter gives.

    With `optimize`, `ConstFold` settles the whole expression before the
    emitter sees it, so the fold itself is checked with it off.
    """
    args = [ListType(RealType(fp.FP32), 4), RealType(fp.INTEGER)]
    raw = TritonCompiler(drop_asserts=True, optimize=False).compile(
        _empty_any, ctx=fp.FP32, arg_types=args)
    assert 'tl.where(False,' in raw.source
    folded = TritonCompiler(drop_asserts=True).compile(
        _empty_any, ctx=fp.FP32, arg_types=args)
    assert folded.source.splitlines()[-1].endswith(
        'tl.store(out_ptr + i, 0.0, mask=(j < 4))')


def test_signbit_reads_the_sign_bit():
    """No float comparison separates `-0.0` from `0.0`, so this bitcasts to
    the same-width integer and tests for negative."""
    src = TritonCompiler(drop_asserts=True).compile(
        signbit, ctx=fp.FP32, arg_types=[
            ListType(RealType(fp.FP32), 8),
            ListType(RealType(fp.FP32), 8),
            RealType(fp.INTEGER)])
    assert '.to(tl.int32, bitcast=True) < 0' in src.source


@fp.fpy(ctx=fp.FP32)
def _bool_merge(xs: list[fp.Real], out: list[fp.Real], BLOCK: fp.Real):
    """A merge of two *booleans*, which has no number format."""
    for i in range(len(out)):
        flag = xs[i] > 0.0
        if xs[i] > 1.0:
            flag = True
        out[i] = 1.0 if flag else 0.0
    return out


def test_a_boolean_merge_has_bool_storage():
    """The *type* says boolean; the *format* is only asked about reals.

    Format inference is defined over real-valued expressions, so reading
    "no format" as "must be a boolean" would infer a type from the absence
    of one -- and be wrong for a rounding context or any other foreign value,
    which have no format either.
    """
    src = TritonCompiler(drop_asserts=True).compile(
        _bool_merge, ctx=fp.FP32, arg_types=[
            ListType(RealType(fp.FP32), 4),
            ListType(RealType(fp.FP32), 4),
            RealType(fp.INTEGER)])
    assert 'tl.where' in src.source
    # the merged boolean needs no cast: `bool` is its own storage
    assert '.to(tl.int1' not in src.source


def test_nan_and_inf_are_literals():
    """Values, not operations, so they are spelled and retyped where used."""
    src = TritonCompiler(drop_asserts=True).compile(
        nan_inf, ctx=fp.FP32, arg_types=[
            ListType(RealType(fp.FP32), 4),
            ListType(RealType(fp.FP32), 4),
            RealType(fp.INTEGER)])
    assert "float('nan')" in src.source
    assert "float('inf')" in src.source
    # negated, not a second constant
    assert "(-tl.full((), float('inf'), dtype=tl.float32))" in src.source


@fp.fpy(ctx=fp.INTEGER)
def _int_nan(xs: list[fp.Real], out: list[fp.Real], BLOCK: fp.Real):
    for i in range(len(out)):
        out[i] = fp.nan()
    return out


def test_nan_is_refused_where_the_context_cannot_hold_one():
    """`fp.nan()` is `C.round(nan)`, and `fp.INTEGER.round(nan)` raises --
    as `1 / 0` there does.  A kernel cannot raise, so this is declined."""
    with pytest.raises(TritonEmitError, match='cannot hold one'):
        TritonCompiler(drop_asserts=True).compile(
            _int_nan, ctx=fp.INTEGER, arg_types=[
                ListType(RealType(fp.INTEGER), 4),
                ListType(RealType(fp.INTEGER), 4),
                RealType(fp.INTEGER)])


@fp.fpy(ctx=fp.FP32)
def _tile_reduction(xss: list[list[fp.Real]], out: list[fp.Real],
                    BLOCK: fp.Real):
    """A reduction over a *pointer-backed* row: must stay a rolled loop."""
    for r in range(len(out)):
        acc = fp.round(0)
        for k in range(8):
            acc = acc + xss[r][k]
        out[r] = acc
    return out


def test_a_reduction_over_memory_stays_rolled():
    """Unrolling is for lists of values, which have no iteration to perform.

    A loop over something in memory does, and this is the shape the backend
    is built around -- unrolling it would rewrite every kernel.
    """
    src = TritonCompiler(drop_asserts=True).compile(
        _tile_reduction, ctx=fp.FP32, arg_types=[
            ListType(ListType(RealType(fp.FP32), 8), 4),
            ListType(RealType(fp.FP32), 4),
            RealType(fp.INTEGER)])
    assert 'for k in tl.static_range(8):' in src.source
    assert src.source.count('tl.load(xss_ptr') == 1, 'one load, not eight'


def test_logb_reads_the_exponent_field():
    """No correctly-rounded primitive exists -- `tl.log2` is a
    transcendental, which the op table excludes -- so the exponent is read
    from the bits, which is exact."""
    src = TritonCompiler(drop_asserts=True).compile(
        logb, ctx=fp.FP32, arg_types=[
            ListType(RealType(fp.FP32), 4),
            ListType(RealType(fp.FP32), 4),
            RealType(fp.INTEGER)])
    assert '.to(tl.int32, bitcast=True)' in src.source
    assert '>> 23' in src.source and '& 255' in src.source
    # the three specials `logB` names
    assert "float('-inf')" in src.source   # at zero
    assert "float('inf')" in src.source    # at an infinity
    assert "float('nan')" in src.source    # at a NaN
    # a subnormal is scaled into range, not counted
    assert '16777216.0' in src.source


@fp.fpy(ctx=fp.FP16)
def _small_const(xs: list[fp.Real], out: list[fp.Real], BLOCK: fp.Real):
    """A small integer constant merged with a float value.

    `-132` is held as `int16`, which no `float16` holds *in general* -- and
    which this one holds exactly.
    """
    e_zero = -132
    for i in range(len(out)):
        out[i] = max(xs[i], e_zero)
    return out


def test_a_cast_asks_about_values_not_only_types():
    """`-132` is stored as `int16`, which `float16` does not nest, but its
    value fits, so it converts rather than being refused."""
    src = TritonCompiler(drop_asserts=True).compile(
        _small_const, ctx=fp.FP16, arg_types=[
            ListType(RealType(fp.FP16), 4),
            ListType(RealType(fp.FP16), 4),
            RealType(fp.INTEGER)])
    assert src.source.splitlines()[-1] == (
        '    tl.store(out_ptr + i, tl.maximum(__t0, -132.0, '
        'propagate_nan=tl.PropagateNan.ALL), mask=(j < 4))')


@fp.fpy(ctx=fp.FP32)
def _scaled_rounding(xs: list[fp.Real], ns: list[fp.Real],
                     out: list[fp.Real], BLOCK: fp.Real):
    """The same product where the context *rounds* it."""
    for i in range(len(out)):
        out[i] = (2 ** ns[i]) * xs[i]
    return out


_SCALE_ARGS = [
    ListType(RealType(fp.FP32), 4),
    # bounded, so the product has a storage: an unbounded exponent has none
    ListType(RealType(fp.SINT8), 4),
    ListType(RealType(fp.FP32), 4),
    RealType(fp.INTEGER),
]


def test_a_power_of_two_product_is_ldexp():
    """`ldexp` is `scaleB`: exact, where the product would round twice and
    rest on `exp2` returning the power exactly, which nothing guarantees."""
    src = TritonCompiler(drop_asserts=True).compile(
        scaled, ctx=fp.REAL, arg_types=_SCALE_ARGS)
    assert 'libdevice.ldexp(' in src.source
    assert '**' not in src.source


def test_ldexp_is_refused_where_the_context_would_round():
    """It stands in for the multiply only where the context would not round
    it -- otherwise it would skip a rounding the program asked for."""
    with pytest.raises(TritonEmitError, match='no signatures for op: Pow'):
        TritonCompiler(drop_asserts=True).compile(
            _scaled_rounding, ctx=fp.FP32, arg_types=_SCALE_ARGS)


def _round_to_int(mode: str) -> str:
    """The source of `round_to_int` under *mode*."""
    return TritonCompiler(drop_asserts=True).compile(
        round_to_int(fp.RoundingMode[mode]), ctx=fp.REAL, arg_types=[
            ListType(RealType(fp.FP32), 4),
            ListType(RealType(fp.FP32), 4),
            RealType(fp.INTEGER)]).source


@pytest.mark.parametrize('mode,fn', [
    ('RTZ', 'trunc'), ('RTN', 'floor'), ('RTP', 'ceil'),
    ('RNE', 'nearbyint'), ('RNA', 'round'),
])
def test_rounding_to_the_integers(mode, fn):
    """A fixed-point context at position zero holds the integers, so its
    round is a C integral rounding rather than a conversion.  `nearbyint` is
    ties-to-even and `round` is ties-away, which separates RNE from RNA."""
    assert f'libdevice.{fn}(' in _round_to_int(mode)


def test_a_mode_with_no_c_function_is_refused():
    """`RTO` rounds to odd, which no C function does -- refused rather than
    rounded differently."""
    with pytest.raises(TritonEmitError, match='not a hardware conversion'):
        _round_to_int('RTO')


def test_an_unbounded_exponent_is_not_narrowed_to_int32():
    """`ldexp` takes its exponent as an `int32`, which an unbounded one does
    not fit."""
    with pytest.raises(TritonEmitError, match='narrow tl.int64 to tl.int32'):
        TritonCompiler(drop_asserts=True).compile(
            scaled, ctx=fp.REAL, arg_types=[
                _SCALE_ARGS[0], ListType(RealType(fp.INTEGER), 4),
                *_SCALE_ARGS[2:]])
