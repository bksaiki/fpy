"""Expect tests: the exact Triton source the compiler emits.

These run anywhere -- no GPU, no torch -- and they are what CI can actually
check of this backend.  Execution is verified separately in `test_launch.py`,
which needs hardware and therefore skips; these do not, so an emitter change
shows up as a diff here rather than silently.

Pinning the *whole* text rather than fragments is the point.  A fragment
assertion passes while everything around it changes; the trap this backend
exists to avoid -- an fp16 product computed in fp16 and widened afterwards --
is a change of one character's position, not of any substring worth grepping
for.

When one of these fails, read the diff before updating it: the question is
whether the new text is a better kernel or a broken one.
"""

import pytest

import fpy2 as fp
from fpy2.backend.triton import TritonCompiler, TritonEmitError
from fpy2.types import ListType, RealType

FP16 = fp.IEEEContext(5, 16)
K = 8


@fp.fpy(ctx=fp.REAL)
def _batched_dot(xss: list[list[fp.Real]], yss: list[list[fp.Real]],
                 out: list[fp.Real], BLOCK: fp.Real):
    """The running example: FP16 in, exact products, FP32 accumulation."""
    for r in range(len(xss)):
        acc = fp.round(0)
        for k in range(K):
            with fp.FP32:
                acc = acc + xss[r][k] * yss[r][k]
        out[r] = acc
    return out


@fp.fpy(ctx=fp.FP32)
def _scale(xs: list[fp.Real], out: list[fp.Real], BLOCK: fp.Real):
    """A map: every element independent, so the whole loop tiles."""
    for i in range(len(xs)):
        out[i] = xs[i] * xs[i]
    return out


_DOT = '''\
@triton.jit
def _batched_dot(xss_ptr, yss_ptr, out_ptr, BLOCK: tl.constexpr):
    t7 = BLOCK
    t8 = 4
    i = tl.program_id(0) * BLOCK
    j = i + tl.arange(0, BLOCK)
    r = j
    acc = 0
    for k in tl.static_range(8):
        acc = (acc + (tl.load(xss_ptr + r * 8 + k, mask=(j < t8), other=0.0).to(tl.float32) * tl.load(yss_ptr + r * 8 + k, mask=(j < t8), other=0.0).to(tl.float32)))
    tl.store(out_ptr + r, acc, mask=(j < t8))'''


def test_the_batched_dot_product():
    """Every design decision in this backend is visible in these ten lines:
    the grid and tile from `tile_loops`, the guard as a `mask=` rather than a
    branch, the `K` fold left sequential because its accumulation rounds, and
    both fp16 operands widened *before* the product rather than after.
    """
    src = TritonCompiler(drop_asserts=True).compile(
        _batched_dot, ctx=fp.REAL, arg_types=[
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
    t4 = BLOCK
    t5 = 6
    i6 = tl.program_id(0) * BLOCK
    j = i6 + tl.arange(0, BLOCK)
    i = j
    tl.store(out_ptr + i, (tl.load(xs_ptr + i, mask=(j < t5), other=0.0) * tl.load(xs_ptr + i, mask=(j < t5), other=0.0)), mask=(j < t5))'''
    assert src.grid_extent == 6


def test_fp32_throughout_disables_fusion():
    """The same program at FP32: the product rounds, so contracting it into
    an `fma` is observable and the launcher must not.  Measured on hardware
    at 590 of 2000 inputs differing."""
    src = TritonCompiler(drop_asserts=True).compile(
        _batched_dot, ctx=fp.FP32, arg_types=[
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
    """`max`/`min`/`sum` over a scalarized row."""
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


def test_reductions_fold_over_the_scalarized_elements():
    """A sequence is gone by emission, so a reduction is a fold over values.

    `propagate_nan` carries FPy's IEEE 754-2019 `maximum`; Triton's default
    is `maximumNumber`, which returns the *other* operand.  `sum` folds left
    because FPy's does, and a tile reduction would reassociate.

    Verified against the interpreter on hardware at 8/8 inputs, NaN and
    infinite rows included.
    """
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
        'propagate_nan=tl.PropagateNan.ALL)) + ((row_0 + row_1) + row_2)), '
        'mask=(j < t7))')


def test_an_empty_sum_is_the_literal_zero():
    """Nothing is allocated: the length is proven zero, so the fold has a
    compile-time value and a scalar literal broadcasts over the tile."""
    src = TritonCompiler(drop_asserts=True).compile(
        _empty, ctx=fp.FP32,
        arg_types=[ListType(RealType(fp.FP32), 6), RealType(fp.INTEGER)])
    assert src.source.splitlines()[-1] == (
        '    tl.store(out_ptr + i, 0, mask=(j < t4))')


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
    t9 = BLOCK
    t10 = 4
    i11 = tl.program_id(0) * BLOCK
    j = i11 + tl.arange(0, BLOCK)
    i = j
    acc = 0
    for _i in tl.static_range(8):
        x = tl.load(xs_ptr + _i, mask=(j < t10), other=0.0)
        y = tl.load(ys_ptr + _i, mask=(j < t10), other=0.0)
        acc = (acc + (x * y))
    tl.store(out_ptr + i, acc, mask=(j < t10))'''


@fp.fpy(ctx=fp.REAL)
def _row_bound(xss: list[list[fp.Real]], yss: list[list[fp.Real]],
               out: list[fp.Real], BLOCK: fp.Real):
    """The rows bound to names, which is what inlining a call produces."""
    for r in range(len(out)):
        xs = xss[r]
        ys = yss[r]
        acc = fp.round(0)
        for k in range(K):
            with fp.FP32:
                acc = acc + xs[k] * ys[k]
        out[r] = acc
    return out


def test_a_row_bound_to_a_name_flattens_to_one_load():
    """`xs = xss[r]` names a sub-list, which Triton has no value for.

    The binding is not emitted: it records that `xs` is `xss` at a prefix, so
    `xs[k]` is one load off `xss_ptr` rather than a load of a load.  Emitting
    it as a scalar `tl.load` produced `tl.load(xs_ptr + k)`, and `xs_ptr` is
    not a parameter -- checked on the card as `NameError: xs_ptr is not
    defined` at JIT time.

    Byte-identical to `_DOT`, which writes `xss[r][k]` inline, up to the name
    the temporaries got.
    """
    src = TritonCompiler(drop_asserts=True).compile(
        _row_bound, ctx=fp.REAL, arg_types=[
            ListType(ListType(RealType(FP16), K), 4),
            ListType(ListType(RealType(FP16), K), 4),
            ListType(RealType(fp.FP32), 4),
            RealType(fp.INTEGER)])
    assert src.source == '''\
@triton.jit
def _row_bound(xss_ptr, yss_ptr, out_ptr, BLOCK: tl.constexpr):
    t9 = BLOCK
    t10 = 4
    i = tl.program_id(0) * BLOCK
    j = i + tl.arange(0, BLOCK)
    r = j
    acc = 0
    for k in tl.static_range(8):
        acc = (acc + (tl.load(xss_ptr + r * 8 + k, mask=(j < t10), other=0.0).to(tl.float32) * tl.load(yss_ptr + r * 8 + k, mask=(j < t10), other=0.0).to(tl.float32)))
    tl.store(out_ptr + r, acc, mask=(j < t10))'''


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
    """A store went straight to `{var}_ptr`, bypassing the row resolution.

    Load and store go through one resolver, so naming a row works the same
    either side of the assignment; before, the load beside it was already
    correct while the store emitted `o_ptr`.
    """
    src = TritonCompiler(drop_asserts=True).compile(
        _store_row, ctx=fp.FP32, arg_types=[
            ListType(ListType(RealType(fp.FP32), 4), 4),
            ListType(ListType(RealType(fp.FP32), 4), 4),
            RealType(fp.INTEGER)])
    assert src.source.splitlines()[-1] == (
        '        tl.store(oss_ptr + r * 4 + k, (tl.load(xss_ptr + r * 4 + k, '
        'mask=(j < t8), other=0.0) * 2.0), mask=(j < t8))')


@fp.fpy(ctx=fp.FP32)
def _scaled(x: fp.Real) -> fp.Real:
    """A callee, so the comprehension below holds a call."""
    t = x * 3.0
    return t


@fp.fpy(ctx=fp.FP32)
def _comp_of_calls(xss: list[list[fp.Real]], out: list[fp.Real],
                   BLOCK: fp.Real):
    """A comprehension whose elements are calls -- what `Scalarize` is for."""
    for r in range(len(out)):
        ys = [_scaled(xss[r][k]) for k in range(3)]
        out[r] = ys[0] + ys[1] + ys[2]
    return out


def test_a_comprehension_of_calls_compiles():
    """`FuncInline` cannot reach a call inside a comprehension, so without
    `Scalarize` ahead of it in the loop the call survives and the normal form
    is never reached.  Unrolled first, each call is its own statement."""
    src = TritonCompiler(drop_asserts=True).compile(
        _comp_of_calls, ctx=fp.FP32, arg_types=[
            ListType(ListType(RealType(fp.FP32), 3), 4),
            ListType(RealType(fp.FP32), 4),
            RealType(fp.INTEGER)])
    assert 'tl.store' in src.source
    # three loads, one per unrolled element, each scaled before the sum
    assert src.source.count('* 3.0') == 3


@fp.fpy(ctx=fp.FP32)
def _lazy_comp(xss: list[list[fp.Real]], out: list[fp.Real], BLOCK: fp.Real):
    """A comprehension in an `if` expression's arm."""
    for r in range(len(out)):
        out[r] = max([xss[r][k] for k in range(3)]) if xss[r][0] > 0.0 else 0.0
    return out


def test_the_emitter_still_expands_what_the_pass_declines():
    """`Scalarize` and the emitter are not two copies of one decision.

    The pass unrolls *early*, so `FuncInline` can reach a call inside a
    comprehension -- but it must not touch a lazily evaluated position, since
    hoisting out of an `IfExpr` arm would make it unconditional.  The emitter
    expands whatever is left, where eager evaluation is already the rule
    (`_emit_where`).

    So this program reaches the emitter holding a comprehension, and stops
    compiling if the emitter's expansion is removed as dead.
    """
    src = TritonCompiler(drop_asserts=True).compile(
        _lazy_comp, ctx=fp.FP32, arg_types=[
            ListType(ListType(RealType(fp.FP32), 3), 4),
            ListType(RealType(fp.FP32), 4),
            RealType(fp.INTEGER)])
    assert 'tl.where' in src.source
    assert src.source.count('tl.maximum') == 2, 'the comprehension was folded'


@fp.fpy(ctx=fp.FP32)
def _any_all(xss: list[list[fp.Real]], out: list[fp.Real], BLOCK: fp.Real):
    for r in range(len(out)):
        flags = [xss[r][k] > 0.0 for k in range(3)]
        out[r] = 2.0 if all(flags) else (1.0 if any(flags) else 0.0)
    return out


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
        _any_all, ctx=fp.FP32, arg_types=[
            ListType(ListType(RealType(fp.FP32), 3), 6),
            ListType(RealType(fp.FP32), 6),
            RealType(fp.INTEGER)])
    assert '(flags_0 & flags_1 & flags_2)' in src.source
    assert '(flags_0 | flags_1 | flags_2)' in src.source


def test_an_empty_any_is_its_identity():
    """`any([])` is False and `all([])` is True, as the interpreter gives.

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
        'tl.store(out_ptr + i, 0, mask=(j < t4))')


@fp.fpy(ctx=fp.FP32)
def _signbit(xs: list[fp.Real], out: list[fp.Real], BLOCK: fp.Real):
    for i in range(len(out)):
        out[i] = 1.0 if fp.signbit(xs[i]) else 0.0
    return out


def test_signbit_reads_the_sign_bit():
    """No float comparison separates `-0.0` from `0.0`, so this bitcasts to
    the same-width integer and tests for negative."""
    src = TritonCompiler(drop_asserts=True).compile(
        _signbit, ctx=fp.FP32, arg_types=[
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


@fp.fpy(ctx=fp.FP32)
def _nan_inf(xs: list[fp.Real], out: list[fp.Real], BLOCK: fp.Real):
    for i in range(len(out)):
        v = fp.nan() if xs[i] > 1.0 else (
            fp.inf() if xs[i] > 0.0 else -fp.inf())
        out[i] = v
    return out


def test_nan_and_inf_are_literals():
    """Values, not operations, so they are spelled and retyped where used."""
    src = TritonCompiler(drop_asserts=True).compile(
        _nan_inf, ctx=fp.FP32, arg_types=[
            ListType(RealType(fp.FP32), 4),
            ListType(RealType(fp.FP32), 4),
            RealType(fp.INTEGER)])
    assert "float('nan')" in src.source
    assert "float('inf')" in src.source
    # negated, not a second constant
    assert "-float('inf')" in src.source or "(-float('inf'))" in src.source


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


@fp.fpy(ctx=fp.FP32)
def _logb(xs: list[fp.Real], out: list[fp.Real], BLOCK: fp.Real):
    for i in range(len(out)):
        out[i] = fp.logb(xs[i])
    return out


def test_logb_reads_the_exponent_field():
    """No correctly-rounded primitive exists -- `tl.log2` is a
    transcendental, which the op table excludes -- so the exponent is read
    from the bits, which is exact."""
    src = TritonCompiler(drop_asserts=True).compile(
        _logb, ctx=fp.FP32, arg_types=[
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
    """`scalar_fits_in` asks whether the two *types* nest; a conversion only
    needs the *values* to.  They come apart wherever storage is wider than
    the bound it was chosen to hold -- which the cpp backend already knew,
    as `_value_fits`.
    """
    src = TritonCompiler(drop_asserts=True).compile(
        _small_const, ctx=fp.FP16, arg_types=[
            ListType(RealType(fp.FP16), 4),
            ListType(RealType(fp.FP16), 4),
            RealType(fp.INTEGER)])
    assert 'tl.maximum' in src.source


@fp.fpy(ctx=fp.REAL)
def _scaled(xs: list[fp.Real], ns: list[fp.Real], out: list[fp.Real],
            BLOCK: fp.Real):
    """`2 ** n * x` with a per-lane `n` -- what `RescaleFixed` emits."""
    for i in range(len(out)):
        with fp.REAL:
            t = (2 ** ns[i]) * xs[i]
        out[i] = t
    return out


@fp.fpy(ctx=fp.FP32)
def _scaled_rounding(xs: list[fp.Real], ns: list[fp.Real],
                     out: list[fp.Real], BLOCK: fp.Real):
    """The same product where the context *rounds* it."""
    for i in range(len(out)):
        out[i] = (2 ** ns[i]) * xs[i]
    return out


_SCALE_ARGS = [
    ListType(RealType(fp.FP32), 4),
    ListType(RealType(fp.INTEGER), 4),
    ListType(RealType(fp.FP32), 4),
    RealType(fp.INTEGER),
]


def test_a_power_of_two_product_is_ldexp():
    """`ldexp` is `scaleB`: exact, where the product would round twice and
    rest on `exp2` returning the power exactly, which nothing guarantees."""
    src = TritonCompiler(drop_asserts=True).compile(
        _scaled, ctx=fp.REAL, arg_types=_SCALE_ARGS)
    assert 'libdevice.ldexp(' in src.source
    assert '**' not in src.source


def test_ldexp_is_refused_where_the_context_would_round():
    """It stands in for the multiply only where the context would not round
    it -- otherwise it would skip a rounding the program asked for."""
    with pytest.raises(TritonEmitError, match='no signatures for op: Pow'):
        TritonCompiler(drop_asserts=True).compile(
            _scaled_rounding, ctx=fp.FP32, arg_types=_SCALE_ARGS)


def _round_to_int(mode: str):
    """A kernel rounding to the integers under *mode*."""
    src = (
        'import fpy2 as fp\n'
        '@fp.fpy(ctx=fp.REAL)\n'
        'def k(xs, out, BLOCK):\n'
        '    for i in range(len(out)):\n'
        f'        with fp.MPFixedContext(-1, fp.RoundingMode.{mode}):\n'
        '            t = fp.round(xs[i])\n'
        '        out[i] = t\n'
        '    return out\n'
    )
    import importlib.util
    import pathlib
    import sys
    import tempfile
    d = pathlib.Path(tempfile.mkdtemp())
    path = d / f'ir_{mode}.py'
    path.write_text(src)
    spec = importlib.util.spec_from_file_location(f'ir_{mode}', path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[f'ir_{mode}'] = m
    spec.loader.exec_module(m)
    return TritonCompiler(drop_asserts=True).compile(
        m.k, ctx=fp.REAL, arg_types=[
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
