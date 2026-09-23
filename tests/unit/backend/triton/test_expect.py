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
