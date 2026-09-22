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

import fpy2 as fp
from fpy2.backend.triton import TritonCompiler
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
    t9 = BLOCK
    t10 = 4
    i = tl.program_id(0) * BLOCK
    j = i + tl.arange(0, BLOCK)
    r = j
    acc = 0
    for k in tl.static_range(8):
        acc = (acc + (tl.load(xss_ptr + r * 8 + k, mask=(j < t10), other=0.0).to(tl.float32) * tl.load(yss_ptr + r * 8 + k, mask=(j < t10), other=0.0).to(tl.float32)))
    tl.store(out_ptr + r, acc, mask=(j < t10))'''


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
