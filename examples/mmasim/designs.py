"""
The MMA-Sim designs, and compiling them to C++ and Triton.

`DESIGNS` lists every modeled instruction.  :func:`prepare` and
:func:`cpp_compiler` compile a design to C++; :func:`compile_matmul` compiles
it to a Triton kernel.
"""

from collections.abc import Callable
from typing import Any

from models import amd, nv
from models.nv import RNE_FP16, RZ_E8M13, RZ_FP32
from models.utils import make_fma_dpa

import fpy2 as fp
import fpy2.strategies as st
from fpy2.ast import FuncDef
from fpy2.backend.triton import KernelSource, TritonCompiler
from fpy2.backend.triton.normalize import normalize_module
from fpy2.strategies import TransformDeclined
from fpy2.transform import CompToLoop, FreeVarElim, RescaleFixed, Simplify, ZipElim
from fpy2.transform.const_fold import ConstFold
from fpy2.types import Type
from fpy2.utils import NamedId

_L = fp.types.ListType
_R = fp.types.RealType

Build = Callable[[], tuple[fp.Function, list[Type]]]
"""A design's builder: the design, and its argument types."""


def _vecs(a_ctx, b_ctx, c_ctx, k):
    """The `(A, B, c)` argument types every model shares."""
    return [_L(_R(a_ctx), k), _L(_R(b_ctx), k), _R(c_ctx)]


def _t_chain(
    L: int, a_ctx: fp.EFloatContext, b_ctx: fp.EFloatContext, c_ctx: fp.EFloatContext,
    F: int, rho: fp.Context, k: int, **kw: Any,
) -> Build:
    """A T-FDPA chain's builder, over vectors of length *k*."""
    return lambda: (nv.make_t_fdpa_chain(L, a_ctx, b_ctx, c_ctx, F, rho, **kw),
                    _vecs(a_ctx, b_ctx, c_ctx, k))


# (tag, A format, B format): each unordered pair once; a swapped pair is the
# same design with its arguments swapped
_FP8_PAIRS = [
    ('e4m3', fp.MX_E4M3, fp.MX_E4M3),
    ('e5m2', fp.MX_E5M2, fp.MX_E5M2),
    ('e4m3.e5m2', fp.MX_E4M3, fp.MX_E5M2),
]
_FP6_PAIRS = [
    ('e3m2', fp.MX_E3M2, fp.MX_E3M2),
    ('e2m3', fp.MX_E2M3, fp.MX_E2M3),
    ('e3m2.e2m3', fp.MX_E3M2, fp.MX_E2M3),
]
# pairs within a width only: the instructions also take FP8 x FP6, FP8 x FP4
# and FP6 x FP4, left out
_F8F6F4_PAIRS = _FP8_PAIRS + _FP6_PAIRS + [
    ('e2m1', fp.MX_E2M1, fp.MX_E2M1),
]
_CDNA3_FP8_PAIRS = [
    ('fp8', fp.S1E4M3, fp.S1E4M3),
    ('bf8', fp.S1E5M2, fp.S1E5M2),
    ('fp8.bf8', fp.S1E4M3, fp.S1E5M2),
]

# (name, builder) where the builder returns (function, argument types): one
# per row of the paper's Tables 4-6, a row listing several input types once
# per pair.  A row that builds the same function as another is listed once,
# its other architectures noted.
DESIGNS = [
    ('nv.volta.f16.f32', _t_chain(4, fp.FP16, fp.FP16, fp.FP32, 23, RZ_FP32, 8)),
    ('nv.volta.f16.f16', _t_chain(4, fp.FP16, fp.FP16, fp.FP16, 23, RNE_FP16, 8)),
    # also Ampere, Ada
    ('nv.turing.f16.f32', _t_chain(8, fp.FP16, fp.FP16, fp.FP32, 24, RZ_FP32, 16)),
    ('nv.turing.f16.f16', _t_chain(8, fp.FP16, fp.FP16, fp.FP16, 24, RNE_FP16, 16)),
    # also Ada
    ('nv.ampere.tf32.f32', _t_chain(4, fp.TF32, fp.TF32, fp.FP32, 24, RZ_FP32, 8)),
    ('nv.ampere.bf16.f32', _t_chain(8, fp.BF16, fp.BF16, fp.FP32, 24, RZ_FP32, 16)),
    *[(f'nv.ada.{tag}.f32', _t_chain(16, a, b, fp.FP32, 13, RZ_E8M13, 16, e_zero=-132))
      for tag, a, b in _FP8_PAIRS],
    *[(f'nv.ada.{tag}.f16', _t_chain(16, a, b, fp.FP16, 13, RNE_FP16, 16, e_zero=-20))
      for tag, a, b in _FP8_PAIRS],
    # also Blackwell, RTX Blackwell
    ('nv.hopper.tf32.f32', _t_chain(8, fp.TF32, fp.TF32, fp.FP32, 25, RZ_FP32, 16)),
    ('nv.hopper.bf16.f32', _t_chain(16, fp.BF16, fp.BF16, fp.FP32, 25, RZ_FP32, 32)),
    ('nv.hopper.f16.f32', _t_chain(16, fp.FP16, fp.FP16, fp.FP32, 25, RZ_FP32, 32)),
    # wgmma and mma; also Blackwell tcgen05, Blackwell and RTX Blackwell mma
    ('nv.hopper.f16.f16', _t_chain(16, fp.FP16, fp.FP16, fp.FP16, 25, RNE_FP16, 32)),
    *[(f'nv.hopper.{tag}.f32', _t_chain(32, a, b, fp.FP32, 13, RZ_E8M13, 32, e_zero=-133))
      for tag, a, b in _FP8_PAIRS],
    *[(f'nv.hopper.{tag}.f16', _t_chain(32, a, b, fp.FP16, 13, RNE_FP16, 32, e_zero=-21))
      for tag, a, b in _FP8_PAIRS],
    # Blackwell tcgen05, also RTX Blackwell mma
    *[(f'nv.blackwell.{tag}.f32', _t_chain(32, a, b, fp.FP32, 25, RZ_FP32, 32))
      for tag, a, b in _F8F6F4_PAIRS],
    *[(f'nv.blackwell.{tag}.f16', _t_chain(32, a, b, fp.FP16, 25, RNE_FP16, 32))
      for tag, a, b in _F8F6F4_PAIRS],
    # also RTX Blackwell
    *[(f'nv.blackwell.mx.{tag}', lambda a=a, b=b: (
        nv.make_st_fdpa(a, b, fp.MX_E8M0, 25, RZ_FP32),
        _vecs(a, b, fp.FP32, 32) + [_R(fp.MX_E8M0), _R(fp.MX_E8M0)]))
      for tag, a, b in _F8F6F4_PAIRS],
    *[(f'nv.blackwell.{tag}', lambda s=s: (
        nv.make_gst_fdpa(16, s, 35, RZ_FP32),
        _vecs(fp.MX_E2M1, fp.MX_E2M1, fp.FP32, 64) + [_L(_R(s), 4), _L(_R(s), 4)]))
      for tag, s in [('nvfp4', fp.MX_E4M3), ('mxfp4', fp.MX_E8M0)]],
    ('amd.cdna1.bf16', lambda: (
        amd.make_e_fdpa(2), _vecs(fp.BF16, fp.BF16, fp.FP32, 4))),
    ('amd.cdna1.f16', lambda: (
        amd.make_e_fdpa(4), _vecs(fp.FP16, fp.FP16, fp.FP32, 4))),
    ('amd.cdna2.bf16', lambda: (
        amd.make_ftz_addmul(fp.BF16, 2), _vecs(fp.BF16, fp.BF16, fp.FP32, 4))),
    ('amd.cdna2.bf16_1k', lambda: (
        amd.make_ftz_addmul(fp.BF16, 4), _vecs(fp.BF16, fp.BF16, fp.FP32, 4))),
    ('amd.cdna2.f16', lambda: (
        amd.make_ftz_addmul(fp.FP16, 4), _vecs(fp.FP16, fp.FP16, fp.FP32, 4))),
    ('amd.cdna3.tf32', lambda: (
        amd.make_tr_fdpa(4, fp.TF32, fp.TF32),
        _vecs(fp.TF32, fp.TF32, fp.FP32, 8))),
    ('amd.cdna3.f16', lambda: (
        amd.make_tr_fdpa(8, fp.FP16, fp.FP16),
        _vecs(fp.FP16, fp.FP16, fp.FP32, 8))),
    ('amd.cdna3.bf16', lambda: (
        amd.make_tr_fdpa(8, fp.BF16, fp.BF16),
        _vecs(fp.BF16, fp.BF16, fp.FP32, 8))),
    *[(f'amd.cdna3.{tag}', lambda a=a, b=b: (
        amd.make_gtr_fdpa(16, a, b), _vecs(a, b, fp.FP32, 16)))
      for tag, a, b in _CDNA3_FP8_PAIRS],
    # also NVIDIA DMMA, CDNA2/3 FP64
    ('fp64 (fma)', lambda: (
        make_fma_dpa(fp.FP64), _vecs(fp.FP64, fp.FP64, fp.FP64, 4))),
    # CDNA1-3 FP32
    ('fp32 (fma)', lambda: (
        make_fma_dpa(fp.FP32), _vecs(fp.FP32, fp.FP32, fp.FP32, 4))),
]


def _declinable(step: Callable[[FuncDef], FuncDef]) -> Callable[[Any, FuncDef], FuncDef]:
    """*step* over a module's functions, where declining (nothing to do) is not
    a failure."""
    def apply(_module: Any, func: FuncDef) -> FuncDef:
        try:
            return step(func)
        except TransformDeclined:
            return func
    return apply


def prepare(build: Build) -> fp.Module:
    """*build*'s module, as the C++ backend takes it: free variables removed and
    `zip`s eliminated, then inlined, as the Triton backend does (inlining lowers
    a `zip` to tuples), then folded and prepared."""
    func, arg_types = build()
    mod = fp.Module()
    mod.add(st.monomorphize(func, args=arg_types))
    mod = normalize_module(mod.map(_declinable(FreeVarElim.apply)).map(_declinable(ZipElim.apply)))
    for step in (ConstFold.apply, CompToLoop.apply, RescaleFixed.apply, Simplify.apply):
        mod = mod.map(_declinable(step))
    return mod


def cpp_compiler() -> fp.CppCompiler:
    """The designs' C++ compiler: every rounding unfolded and only
    round-to-nearest-even native (`enable_fenv=False`), so no model calls
    `fesetround`."""
    return fp.CppCompiler(unfold=fp.CppCompiler.UnfoldMode.ROUNDINGS, enable_fenv=False)


def _length(t: Type) -> int:
    """*t*'s length, where it is a list of known length."""
    if not (isinstance(t, _L) and isinstance(t.length, int)):
        raise TypeError(f'expected a list of known length, got {t}')
    return t.length


def _matmul(design: fp.Function, arity: int) -> fp.Function:
    """*design* as a matmul: `out[i][j]` is row `i` of `A` against row `j` of
    `BT`, and a scale is indexed the way its operand is."""
    if arity == 3:
        @fp.fpy(ctx=fp.REAL)
        def matmul3(A, BT, C, out, BLOCK):
            for i in range(len(out)):
                row = out[i]
                for j in range(len(row)):
                    row[j] = design(A[i], BT[j], C[i][j])
            return out
        return matmul3
    if arity == 5:
        @fp.fpy(ctx=fp.REAL)
        def matmul5(A, BT, C, xs, ys, out, BLOCK):
            for i in range(len(out)):
                row = out[i]
                for j in range(len(row)):
                    row[j] = design(A[i], BT[j], C[i][j], xs[i], ys[j])
            return out
        return matmul5
    raise ValueError(f'no matmul wrapper for {arity} arguments')


def typed(build: Build, k: int | None) -> tuple[fp.Function, list[Type]]:
    """The function :func:`compile_matmul` compiles for *build*, and its
    argument types at *k* (`None`: the design's own): a block-scaled design's
    chain of instructions (:func:`_chained`), its scales one per instruction
    or group."""
    design, (a, b, c, *scales) = build()
    assert isinstance(a, _L) and isinstance(b, _L)
    k0 = _length(a)
    k = k0 if k is None else k
    if k % k0:
        raise ValueError(f'k = {k} is not a multiple of the design\'s {k0}')
    if not scales:
        return design, [_L(a.elt, k), _L(b.elt, k), c]
    sx = scales[0]
    per = _length(sx) if isinstance(sx, _L) else None
    s, elt = k // k0 * (per or 1), getattr(sx, 'elt', sx)
    return _chained(design, k0, per), [_L(a.elt, k), _L(b.elt, k), c, _L(elt, s), _L(elt, s)]


def compile_matmul(
    build: Build, k: int | None,
) -> tuple[KernelSource, fp.Function, list[Type]]:
    """The kernel for one design as a matmul, the function it computes and its
    argument types at *k* (:func:`typed`); raises whatever refused it.  The
    sizes are the kernel's to be told, so one kernel runs at any `k`."""
    f, arg_types = typed(build, k)
    a, b, c, *scales = arg_types
    sym = [_L(a.elt, NamedId('k')), _L(b.elt, NamedId('k')), c,
           *(_L(t.elt, NamedId('s')) for t in scales)]
    return _compile(f, sym), f, arg_types


def _compile(f: fp.Function, arg_types: list[Type]) -> KernelSource:
    """*f*, one output from `(A, B, C, *scales)` of *arg_types*, as a matmul
    kernel with `m` and `n` symbolic, a scale indexed as its operand."""
    a, b, c, *scales = arg_types
    m, n = NamedId('m'), NamedId('n')
    square = _L(_L(c, n), m)
    return TritonCompiler(
        drop_asserts=True, unfold=TritonCompiler.UnfoldMode.DOUBLE_ROUND,
    ).compile(
        _matmul(f, len(arg_types)), ctx=fp.REAL,
        arg_types=[_L(a, m), _L(b, n), square,
                   *(_L(t, d) for t, d in zip(scales, (m, n))),
                   square, _R(fp.INTEGER)],
    )


def _chained(design: fp.Function, k0: int, per: int | None) -> fp.Function:
    """A block-scaled *design*, one instruction of *k0* elements, chained over
    its scales: instruction `t` takes its elements, its scales (one, or
    *per*), and the last one's result as its accumulator."""
    if per is None:
        @fp.fpy(ctx=fp.REAL)
        def chain1(A, B, c, xs, ys):
            d = c
            for t in range(len(xs)):
                i = t * k0
                d = design(A[i:i + k0], B[i:i + k0], d, xs[t], ys[t])
            return d
        return chain1
    g = k0 // per

    @fp.fpy(ctx=fp.REAL)
    def chain(A, B, c, xs, ys):
        d = c
        for t in range(0, len(xs), per):
            i = t * g
            d = design(A[i:i + k0], B[i:i + k0], d, xs[t:t + per], ys[t:t + per])
        return d
    return chain
