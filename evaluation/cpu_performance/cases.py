"""
Each FPy design paired with the MMA-Sim instruction it reproduces; the same
inputs in both's representations; and the check that their outputs agree
bit for bit.

`prepare` needs the designs' operators loaded (`run.build`).
"""

import os
import platform
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import torch
from torch import Tensor
from torchao.prototype.mx_formats.kernels import (
    f4_unpacked_to_f32,
    f32_to_f4_unpacked,
    unpack_uint4,
)
from torchao.prototype.mx_formats.mx_tensor import to_mx
from torchao.prototype.mx_formats.nvfp4_tensor import nvfp4_quantize


def _import_mmasim() -> tuple[Any, Any, Any]:
    """MMA-Sim's `amd`, `nv_ptx` and `truncate_fp32_to_tf32`.  Its FMA helper
    compiles on import and needs OpenMP, which Apple's clang lacks: on macOS,
    Homebrew's LLVM compiles it, unless `$CXX` is set."""
    brew = Path('/opt/homebrew/opt/llvm/bin/clang++')
    swap = platform.system() == 'Darwin' and 'CXX' not in os.environ and brew.exists()
    if swap:
        os.environ['CXX'] = str(brew)
    try:
        from mmasim import amd, nv_ptx
        from mmasim.arithmetic.helper import truncate_fp32_to_tf32
    except (ImportError, RuntimeError) as e:
        sys.exit(f'cannot import MMA-Sim: {e}\non macOS, its FMA helper needs an OpenMP-capable '
                 'compiler: install Homebrew LLVM or set CXX')
    finally:
        if swap:   # the designs' operators build with the default compiler
            del os.environ['CXX']
    return amd, nv_ptx, truncate_fp32_to_tf32


amd, nv_ptx, truncate_fp32_to_tf32 = _import_mmasim()

# Operand formats without a torch dtype.  MMA-Sim takes TF32 operands as FP32
# and truncates them itself; E2M1 (FP4) operands only as FP16 values, which
# its group-scaled model unpacks them to.
TF32 = 'tf32'
E2M1 = 'e2m1'
Format = torch.dtype | str


@dataclass(frozen=True)
class Case:
    """One FPy design and the MMA-Sim simulator it is compared with."""
    name: str
    reference: Any
    """MMA-Sim's simulator; `reference.dpa` computes one chunk."""
    a_format: Format
    b_format: Format
    accumulator: torch.dtype
    k: int
    """Products per call."""
    op: str
    """`torch.ops.fpy2_models.<op>` computes one chunk, as `dpa` does."""
    compute: torch.dtype
    """FPy's C++ operand type."""
    scale_block: int | None = None
    scale_dtype: torch.dtype | None = None


class Reference(NamedTuple):
    """The MMA-Sim simulator a design reproduces, and its operands' formats."""
    simulator: Any
    a_format: Format
    b_format: Format
    accumulator: torch.dtype
    scale_block: int | None = None
    scale_dtype: torch.dtype | None = None


def references() -> dict[str, Reference]:
    """Each design's name to its `Reference`."""
    f16, bf16, f32, f64 = torch.float16, torch.bfloat16, torch.float32, torch.float64
    e8m0, e4m3 = torch.float8_e8m0fnu, torch.float8_e4m3fn
    nv = {'e4m3': e4m3, 'e5m2': torch.float8_e5m2, 'e2m1': E2M1}
    cdna3 = {'fp8': torch.float8_e4m3fnuz, 'bf8': torch.float8_e5m2fnuz}
    fp8_pairs = [('e4m3', 'e4m3'), ('e5m2', 'e5m2'), ('e4m3', 'e5m2')]
    f8f6f4_pairs = fp8_pairs + [('e2m1', 'e2m1'), ('e4m3', 'e2m1'), ('e5m2', 'e2m1')]

    def tag(a: str, b: str) -> str:
        return a if a == b else f'{a}.{b}'

    out = {
        'nv.volta.f16.f32': Reference(nv_ptx.MMA('Volta', 'm8n8k4.f32.f16.f16.f32'), f16, f16, f32),
        'nv.volta.f16.f16': Reference(nv_ptx.MMA('Volta', 'm8n8k4.f16.f16.f16.f16'), f16, f16, f16),
        'nv.turing.f16.f32': Reference(nv_ptx.MMA('Turing', 'm16n8k8.f32.f16.f16.f32'), f16, f16, f32),
        'nv.turing.f16.f16': Reference(nv_ptx.MMA('Turing', 'm16n8k8.f16.f16.f16.f16'), f16, f16, f16),
        'nv.ampere.tf32.f32': Reference(nv_ptx.MMA('Ampere', 'm16n8k8.f32.tf32.tf32.f32'), TF32, TF32, f32),
        'nv.ampere.bf16.f32': Reference(nv_ptx.MMA('Ampere', 'm16n8k16.f32.bf16.bf16.f32'), bf16, bf16, f32),
        'nv.hopper.tf32.f32': Reference(nv_ptx.WGMMA('Hopper', 'm64n8k8.f32.tf32.tf32'), TF32, TF32, f32),
        'nv.hopper.bf16.f32': Reference(nv_ptx.WGMMA('Hopper', 'm64n8k16.f32.bf16.bf16'), bf16, bf16, f32),
        'nv.hopper.f16.f32': Reference(nv_ptx.WGMMA('Hopper', 'm64n8k16.f32.f16.f16'), f16, f16, f32),
        'nv.hopper.f16.f16.wgmma': Reference(nv_ptx.WGMMA('Hopper', 'm64n8k16.f16.f16.f16'), f16, f16, f16),
        'nv.hopper.f16.f16.mma': Reference(nv_ptx.MMA('Hopper', 'm16n8k16.f16.f16.f16.f16'), f16, f16, f16),
        'nv.blackwell.nvfp4': Reference(
            nv_ptx.TCGen05MMABlockScale('Blackwell', 'm128n8k64.block16.f32.e2m1.e2m1.ue4m3'),
            E2M1, E2M1, f32, 16, e4m3),
        'nv.blackwell.mxfp4': Reference(
            nv_ptx.TCGen05MMABlockScale('Blackwell', 'm128n8k64.block16.f32.e2m1.e2m1.ue8m0'),
            E2M1, E2M1, f32, 16, e8m0),
        'amd.cdna2.bf16': Reference(amd.MFMA('CDNA2', 'f32_32x32x4bf16'), bf16, bf16, f32),
        'amd.cdna2.bf16_1k': Reference(amd.MFMA('CDNA2', 'f32_32x32x8bf16_1k'), bf16, bf16, f32),
        'amd.cdna2.f16': Reference(amd.MFMA('CDNA2', 'f32_32x32x8f16'), f16, f16, f32),
        'amd.cdna3.tf32': Reference(amd.MFMA('CDNA3', 'f32_32x32x4_xf32'), TF32, TF32, f32),
        'amd.cdna3.f16': Reference(amd.MFMA('CDNA3', 'f32_32x32x8_f16'), f16, f16, f32),
        'amd.cdna3.bf16': Reference(amd.MFMA('CDNA3', 'f32_32x32x8_bf16'), bf16, bf16, f32),
        'fp64.fma': Reference(nv_ptx.MMA('Ampere', 'm8n8k4.f64.f64.f64.f64'), f64, f64, f64),
        'fp32.fma': Reference(amd.MFMA('CDNA3', 'f32_32x32x2_f32'), f32, f32, f32),
    }
    for a, b in fp8_pairs:
        for acc, dtype in (('f32', f32), ('f16', f16)):
            out[f'nv.ada.{tag(a, b)}.{acc}'] = Reference(
                nv_ptx.MMA('Ada Lovelace', f'm16n8k32.{acc}.{a}.{b}.{acc}'), nv[a], nv[b], dtype)
            out[f'nv.hopper.{tag(a, b)}.{acc}'] = Reference(
                nv_ptx.WGMMA('Hopper', f'm64n8k32.{acc}.{a}.{b}'), nv[a], nv[b], dtype)
    for a, b in f8f6f4_pairs:
        out[f'nv.blackwell.{tag(a, b)}.f32'] = Reference(
            nv_ptx.TCGen05MMA('Blackwell', f'm128n8k32.f32.{a}.{b}'), nv[a], nv[b], f32)
        out[f'nv.blackwell.{tag(a, b)}.f16.tcgen05'] = Reference(
            nv_ptx.TCGen05MMA('Blackwell', f'm128n8k32.f16.{a}.{b}'), nv[a], nv[b], f16)
        out[f'nv.rtx_blackwell.{tag(a, b)}.f16.mma'] = Reference(
            nv_ptx.MMA('RTX Blackwell', f'm16n8k32.f16.{a}.{b}.f16'), nv[a], nv[b], f16)
        out[f'nv.blackwell.mx.{tag(a, b)}'] = Reference(
            nv_ptx.TCGen05MMABlockScale('Blackwell', f'm128n8k32.block32.f32.{a}.{b}.ue8m0'),
            nv[a], nv[b], f32, 32, e8m0)
    for a, b in (('fp8', 'fp8'), ('bf8', 'bf8'), ('fp8', 'bf8')):
        out[f'amd.cdna3.{tag(a, b)}'] = Reference(
            amd.MFMA('CDNA3', f'f32_32x32x16_{a}_{b}'), cdna3[a], cdna3[b], f32)
    return out


def cases(manifest: dict[str, Any]) -> list[Case]:
    """Every design in *manifest* (`generate_models.py`'s) with its simulator."""
    refs = references()
    out = []
    for m in manifest['models']:
        name = m['name'].replace(' (', '.').rstrip(')')   # `fp64 (fma)` -> `fp64.fma`
        if name not in refs:
            sys.exit(f'no MMA-Sim reference for {name}')
        r = refs[name]
        out.append(Case(name, r.simulator, r.a_format, r.b_format, r.accumulator, m['k'], m['op'],
                        torch.float64 if m['compute'] == 'double' else torch.float32,
                        r.scale_block, r.scale_dtype))
    return out


def select(cs: Sequence[Case], names: Sequence[str]) -> list[Case]:
    """The cases whose names contain any of *names*; all of them if none."""
    if not names:
        return list(cs)
    if not (out := [c for c in cs if any(n in c.name for n in names)]):
        sys.exit(f'no design matches {list(names)}; of {", ".join(c.name for c in cs)}')
    return out


# ---------------------------------------------------------------------------
# Inputs: the same values in MMA-Sim's and FPy's representations

DISTRIBUTIONS = {'normal': 'normal(0, 0.5)', 'uniform': 'uniform[-1, 1)'}


def sample(batch: int, length: int, seed: int, distribution: str) -> Tensor:
    """FP64 values to round to a design's input format."""
    g = torch.Generator().manual_seed(seed)
    if distribution == 'normal':
        return torch.randn((batch, length), dtype=torch.float64, generator=g) * 0.5
    if distribution == 'uniform':
        return torch.rand((batch, length), dtype=torch.float64, generator=g) * 2 - 1
    raise ValueError(f'unknown distribution {distribution!r}')


def round_operand(fmt: Format, values: Tensor, compute: torch.dtype) -> tuple[Tensor, Tensor]:
    """*values* rounded to *fmt*, as (MMA-Sim's operand, FPy's)."""
    if fmt == TF32:
        # FPy's models take operands already TF32; MMA-Sim's truncate FP32 ones
        mm = values.float()
        return mm, truncate_fp32_to_tf32(mm)
    if fmt == E2M1:
        fpy = f4_unpacked_to_f32(f32_to_f4_unpacked(values.float().contiguous()))
        return fpy.half(), fpy
    mm = values.to(fmt)
    return mm, mm.to(compute)


def quantize_blocks(case: Case, fmt: Format, values: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """*values* block-quantized by TorchAO, as (MMA-Sim's operand, FPy's, scales)."""
    values = values.float().contiguous()
    if case.scale_dtype == torch.float8_e4m3fn:
        # single-level NVFP4: any per-tensor scale lies outside the MMA
        scales, packed = nvfp4_quantize(values, case.scale_block)
    else:
        scales, packed = to_mx(values, torch.float4_e2m1fn_x2 if fmt == E2M1 else fmt, case.scale_block)
    if fmt == E2M1:
        fpy = f4_unpacked_to_f32(unpack_uint4(packed.view(torch.uint8)))
        return fpy.half(), fpy, scales
    return packed, packed.float(), scales


@dataclass(frozen=True)
class Prepared:
    """Functions of no arguments, each computing the batch of dot products."""
    fpy: Callable[[], Tensor]
    mmasim: Callable[[], Tensor]
    mmasim_compiled: Callable[[], Tensor] | None = None


def prepare(case: Case, batch: int, seed: int, length: int = 1024, compiled: bool = False,
            distribution: str = 'normal') -> Prepared:
    """*batch* dot products of *length* terms, each from a zero accumulator, by
    each implementation; their inputs built here, outside any timing.  With
    *compiled*, also MMA-Sim's model under default `torch.compile`, which
    compiles on its first call."""
    if length <= 0 or length % case.k:
        raise ValueError(f'length must be a positive multiple of {case.k}')
    a_values = sample(batch, length, seed, distribution)
    b_values = sample(batch, length, seed + 1, distribution)
    if case.scale_block is None:
        a_mm, a_fpy = round_operand(case.a_format, a_values, case.compute)
        b_mm, b_fpy = round_operand(case.b_format, b_values, case.compute)
        scales = []
    else:
        a_mm, a_fpy, a_scales = quantize_blocks(case, case.a_format, a_values)
        b_mm, b_fpy, b_scales = quantize_blocks(case, case.b_format, b_values)
        scales = [a_scales, b_scales]

    def chain(dpa: Callable[..., Tensor], acc: torch.dtype, *operands: Tensor) -> Callable[[], Tensor]:
        """Dot products by one *dpa* call per K-wide chunk, the accumulator
        passed from chunk to chunk."""
        # every operand and its scales sliced into contiguous chunks now, so
        # that timing excludes strided copies
        chunks = list(zip(*([x.contiguous() for x in op.chunk(length // case.k, dim=1)]
                            for op in operands)))

        def dot() -> Tensor:
            c = torch.zeros(batch, dtype=acc)
            for a, b, *s in chunks:
                c = dpa(a, b, c, *s)
            return c
        return dot

    mm = (case.accumulator, a_mm, b_mm, *scales)
    return Prepared(
        fpy=chain(getattr(torch.ops.fpy2_models, case.op).default, case.compute,
                  a_fpy, b_fpy, *(s.to(case.compute) for s in scales)),
        mmasim=chain(case.reference.dpa, *mm),
        mmasim_compiled=chain(torch.compile(case.reference.dpa), *mm) if compiled else None)


# ---------------------------------------------------------------------------
# Validation: outputs must match MMA-Sim's bit for bit

BITS = {torch.float16: torch.int16, torch.float32: torch.int32, torch.float64: torch.int64}


def compare(got: Tensor, want: Tensor, name: str) -> dict[str, Any]:
    """*got* against MMA-Sim's *want*, bit for bit."""
    if got.shape != want.shape or got.dtype != want.dtype:
        return {'valid': False, 'reason': 'shape or dtype mismatch',
                f'{name}_shape': list(got.shape), 'mmasim_shape': list(want.shape),
                f'{name}_dtype': str(got.dtype), 'mmasim_dtype': str(want.dtype)}
    differ = (got.contiguous().view(BITS[got.dtype]) != want.contiguous().view(BITS[want.dtype]))
    if not differ.any():
        return {'valid': True}
    i = int(differ.flatten().nonzero()[0])
    return {'valid': False, 'reason': 'output bit mismatch', 'mismatches': int(differ.sum()),
            'first_difference': i, f'{name}_value': float(got.flatten()[i]),
            'mmasim_value': float(want.flatten()[i])}


def validate(p: Prepared) -> dict[str, Any]:
    """FPy's output against eager MMA-Sim's."""
    want, got = p.mmasim(), p.fpy()
    if got.dtype != want.dtype:
        # FPy computes FP16-output designs in FP32; their results must still
        # be exactly FP16
        narrow = got.to(want.dtype)
        wide = narrow.to(got.dtype)
        if not bool(((wide == got) | (wide.isnan() & got.isnan())).all()):
            return {'valid': False, 'reason': f'FPy output is not exactly {want.dtype}'}
        got = narrow
    return compare(got, want, 'fpy')


def validate_compiled(p: Prepared) -> dict[str, Any]:
    """`torch.compile`'d MMA-Sim against eager; failing to compile is a result."""
    try:
        got = p.mmasim_compiled()
    except Exception as e:  # noqa: BLE001 -- a compile failure is a result
        return {'valid': False, 'reason': 'compile or runtime error',
                'error': f'{type(e).__name__}: {str(e).splitlines()[0][:300]}'}
    return compare(got, p.mmasim(), 'compiled')
