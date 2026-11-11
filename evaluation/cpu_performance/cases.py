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
from torchao.prototype.mx_formats.constants import DTYPE_FP6_E2M3, DTYPE_FP6_E3M2
from torchao.prototype.mx_formats.kernels import (
    f4_unpacked_to_f32,
    f6_e2m3_unpacked_to_f32,
    f6_e3m2_unpacked_to_f32,
    f32_to_f4_unpacked,
    f32_to_f6_e2m3_unpacked,
    f32_to_f6_e3m2_unpacked,
    unpack_uint4,
)
from torchao.prototype.mx_formats.mx_tensor import to_mx
from torchao.prototype.mx_formats.nvfp4_tensor import nvfp4_quantize


def _import_mmasim() -> tuple[Any, Any, Any, str]:
    """MMA-Sim's `amd`, `nv_ptx` and `truncate_fp32_to_tf32`, and the
    compiler of its FMA helper.  The helper compiles on import and needs
    OpenMP, which Apple's clang lacks: on macOS, Homebrew's LLVM compiles it,
    unless `$CXX` is set."""
    brew = Path('/opt/homebrew/opt/llvm/bin/clang++')
    swap = platform.system() == 'Darwin' and 'CXX' not in os.environ and brew.exists()
    if swap:
        os.environ['CXX'] = str(brew)
    cxx = os.environ.get('CXX', 'c++')   # as `torch.utils.cpp_extension` picks it
    try:
        from mmasim import amd, nv_ptx
        from mmasim.arithmetic.helper import truncate_fp32_to_tf32
    except (ImportError, RuntimeError) as e:
        sys.exit(f'cannot import MMA-Sim: {e}\non macOS, its FMA helper needs an OpenMP-capable '
                 'compiler: install Homebrew LLVM or set CXX')
    finally:
        if swap:   # the designs' operators build with the default compiler
            del os.environ['CXX']
    return amd, nv_ptx, truncate_fp32_to_tf32, cxx


amd, nv_ptx, truncate_fp32_to_tf32, MMASIM_CXX = _import_mmasim()

# after MMA-Sim, whose FMA helper compiles on import.  Installed throughout:
# `Prepared.mmasim` alone runs MMA-Sim as released
import no_sync

no_sync.install()

# Operand formats without a torch dtype.  MMA-Sim takes TF32 operands as FP32
# and truncates them itself; FP6 and E2M1 (FP4) operands as FP16 values, which
# its group-scaled model unpacks E2M1 to, their PTX types giving their exponent
# floors.
TF32 = 'tf32'
E3M2, E2M3, E2M1 = 'e3m2', 'e2m3', 'e2m1'
Format = torch.dtype | str

_SUBBYTE: dict[str, tuple[Callable[[Tensor], Tensor], Callable[[Tensor], Tensor], torch.dtype | str]] = {
    E3M2: (f32_to_f6_e3m2_unpacked, f6_e3m2_unpacked_to_f32, DTYPE_FP6_E3M2),
    E2M3: (f32_to_f6_e2m3_unpacked, f6_e2m3_unpacked_to_f32, DTYPE_FP6_E2M3),
    E2M1: (f32_to_f4_unpacked, f4_unpacked_to_f32, torch.float4_e2m1fn_x2),
}
"""Each FP6 and FP4 format's TorchAO conversions, to its codes and from them
to FP32, and the element dtype `to_mx` takes for it."""


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
    nv = {'e4m3': e4m3, 'e5m2': torch.float8_e5m2, 'e3m2': E3M2, 'e2m3': E2M3, 'e2m1': E2M1}
    cdna3 = {'fp8': torch.float8_e4m3fnuz, 'bf8': torch.float8_e5m2fnuz}
    fp8_pairs = [('e4m3', 'e4m3'), ('e5m2', 'e5m2'), ('e4m3', 'e5m2')]
    # pairs within a width only, as `compile._F8F6F4_PAIRS`
    f8f6f4_pairs = fp8_pairs + [('e3m2', 'e3m2'), ('e2m3', 'e2m3'), ('e3m2', 'e2m3'), ('e2m1', 'e2m1')]

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
        'nv.hopper.f16.f16': Reference(nv_ptx.WGMMA('Hopper', 'm64n8k16.f16.f16.f16'), f16, f16, f16),
        'nv.blackwell.nvfp4': Reference(
            nv_ptx.TCGen05MMABlockScale('Blackwell', 'm128n8k64.kind::mxf4nvf4.block16.f32.e2m1.e2m1.ue4m3'),
            E2M1, E2M1, f32, 16, e4m3),
        'nv.blackwell.mxfp4': Reference(
            nv_ptx.TCGen05MMABlockScale('Blackwell', 'm128n8k64.kind::mxf4nvf4.block16.f32.e2m1.e2m1.ue8m0'),
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
            nv_ptx.TCGen05MMA('Blackwell', f'm128n8k32.kind::f8f6f4.f32.{a}.{b}'), nv[a], nv[b], f32)
        out[f'nv.blackwell.{tag(a, b)}.f16'] = Reference(
            nv_ptx.TCGen05MMA('Blackwell', f'm128n8k32.kind::f8f6f4.f16.{a}.{b}'), nv[a], nv[b], f16)
        out[f'nv.blackwell.mx.{tag(a, b)}'] = Reference(
            nv_ptx.TCGen05MMABlockScale('Blackwell', f'm128n8k32.kind::mxf8f6f4.block32.f32.{a}.{b}.ue8m0'),
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
    if len({c.name for c in out}) != len(out) or len({c.op for c in out}) != len(out):
        sys.exit('two designs share a name or an operator')
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
    if fmt in _SUBBYTE:
        to_codes, to_f32, _ = _SUBBYTE[fmt]
        fpy = to_f32(to_codes(values.float().contiguous()))
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
        scales, packed = to_mx(values, _SUBBYTE[fmt][2] if fmt in _SUBBYTE else fmt, case.scale_block)
    if fmt in _SUBBYTE:
        # E2M1 two codes a byte, FP6 one
        codes = unpack_uint4(packed.view(torch.uint8)) if fmt == E2M1 else packed
        fpy = _SUBBYTE[fmt][1](codes)
        return fpy.half(), fpy, scales
    return packed, packed.float(), scales


@dataclass(frozen=True)
class Prepared:
    """Functions of no arguments, each computing the batch of dot products."""
    fpy: Callable[[], Tensor]
    mmasim: Callable[[], Tensor]
    """MMA-Sim as released: the reference, untimed."""
    tuned: Callable[[], Tensor]
    """MMA-Sim tuned, the baseline: `no_sync`'s arithmetic under default
    `torch.compile`, which compiles on its first call."""


def _released(f: Callable[[], Tensor]) -> Callable[[], Tensor]:
    """*f* on MMA-Sim's arithmetic as released."""
    def call() -> Tensor:
        no_sync.install(False)
        try:
            return f()
        finally:
            no_sync.install()
    return call


def prepare(case: Case, batch: int, seed: int, length: int = 1024,
            distribution: str = 'normal') -> Prepared:
    """*batch* dot products of *length* terms, each from a zero accumulator, by
    each implementation; their inputs built here, outside any timing."""
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
        mmasim=_released(chain(case.reference.dpa, *mm)),
        tuned=chain(torch.compile(case.reference.dpa), *mm))


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


def _identical(x: Tensor, y: Tensor) -> bool:
    """Bit for bit, NaNs included."""
    return x.dtype == y.dtype and torch.equal(x.contiguous().view(BITS[x.dtype]),
                                              y.contiguous().view(BITS[y.dtype]))


def check(case: Case) -> str | None:
    """What is wrong with *case*'s operator, or `None`: repeat calls must agree,
    one thread with all of them, and a batched call with one row at a time."""
    p = prepare(case, 2, 123, 2 * case.k)
    if not (_identical(p.fpy(), p.fpy()) and _identical(p.mmasim(), p.mmasim())):
        return 'repeat calls differ'

    threads = torch.get_num_threads()
    # enough rows that `runners.h` splits them over every thread, the last chunk short
    p = prepare(case, 4 * threads + 7, 7, 4 * case.k)
    many = p.fpy()
    torch.set_num_threads(1)
    try:
        one = p.fpy()
    finally:
        torch.set_num_threads(threads)
    if not _identical(many, one):
        return f'{threads} threads and one differ'

    # distinct rows, accumulators and scales catch row offsets, and rows sharing
    # an accumulator or scale
    a = torch.arange(3 * case.k, dtype=case.compute).reshape(3, case.k) / 128
    b = torch.full_like(a, 0.5)
    acc = torch.tensor([0.0, 1.0, -2.0], dtype=case.compute)
    scales: tuple[Tensor, ...] = ()
    if case.scale_block:
        e = torch.arange(3 * (case.k // case.scale_block)).reshape(3, -1) % 5 - 2
        scales = tuple(s.to(case.compute) for s in (2.0 ** e, 2.0 ** e.flip(-1)))
    dpa = getattr(torch.ops.fpy2_models, case.op)
    batched = dpa(a, b, acc, *scales)
    for row in range(3):
        if not _identical(batched[row], dpa(a[row], b[row], acc[row], *(s[row] for s in scales))):
            return f'row {row} of a batched call differs from it alone'
    return None


def validate(p: Prepared, want: Tensor | None = None) -> dict[str, Any]:
    """FPy's output against MMA-Sim's as released, *want* if already computed."""
    want = p.mmasim() if want is None else want
    got = p.fpy()
    if got.dtype != want.dtype:
        # FPy computes FP16-output designs in FP32; their results must still
        # be exactly FP16
        narrow = got.to(want.dtype)
        wide = narrow.to(got.dtype)
        if not bool(((wide == got) | (wide.isnan() & got.isnan())).all()):
            return {'valid': False, 'reason': f'FPy output is not exactly {want.dtype}'}
        got = narrow
    return compare(got, want, 'fpy')


def validate_tuned(p: Prepared, want: Tensor) -> dict[str, Any]:
    """Tuned MMA-Sim against MMA-Sim as released, *want*; failing to compile
    is a result."""
    try:
        got = p.tuned()
    except Exception as e:  # noqa: BLE001 -- a compile failure is a result
        return {'valid': False, 'reason': 'compile or runtime error',
                'error': f'{type(e).__name__}: {str(e).splitlines()[0][:300]}'}
    return compare(got, want, 'tuned')
