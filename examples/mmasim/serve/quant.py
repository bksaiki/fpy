"""
Quantization schemes, in FPy formats: `torchao`'s RTN quantizes, and
:meth:`Quantized.dequantize` multiplies out exactly in FP64 (`torchao`'s
rounds to FP32).  See `docs/todos/mmasim-serving.md`.
"""

from dataclasses import dataclass, replace
from typing import Literal

import torch
from torchao.prototype.mx_formats.config import ScaleCalculationMode
from torchao.prototype.mx_formats.kernels import f4_unpacked_to_f32, unpack_uint4
from torchao.prototype.mx_formats.mx_tensor import MXTensor
from torchao.prototype.mx_formats.nvfp4_tensor import (
    nvfp4_quantize,
    per_tensor_amax_to_scale,
)
from torchao.quantization import Float8Tensor, PerBlock, PerRow
from torchao.quantization.quantize_.common.kernel_preference import KernelPreference

import fpy2 as fp
from fpy2 import Context


@dataclass(frozen=True)
class Scaling:
    """An operand's scales: one per block of `rows` x `cols` (`None`: the
    whole of that axis), in *fmt*; `tensor` adds NVFP4's per-tensor FP32
    level above them."""

    fmt: Context
    rows: int
    cols: int | None
    tensor: bool = False


@dataclass(frozen=True)
class Operand:
    elements: Context
    scaling: Scaling | None = None


@dataclass(frozen=True)
class Scheme:
    """How `x` `[m, k]` and `W` `[n, k]` are represented, and where the
    scales are applied: not at all, after the GEMM (`epilogue`), to each
    block of `k`'s partial (`k-blocks`), or by the instruction."""

    name: str
    x: Operand
    w: Operand
    applied: Literal['none', 'epilogue', 'k-blocks', 'instruction']


def _mx(elements: Context, block: int) -> Operand:
    return Operand(elements, Scaling(fp.MX_E8M0, 1, block))


def _nvfp4() -> Operand:
    return Operand(fp.MX_E2M1, Scaling(fp.MX_E4M3, 1, 16, tensor=True))


SCHEMES = {s.name: s for s in [
    Scheme('bf16', Operand(fp.BF16), Operand(fp.BF16), 'none'),
    Scheme('fp8-row', Operand(fp.MX_E4M3, Scaling(fp.FP32, 1, None)),
           Operand(fp.MX_E4M3, Scaling(fp.FP32, 1, None)), 'epilogue'),
    Scheme('fp8-block', Operand(fp.MX_E4M3, Scaling(fp.FP32, 1, 128)),
           Operand(fp.MX_E4M3, Scaling(fp.FP32, 128, 128)), 'k-blocks'),
    Scheme('mxfp8', _mx(fp.MX_E4M3, 32), _mx(fp.MX_E4M3, 32), 'instruction'),
    Scheme('mxfp4', _mx(fp.MX_E2M1, 32), _mx(fp.MX_E2M1, 32), 'instruction'),
    Scheme('nvfp4', _nvfp4(), _nvfp4(), 'instruction'),
]}
"""The named schemes (`docs/todos/mmasim-serving.md`)."""

_FNUZ = {fp.MX_E4M3: fp.S1E4M3, fp.MX_E5M2: fp.S1E5M2}

DTYPES = {
    fp.BF16: torch.bfloat16,
    fp.MX_E4M3: torch.float8_e4m3fn, fp.MX_E5M2: torch.float8_e5m2,
    fp.S1E4M3: torch.float8_e4m3fnuz, fp.S1E5M2: torch.float8_e5m2fnuz,
    fp.MX_E2M1: torch.float4_e2m1fn_x2,
}
"""Each element format's torch dtype, as `torchao` takes it."""


def context(fmt: object) -> Context:
    """The element format among :data:`DTYPES` that is *fmt* (a design's
    argument format)."""
    return next(c for c in DTYPES if c.format() == fmt)


def scheme(name: str) -> Scheme:
    """A named scheme; `fp8-row:fnuz` and `fp8-block:fnuz` take FNUZ FP8
    elements, as CDNA3's designs do."""
    base, _, variant = name.partition(':')
    s = SCHEMES[base]
    if not variant:
        return s
    if variant != 'fnuz' or s.applied not in ('epilogue', 'k-blocks'):
        raise ValueError(f'no scheme {name!r}')
    return replace(s, name=name, x=replace(s.x, elements=_FNUZ[s.x.elements]),
                   w=replace(s.w, elements=_FNUZ[s.w.elements]))


@dataclass
class Quantized:
    """An operand `[r, k]` quantized: FP32 tensors holding its elements'
    values, its scales' (one per block, `[ceil(r / rows), ceil(k / cols)]`),
    and its per-tensor scale's (a scalar, or `[r, 1]`, one per row)."""

    operand: Operand
    elements: torch.Tensor
    scales: torch.Tensor | None = None
    tensor: torch.Tensor | None = None

    def take(self, rows: torch.Tensor | slice) -> 'Quantized':
        """Its rows *rows*, for an operand whose scale blocks are one row
        tall (every scheme's activations)."""
        if self.operand.scaling is not None and self.operand.scaling.rows != 1:
            raise ValueError('rows of scale blocks taller than one')
        return replace(self, elements=self.elements[rows], scales=_rows(self.scales, rows),
                       tensor=_rows(self.tensor, rows))

    def dequantize(self, rows: slice = slice(None)) -> torch.Tensor:
        """Its values, of *rows*, in FP64, exactly."""
        v = self.elements[rows].double()
        if self.scales is not None:
            s, (r, k) = self.operand.scaling, v.shape
            i = torch.arange(self.elements.shape[0], device=v.device)[rows] // s.rows
            v = (v.view(r, -1, s.cols or k) * self.scales[i].double()[..., None]).view(r, k)
        return v if self.tensor is None else v * _rows(self.tensor, rows).double()


def _rows(t: torch.Tensor | None, rows: torch.Tensor | slice) -> torch.Tensor | None:
    """*t*'s *rows*, unless it is a scalar or absent."""
    return t if t is None or t.dim() == 0 else t[rows]


def _fp4(packed: torch.Tensor) -> torch.Tensor:
    return f4_unpacked_to_f32(unpack_uint4(packed))


def quantize(t: torch.Tensor, op: Operand, tensor: torch.Tensor | None = None,
             amax: torch.Tensor | None = None) -> Quantized:
    """*t* `[r, k]` by `torchao`'s RTN for *op*: MX with `RCEIL` scales
    (cuBLAS's; nothing saturates); NVFP4 with per-tensor scale *tensor*,
    else from *amax* (a scalar, or `[r, 1]` per row), else from `amax(|t|)`;
    FP8 with `amax / max` per block, a ragged last row block zero-padded.
    `k` must be whole blocks."""
    s = op.scaling
    dtype = DTYPES[op.elements]
    if s is None:
        # reuse *t* when lossless (e.g. BF16 weights held in FP32)
        q = t.to(dtype).float()
        return Quantized(op, t if t.dtype == torch.float32 and torch.equal(q, t) else q)
    if s.fmt == fp.MX_E8M0:
        q = MXTensor.to_mx(t, dtype, s.cols, ScaleCalculationMode.RCEIL)
        elements = _fp4(q.qdata) if op.elements == fp.MX_E2M1 else q.qdata.float()
        return Quantized(op, elements, q.scale.float())
    if s.tensor:
        if tensor is None:
            tensor = per_tensor_amax_to_scale(t.abs().amax() if amax is None else amax)
        scales, packed = nvfp4_quantize(t.float().contiguous(), s.cols, tensor)
        return Quantized(op, _fp4(packed), scales.float(), tensor)
    granularity = PerRow() if s.cols is None else PerBlock([s.rows, s.cols])
    r = t.shape[0]
    padded = torch.nn.functional.pad(t, (0, 0, 0, -r % s.rows))
    # a lower bound keeps an all-zero block's scale from 0 (and its elements from NaN)
    q = Float8Tensor.from_hp(padded, float8_dtype=dtype, granularity=granularity,
                             hp_value_lb=torch.finfo(torch.float32).tiny,
                             kernel_preference=KernelPreference.TORCH)
    return Quantized(op, q.qdata.float()[:r], q.scale.float())
