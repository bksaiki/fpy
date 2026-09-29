"""
Quantization schemes, in FPy formats: :func:`quantize` is round-to-nearest
as `torchao` computes it, in a few tensor ops, and
:meth:`Quantized.dequantize` multiplies out exactly in FP64.  See
`docs/todos/mmasim-serving.md`.
"""

from dataclasses import dataclass, replace
from typing import Literal

import torch

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

NAMES = (*SCHEMES, 'fp8-row:fnuz', 'fp8-block:fnuz')
"""Every scheme :func:`scheme` takes by name."""

_FNUZ = {fp.MX_E4M3: fp.S1E4M3, fp.MX_E5M2: fp.S1E5M2}

DTYPES = {
    fp.BF16: torch.bfloat16,
    fp.MX_E4M3: torch.float8_e4m3fn, fp.MX_E5M2: torch.float8_e5m2,
    fp.S1E4M3: torch.float8_e4m3fnuz, fp.S1E5M2: torch.float8_e5m2fnuz,
    fp.MX_E2M1: torch.float4_e2m1fn_x2,
}
"""Each element format's torch dtype."""


def describe(s: Scheme) -> str:
    """*s* in a line: its elements, each operand's scale block, and where the
    scales apply."""
    def block(op: Operand) -> str:
        sc = op.scaling
        if sc is None:
            return 'none'
        return f'{sc.rows}x{sc.cols or "k"}' + (' and per tensor' if sc.tensor else '')

    elements = str(DTYPES[s.x.elements]).removeprefix('torch.')
    if s.x.scaling is None:
        return f'{elements}, unscaled'
    return f'{elements}; scales x {block(s.x)}, w {block(s.w)}; applied {s.applied}'


def context(fmt: object) -> Context:
    """The element format among :data:`DTYPES` that is *fmt* (a design's
    argument format)."""
    return next(c for c in DTYPES if c.format() == fmt)


def scheme(name: str) -> Scheme:
    """A named scheme; `fp8-row:fnuz` and `fp8-block:fnuz` take FNUZ FP8
    elements, as CDNA3's designs do."""
    base, _, variant = name.partition(':')
    if base not in SCHEMES:
        raise ValueError(f'no scheme {name!r}')
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


_E4M3_TINY = torch.finfo(torch.float8_e4m3fn).tiny
_F32_TINY = torch.finfo(torch.float32).tiny


def _e2m1(x: torch.Tensor) -> torch.Tensor:
    """*x*, within ±6, rounded to E2M1 (nearest, ties to even): steps of
    1/2 below 2, 1 below 4, 2 up to 6."""
    a = x.abs()
    step = torch.where(a < 2, 0.5, torch.where(a < 4, 1.0, 2.0))
    return torch.copysign(torch.round(a / step) * step, x)


def quantize(t: torch.Tensor, op: Operand, tensor: torch.Tensor | None = None,
             amax: torch.Tensor | None = None) -> Quantized:
    """*t* `[r, k]` (FP32, finite) by round-to-nearest for *op*, as `torchao`
    computes it:

    - FP8 per block: `s = max(amax, tiny) / max`, elements `(v / s)`
      clamped and cast; a ragged last row block zero-padded.
    - MX: NVIDIA's `RCEIL` scale `2^ceil(log2(amax / max))`, so nothing
      saturates.
    - NVFP4: per-tensor scale *tensor*, else `amax / (448 * 6)` from *amax*
      (a scalar, or `[r, 1]` per row), else from `amax(|t|)`; per 16,
      `rne_e4m3((amax_block / 6) / g)`.

    `k` must be whole blocks."""
    s = op.scaling
    dtype = DTYPES[op.elements]
    if s is None:
        # reuse *t* when lossless (e.g. BF16 weights held in FP32)
        q = t.to(dtype).float()
        return Quantized(op, t if t.dtype == torch.float32 and torch.equal(q, t) else q)
    r, k = t.shape
    if s.fmt == fp.MX_E8M0:
        top = 6.0 if op.elements == fp.MX_E2M1 else torch.finfo(dtype).max
        v = t.view(r, k // s.cols, s.cols)
        e = torch.ceil(torch.log2(v.abs().amax(-1, keepdim=True) / top)).clamp(-127, 127) + 127
        rcp = torch.where(e == 0, 1.0, torch.exp2(127 - e))
        v = (v * rcp).clamp(-top, top)
        elements = _e2m1(v) if op.elements == fp.MX_E2M1 else v.to(dtype).float()
        scales = e[..., 0].to(torch.uint8).view(torch.float8_e8m0fnu).float()
        return Quantized(op, elements.view(r, k), scales)
    if s.tensor:
        v = t.view(r, k // s.cols, s.cols)
        block = v.abs().amax(-1)
        if tensor is None:
            tensor = (block.amax() if amax is None else amax).float() / (448.0 * 6.0)
        scales = ((block / 6.0) / tensor).clamp(_E4M3_TINY, 448.0).to(torch.float8_e4m3fn).float()
        v = (v * ((1.0 / tensor) / scales)[..., None]).clamp(-6.0, 6.0)
        return Quantized(op, _e2m1(v).view(r, k), scales, tensor)
    top = torch.finfo(dtype).max
    rows, cols = s.rows, s.cols or k
    v = torch.nn.functional.pad(t, (0, 0, 0, -r % rows))
    v = v.view(-1, rows, k // cols, cols)
    scales = v.abs().amax((1, 3), keepdim=True).clamp(min=_F32_TINY) / top
    elements = (v / scales).clamp(-top, top).to(dtype).float().view(-1, k)[:r]
    return Quantized(op, elements, scales.view(-1, k // cols))
