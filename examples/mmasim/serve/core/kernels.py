"""
The designs' Triton matmuls, as a linear layer's `x @ W.T`.

Each design compiles once with `m`, `n` and `k` symbolic; `k` must be a
multiple of the design's length.  Its inputs hold values of its input
formats in the kernel's storage dtype; the result is its FP32 accumulator.
:func:`linear` does it all per call; `swap` calls :func:`prepare` once per
weight and :func:`matmul` per call.
"""

import os
import sys
from collections.abc import Callable, Sequence
from functools import cache, partial
from pathlib import Path
from typing import Any, Literal

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import compile_triton as ct
from compile import DESIGNS, in_processes
from compile import _length as _arg_length

from fpy2.backend.triton import KernelSource, launch
from fpy2.backend.triton.launcher import _torch_dtype

from . import quant

TILES: dict[str, tuple[int, int]] = {
    'nv.ampere.bf16.f32': (32, 1),
    'nv.hopper.bf16.f32': (32, 1),
    'amd.cdna2.bf16': (16, 64),
    'amd.cdna2.bf16_1k': (16, 64),
    'amd.cdna3.bf16': (64, 1),
    'nv.ada.e4m3.f32': (128, 1),
    'nv.hopper.e4m3.f32': (32, 1),
    'nv.blackwell.e4m3.f32': (32, 1),
    'amd.cdna3.fp8': (64, 1),
    'nv.blackwell.mx.e4m3': (32, 1),
    'nv.blackwell.mx.e2m1': (32, 1),
    'nv.blackwell.mxfp4': (16, 1),
    'nv.blackwell.nvfp4': (16, 1),
}
"""Each design and its fixed tile `(block, block_m)`, the fastest per
`bench/speed.py --best`."""

Combine = Literal['linear', 'tree']

_BUILDS: dict[str, Callable] = dict(DESIGNS)

_KERNELS: dict[str, KernelSource] = {}
"""Each design compiled so far."""

_ELEMS = 1 << 26
"""Output elements per block of rows: bounds the partials' memory under
`split_k > 1`, and a launch's size."""


def register(name: str, build: Callable, block: int, block_m: int) -> None:
    """Add a design, as `compile.DESIGNS` defines them (*build* returns the
    FPy function and its argument types), run at the tile `(block, block_m)`."""
    if name in TILES:
        raise ValueError(f'{name!r} is already a design')
    _BUILDS[name] = build
    TILES[name] = (block, block_m)


@cache
def _args(design: str) -> list[Any]:
    """*design*'s argument types."""
    if design not in TILES:
        raise ValueError(f'{design!r} is not one of {sorted(TILES)}')
    return _BUILDS[design]()[1]


def formats(design: str) -> tuple[Any, Any, Any]:
    """*design*'s activation, weight and accumulator formats."""
    a, b, c = _args(design)[:3]
    return a.elt.fmt, b.elt.fmt, c.fmt


def block_scaled(design: str) -> bool:
    """Whether *design* takes scales, as an instruction's own arguments."""
    return len(_args(design)) == 5


def group(design: str) -> int:
    """How many elements of `k` one of a block-scaled *design*'s scales
    covers in a call."""
    k0, scale = _arg_length(_args(design)[0]), _args(design)[3]
    return k0 // scale.length if hasattr(scale, 'length') else k0


def applicable(design: str, scheme: quant.Scheme) -> bool:
    """Whether *design* takes *scheme*'s elements, and its scales: applied
    outside an unscaled design, or taken by a block-scaled one in its format,
    in groups dividing the scheme's blocks."""
    x, w, _ = formats(design)
    if (x, w) != (scheme.x.elements.format(), scheme.w.elements.format()):
        return False
    if not block_scaled(design):
        return scheme.applied in ('none', 'epilogue', 'k-blocks')
    s, arg = scheme.x.scaling, _args(design)[3]
    scale = getattr(arg, 'elt', arg)
    return (scheme.applied == 'instruction' and scale.fmt == s.fmt.format()
            and s.cols % group(design) == 0)


def designs(scheme: quant.Scheme) -> list[str]:
    return [d for d in TILES if applicable(d, scheme)]


def compiled(design: str) -> tuple[KernelSource, int]:
    """*design*'s kernel and the length its `k` must be a multiple of (a
    block-scaled design's, one instruction)."""
    if design not in _KERNELS:
        _KERNELS[design] = ct.compile_matmul(_BUILDS[design], None)[0]
    return _KERNELS[design], _arg_length(_args(design)[0])


def _compile_named(name: str) -> KernelSource:
    return ct.compile_matmul(dict(DESIGNS)[name], None)[0]


def precompile(designs: Sequence[str], jobs: int | None = None) -> None:
    """Compile *designs* not yet compiled, MMA-Sim's in *jobs* processes (one
    per core by default); a :func:`register`ed one compiles here."""
    todo = [d for d in designs if d not in _KERNELS]
    named = [d for d in todo if _BUILDS[d] is dict(DESIGNS).get(d)]
    for d, kernel in zip(named, in_processes(_compile_named, named,
                                             min(len(named), jobs or os.cpu_count() or 1))):
        _KERNELS[d] = kernel
    for d in todo:
        compiled(d)


@cache
def storage(design: str) -> tuple[torch.dtype, ...]:
    """The torch dtypes *design*'s kernel holds its activations and weights
    in, and a block-scaled one's scales."""
    held = dict(compiled(design)[0].dtypes)
    return tuple(_torch_dtype(held[i]) for i in ((0, 1, 3) if block_scaled(design) else (0, 1)))


def prepare(w: torch.Tensor, dtype: torch.dtype, split_k: int = 1) -> torch.Tensor:
    """*w* `[n, k]`, holding a design's weight values, in *dtype* as
    `[split_k, n, k / split_k]` contiguous slices of `k`: a view of *w* at
    `split_k = 1` when it is in *dtype* already."""
    n, k = w.shape
    if split_k < 1 or k % split_k:
        raise ValueError(f'`k` = {k} does not split into {split_k} slices')
    b = w.to(dtype)
    if split_k == 1:
        return b.unsqueeze(0)
    return b.view(n, split_k, k // split_k).transpose(0, 1).contiguous()


def _launch(a: torch.Tensor, b: torch.Tensor, y: torch.Tensor, design: str,
            *scales: torch.Tensor) -> None:
    """Accumulate `a @ b.T` by *design* (block-scaled: with *scales*) onto
    *y* `[m, n]`, which is both `C` and the output.  `block_m` is capped at
    the power of two `>= m`."""
    block, block_m = TILES[design]
    m = a.shape[0]
    launch(compiled(design)[0], [a, b, y, *scales, y],
           block=block, block_m=min(block_m, 1 << (m - 1).bit_length()))


def _part(a: torch.Tensor, w: torch.Tensor, design: str, j: int) -> torch.Tensor:
    """Slice *j*'s partial, `a[j] @ w[j].T` from a zero accumulator."""
    out = torch.zeros(a.shape[1], w.shape[1], dtype=torch.float32, device=a.device)
    _launch(a[j], w[j], out, design)
    return out


def _tree(part: Callable[[int], torch.Tensor], s: int, n: int) -> torch.Tensor:
    """Partials `[s, s + n)` summed pairwise, the left half the largest power
    of two below `n`."""
    if n == 1:
        return part(s)
    h = 1 << ((n - 1).bit_length() - 1)
    return _tree(part, s, h).add_(_tree(part, s + h, n - h))


def matmul(
    a: torch.Tensor, w: torch.Tensor, design: str, combine: Combine = 'linear',
    scales: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """`a @ w.T` by *design*, FP32 `[m, n]`, for `a` holding its activation values
    and `w` from :func:`prepare` in *design*'s :func:`storage`.

    With `w` in `s > 1` slices, each slice goes through the kernel from a zero
    accumulator and the FP32 partials are summed left to right (`linear`) or
    pairwise (`tree`); one slice accumulates `k` in order through the design
    alone.  With *scales* `(s_x [m, s], s_w [n, s])`, each slice's partial is
    scaled as it is summed, left to right, `y + p * (s_x * s_w)` in FP32, as
    block-scaled FP8 promotes it.  A block-scaled design takes `w` in one
    slice per instruction and *scales* `(xs [s, m, ...], ys [s, n, ...])`,
    each instruction's, and chains them: each takes the last one's result
    as its accumulator."""
    s, n, step = w.shape
    m, k = a.shape
    k0 = compiled(design)[1]
    chained = block_scaled(design)
    if k != s * step or step % k0 or (chained and step != k0):
        raise ValueError(
            f'`k` = {k} does not split into {s} slices of '
            f"{'' if chained else 'a multiple of '}{design}'s length {k0}")
    y = torch.zeros(m, n, dtype=torch.float32, device=a.device)
    parts = a.view(m, s, step).transpose(0, 1).contiguous() if s > 1 else a.unsqueeze(0)
    rows = max(1, _ELEMS // n)
    for i in range(0, m, rows):
        blk, ab = y[i:i + rows], parts[:, i:i + rows]
        if chained and scales is not None:
            xs, ys = scales
            for j in range(s):
                _launch(ab[j], w[j], blk, design, xs[j, i:i + rows], ys[j])
        elif scales is not None:
            sx, sw = scales
            for j in range(s):
                blk += _part(ab, w, design, j) * (sx[i:i + rows, j, None] * sw[:, j])
        elif combine == 'tree' and s > 1:
            blk.copy_(_tree(partial(_part, ab, w, design), 0, s))
        else:
            _launch(ab[0], w[0], blk, design)
            for j in range(1, s):
                blk += _part(ab, w, design, j)
    return y


def linear(
    x: torch.Tensor, w: torch.Tensor, design: str, *,
    split_k: int = 1, combine: Combine = 'linear',
) -> torch.Tensor:
    """`x @ w.T` by an unscaled *design*, for `x` `[..., k]` and `w` `[n, k]` (as
    `nn.Linear` stores it): both rounded to its input formats (unscaled, as
    torch casts), the FP32 result.  *split_k* splits `k` into that many
    contiguous slices (:func:`matmul`)."""
    fx, fw = (quant.DTYPES[quant.context(f)] for f in formats(design)[:2])
    held_x, held_w = storage(design)
    a = x.reshape(-1, x.shape[-1]).to(fx).to(held_x)
    y = matmul(a, prepare(w.to(fw), held_w, split_k), design, combine)
    return y.reshape(*x.shape[:-1], w.shape[0])
