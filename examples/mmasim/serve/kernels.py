"""
The BF16 designs' Triton matmuls, as a linear layer's `x @ W.T`.

Each design compiles once with `m`, `n` and `k` symbolic; `k` must be a
multiple of the design's length.  Inputs are rounded to BF16 (RNE) and held
in the kernel's storage dtype (FP32 here); the result is the design's FP32
accumulator.  :func:`linear` does it all per call; a caller reusing weights
(`swap.Run`) calls :func:`prepare` and :func:`round_input` once and
:func:`matmul` per call.
"""

import sys
from collections.abc import Callable
from functools import cache, partial
from pathlib import Path
from typing import Literal

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import compile_triton as ct
from compile import DESIGNS

from fpy2.backend.triton import KernelSource, launch
from fpy2.backend.triton.launcher import _torch_dtype

BF16_DESIGNS: dict[str, tuple[int, int]] = {
    'nv.ampere.bf16.f32': (32, 1),
    'nv.hopper.bf16.f32': (32, 1),
    'amd.cdna2.bf16': (16, 64),
    'amd.cdna2.bf16_1k': (16, 64),
    'amd.cdna3.bf16': (64, 1),
}
"""Each design's fixed tile `(block, block_m)`, the fastest per
`bench/speed.py --best`."""

Combine = Literal['linear', 'tree']

_BUILDS: dict[str, Callable] = dict(DESIGNS)

_ELEMS = 1 << 26
"""Output elements per block of rows: bounds the partials' memory under
`split_k > 1`, and a launch's size."""


def register(name: str, build: Callable, block: int, block_m: int) -> None:
    """Add a BF16 x BF16 -> FP32 design, as `compile.DESIGNS` defines them
    (*build* returns the FPy function and its argument types), run at the
    tile `(block, block_m)`."""
    if name in BF16_DESIGNS:
        raise ValueError(f'{name!r} is already a design')
    _BUILDS[name] = build
    BF16_DESIGNS[name] = (block, block_m)


@cache
def compiled(design: str) -> tuple[KernelSource, int]:
    """*design*'s kernel and the length its `k` must be a multiple of."""
    if design not in BF16_DESIGNS:
        raise ValueError(f'{design!r} is not one of {sorted(BF16_DESIGNS)}')
    kernel, _, arg_types = ct.compile_matmul(_BUILDS[design], None)
    return kernel, ct._length(arg_types[0])


@cache
def storage(design: str) -> tuple[torch.dtype, torch.dtype]:
    """The torch dtypes *design*'s kernel holds its activations and weights in."""
    held = dict(compiled(design)[0].dtypes)
    return _torch_dtype(held[0]), _torch_dtype(held[1])


def round_input(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """*x* `[..., k]` as rows `[m, k]` of BF16 values (RNE), held in *dtype*."""
    return x.reshape(-1, x.shape[-1]).to(torch.bfloat16).to(dtype)


def prepare(w: torch.Tensor, dtype: torch.dtype, split_k: int = 1) -> torch.Tensor:
    """*w* `[n, k]` as BF16 values held in *dtype*, as `[split_k, n, k /
    split_k]` contiguous slices of `k`: a view of *w* at `split_k = 1` when it
    already holds BF16 values in *dtype*."""
    n, k = w.shape
    if split_k < 1 or k % split_k:
        raise ValueError(f'`k` = {k} does not split into {split_k} slices')
    b = w.to(torch.bfloat16).to(dtype)
    if w.dtype == dtype and torch.equal(b, w):
        b = w
    if split_k == 1:
        return b.unsqueeze(0)
    return b.view(n, split_k, k // split_k).transpose(0, 1).contiguous()


def _launch(a: torch.Tensor, b: torch.Tensor, y: torch.Tensor, design: str) -> None:
    """*y* `[m, n]`, zeros, becomes `a @ b.T` by *design*: it is both the
    kernel's accumulator `C` and its output, each program reading its tile
    of `C` before writing it.  `block_m` is capped at the power of two
    `>= m`."""
    block, block_m = BF16_DESIGNS[design]
    m = a.shape[0]
    launch(compiled(design)[0], [a, b, y, y],
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


def matmul(a: torch.Tensor, w: torch.Tensor, design: str, combine: Combine = 'linear') -> torch.Tensor:
    """`a @ w.T` by *design*, FP32 `[m, n]`, for `a` from :func:`round_input`
    and `w` from :func:`prepare` in *design*'s :func:`storage`.

    With `w` in `s > 1` slices, each slice goes through the kernel from a zero
    accumulator and the FP32 partials are summed left to right (`linear`) or
    pairwise (`tree`); one slice accumulates `k` in order through the design
    alone."""
    s, n, step = w.shape
    m, k = a.shape
    k0 = compiled(design)[1]
    if k != s * step or step % k0:
        raise ValueError(
            f'`k` = {k} does not split into {s} slices of a multiple of '
            f"{design}'s length {k0}")
    y = torch.zeros(m, n, dtype=torch.float32, device=a.device)
    parts = a.view(m, s, step).transpose(0, 1).contiguous() if s > 1 else a.unsqueeze(0)
    rows = max(1, _ELEMS // n)
    for i in range(0, m, rows):
        blk, ab = y[i:i + rows], parts[:, i:i + rows]
        if combine == 'tree' and s > 1:
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
    """`x @ w.T` by *design*, for `x` `[..., k]` and `w` `[n, k]` (as
    `nn.Linear` stores it): both rounded to BF16, the FP32 result.  *split_k*
    splits `k` into that many contiguous slices (:func:`matmul`)."""
    held_x, held_w = storage(design)
    y = matmul(round_input(x, held_x), prepare(w, held_w, split_k), design, combine)
    return y.reshape(*x.shape[:-1], w.shape[0])
