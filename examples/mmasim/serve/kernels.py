"""
The BF16 designs' Triton matmuls, as a linear layer's `x @ W.T`.

Each design compiles once (`compile_triton.compile_matmul`) with `m`, `n` and
`k` symbolic, so one kernel serves every layer shape whose `k` is a multiple
of the design's own length.  Activations are rounded to BF16 (RNE), as the
instruction's input format requires, and passed in the kernel's storage for
them (Triton has no BF16 storage here, so FP32 holding BF16 values); the result
is the design's FP32 accumulator, not rounded further.
"""

import sys
from functools import cache
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
"""Each BF16 design and its fixed `(block, block_m)`: the fastest at a
2048-token FFN shape (`bench/speed.py -m 2048 -n 3072 -k 1024 --best`).  A
fixed tile, as autotuning would retune at every new sequence length."""

Combine = Literal['linear', 'tree']

_ELEMS = 1 << 26
"""Output elements per block of rows, so memory does not grow with `split_k`:
at 2048 tokens each of `lm_head`'s partials is 1.2 GB whole."""


@cache
def compiled(design: str) -> tuple[KernelSource, int]:
    """*design*'s kernel and the length its `k` must be a multiple of."""
    if design not in BF16_DESIGNS:
        raise ValueError(f'{design!r} is not one of {sorted(BF16_DESIGNS)}')
    kernel, _, arg_types = ct.compile_matmul(dict(DESIGNS)[design], None)
    return kernel, ct._length(arg_types[0])


def _matmul(x: torch.Tensor, w: torch.Tensor, design: str) -> torch.Tensor:
    """`x @ w.T` for `x` `[m, k]` and `w` `[n, k]` holding BF16 values, by
    *design*: FP32."""
    kernel, _ = compiled(design)
    block, block_m = BF16_DESIGNS[design]
    m, n = x.shape[0], w.shape[0]
    c = torch.zeros(m, n, dtype=torch.float32, device=x.device)
    out = torch.empty(m, n, dtype=torch.float32, device=x.device)
    launch(kernel, [x, w, c, out], block=block, block_m=block_m)
    return out


def linear(
    x: torch.Tensor, w: torch.Tensor, design: str, *,
    split_k: int = 1, combine: Combine = 'linear',
) -> torch.Tensor:
    """`x @ w.T` by *design*, for `x` `[..., k]` and `w` `[n, k]` (as
    `nn.Linear` stores it): `x` rounded to BF16, the FP32 result.

    *split_k* splits `k` into that many contiguous slices, each through the
    kernel from a zero accumulator; the FP32 partials are summed left to
    right (`linear`) or pairwise (`tree`).  `split_k = 1` accumulates `k` in
    order through the design alone."""
    kernel, k0 = compiled(design)
    k = x.shape[-1]
    if split_k < 1 or k % split_k or (k // split_k) % k0:
        raise ValueError(
            f'`k` = {k} does not split into {split_k} slices of a multiple of '
            f"{design}'s length {k0}")
    held = dict(kernel.dtypes)
    a = x.reshape(-1, k).to(torch.bfloat16).to(_torch_dtype(held[0]))
    b = w.to(torch.bfloat16).to(_torch_dtype(held[1]))
    y = torch.empty(a.shape[0], w.shape[0], dtype=torch.float32, device=x.device)
    rows = max(1, _ELEMS // w.shape[0])
    for i in range(0, a.shape[0], rows):
        y[i:i + rows] = _split(a[i:i + rows], b, design, split_k, combine)
    return y.reshape(*x.shape[:-1], w.shape[0])


def _split(a: torch.Tensor, b: torch.Tensor, design: str, split_k: int, combine: Combine) -> torch.Tensor:
    """:func:`linear` on BF16 values in the kernel's storage."""
    step = a.shape[1] // split_k

    def part(s: int) -> torch.Tensor:
        lo = s * step
        return _matmul(a[:, lo:lo + step].contiguous(), b[:, lo:lo + step].contiguous(), design)

    def tree(s: int, n: int) -> torch.Tensor:
        """Slices `s` to `s + n` pairwise, the left half the largest power of
        two below `n`."""
        if n == 1:
            return part(s)
        h = 1 << ((n - 1).bit_length() - 1)
        return tree(s, h).add_(tree(s + h, n - h))

    if combine == 'tree':
        return tree(0, split_k)
    y = part(0)
    for s in range(1, split_k):
        y += part(s)
    return y
