"""
`kernels.linear` against the FPy interpreter.  Needs a GPU; skipped without
one.

    pytest serve/tests
"""

import math
import random
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fpy2.backend.triton import unavailable

_WHY = unavailable()
pytestmark = pytest.mark.skipif(_WHY is not None, reason=_WHY or '')

import kernels

DESIGNS = list(kernels.BF16_DESIGNS)


def _bits(x: float) -> int | str:
    """*x*'s bits, but any NaN as one: a payload is not part of the result."""
    return 'nan' if math.isnan(x) else struct.unpack('<I', struct.pack('<f', x))[0]


@pytest.mark.parametrize('m', [1, 3])
@pytest.mark.parametrize('design', DESIGNS)
def test_agrees_with_the_interpreter(design: str, m: int) -> None:
    """Bit for bit (any NaN as one), on FP32 activations the wrapper rounds to
    BF16 and on the hard cases of BF16, at `m` below the design's tile
    height."""
    import compile_triton as ct
    import torch
    from compile import DESIGNS as ALL

    f, arg_types = dict(ALL)[design]()
    _, k0 = kernels.compiled(design)
    rng = random.Random(0)
    n, k = 3, 2 * k0
    x = torch.tensor([[rng.gauss(0, 4) for _ in range(k)] for _ in range(m)])
    x[0, :4] = torch.tensor([0.0, -0.0, 3.0e38, 1.0e-40])
    w = torch.tensor([[ct._sample(ct._fmt(arg_types[1]), rng, hard=0.25) for _ in range(k)]
                      for _ in range(n)]).to(torch.bfloat16)
    got = kernels.linear(x.cuda(), w.cuda(), design).cpu()
    xb = x.to(torch.bfloat16).float().tolist()
    wb = w.float().tolist()
    for i in range(m):
        for j in range(n):
            want = float(f(xb[i], wb[j], 0.0))
            assert _bits(got[i, j].item()) == _bits(want), (i, j, got[i, j].item(), want)


@pytest.mark.parametrize('design', DESIGNS)
def test_split_k_sums_its_partials_in_the_order_asked(design: str) -> None:
    """Four slices, one product each, of 1, 2**24, 1 and -2**24: left to
    right the ones round away at 2**24, pairwise they survive as 1."""
    import torch

    _, k0 = kernels.compiled(design)
    x = torch.zeros(1, 4 * k0)
    w = torch.zeros(1, 4 * k0)
    for s, p in enumerate([1.0, 2.0 ** 24, 1.0, -2.0 ** 24]):
        x[0, s * k0], w[0, s * k0] = p, 1.0
    x, w = x.cuda(), w.cuda()
    assert kernels.linear(x, w, design, split_k=4, combine='linear').item() == 0.0
    assert kernels.linear(x, w, design, split_k=4, combine='tree').item() == 1.0


def test_a_call_frees_what_it_allocates() -> None:
    """Nothing a call allocates outlives it with the garbage collector off,
    as a reference cycle would hold its inputs until a collection."""
    import gc

    import torch

    design = 'amd.cdna2.bf16'
    _, k0 = kernels.compiled(design)
    x, w = torch.randn(8, 4 * k0).cuda(), torch.randn(16, 4 * k0).cuda()
    kernels.linear(x, w, design, split_k=4, combine='tree')
    gc.disable()
    try:
        before = torch.cuda.memory_allocated()
        for combine in ('linear', 'tree'):
            kernels.linear(x, w, design, split_k=4, combine=combine)
        assert torch.cuda.memory_allocated() == before
    finally:
        gc.enable()


def test_a_k_the_design_cannot_take_is_refused() -> None:
    import torch

    design = 'nv.hopper.bf16.f32'
    _, k0 = kernels.compiled(design)
    x, w = torch.zeros(1, k0 + 1).cuda(), torch.zeros(1, k0 + 1).cuda()
    with pytest.raises(ValueError, match='does not split'):
        kernels.linear(x, w, design)
    with pytest.raises(ValueError, match='does not split'):
        kernels.linear(torch.zeros(1, k0).cuda(), torch.zeros(1, k0).cuda(), design, split_k=2)
