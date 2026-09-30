"""
`kernels.linear` and `kernels.matmul`.  Needs a GPU; skipped without one.

    pytest serve/tests
"""

import gc
import math
import random
import struct

import numpy as np
import pytest
import torch
from core import kernels, quant

from fpy2.backend.triton import unavailable

_WHY = unavailable()
pytestmark = pytest.mark.skipif(_WHY is not None, reason=_WHY or '')


DESIGNS = [d for d in kernels.TILES if not kernels.block_scaled(d)]
SCALED = [d for d in kernels.TILES if kernels.block_scaled(d)]
BF16 = kernels.designs(quant.SCHEMES['bf16'])


def _fuse(monkeypatch: pytest.MonkeyPatch, design: str, on: bool) -> None:
    """Force `kernels.matmul` to fuse *design*'s scaled products, or not, at
    any `m`."""
    monkeypatch.setitem(kernels.FUSED_ROWS, design, 1 << 30 if on else 0)


def _draw(t: object, rng: random.Random, *shape: int) -> torch.Tensor:
    """Hard-case values of *t*'s format, in *shape*."""
    from compile import _fmt, _sample

    fmt = _fmt(t)
    return torch.tensor([_sample(fmt, rng, hard=0.25) for _ in range(math.prod(shape))]).view(shape)


def _rows(t: torch.Tensor) -> torch.Tensor:
    """Scales `[s, rows, per]`, instruction by instruction, as each row's in
    order along `k`."""
    return t.permute(1, 0, 2).reshape(t.shape[1], -1).contiguous()


def _bits(x: float) -> int | str:
    """*x*'s bits, but any NaN as one: a payload is not part of the result."""
    return 'nan' if math.isnan(x) else struct.unpack('<I', struct.pack('<f', x))[0]


@pytest.mark.parametrize('m', [1, 3])
@pytest.mark.parametrize('design', DESIGNS)
def test_agrees_with_the_interpreter(design: str, m: int) -> None:
    """Bit for bit (any NaN as one), on FP32 activations the wrapper rounds to
    the design's format and hard-case weights in its format."""
    from compile import DESIGNS as ALL
    from compile import _fmt, _sample

    f, arg_types = dict(ALL)[design]()
    _, k0 = kernels.compiled(design)
    rng = random.Random(0)
    n, k = 3, 2 * k0
    x = torch.tensor([[rng.gauss(0, 4) for _ in range(k)] for _ in range(m)])
    x[0, :4] = torch.tensor([0.0, -0.0, 3.0e38, 1.0e-40])
    w = torch.tensor([[_sample(_fmt(arg_types[1]), rng, hard=0.25) for _ in range(k)]
                      for _ in range(n)])
    got = kernels.linear(x.cuda(), w.cuda(), design).cpu()
    fx = quant.DTYPES[quant.context(kernels.formats(design)[0])]
    xb = x.to(fx).float().tolist()
    wb = w.float().tolist()
    for i in range(m):
        for j in range(n):
            want = float(f(xb[i], wb[j], 0.0))
            assert _bits(got[i, j].item()) == _bits(want), (i, j, got[i, j].item(), want)


@pytest.mark.parametrize('design', BF16)
def test_split_k_sums_its_partials_in_the_order_asked(design: str) -> None:
    """Four slices, one product each, of 1, 2**24, 1 and -2**24: left to
    right the ones round away at 2**24, pairwise they survive as 1."""
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


@pytest.mark.parametrize('fused', [True, False])
@pytest.mark.parametrize('design', SCALED)
def test_a_chain_of_instructions_agrees_with_the_interpreter(
    design: str, fused: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`k` as three instructions, each accumulating onto the last's result,
    in one kernel or launches: bit for bit (any NaN as one) against the
    interpreter chained so, on hard-case elements and scales."""
    _fuse(monkeypatch, design, fused)
    from compile import DESIGNS as ALL

    f, args = dict(ALL)[design]()
    k0 = kernels.compiled(design)[1]
    per = k0 // kernels.group(design)
    rng = random.Random(0)
    m, n, s = 2, 3, 3
    x, w = _draw(args[0], rng, m, s * k0), _draw(args[1], rng, n, s * k0)
    xs, ys = _draw(args[3], rng, s, m, per), _draw(args[4], rng, s, n, per)
    held = kernels.storage(design)
    got = kernels.matmul(x.cuda().to(held[0]), kernels.prepare(w.cuda(), held[1]), design,
                         scales=(_rows(xs).cuda().to(held[2]), _rows(ys).cuda().to(held[2]))).cpu()
    if per == 1:
        xs, ys = xs[..., 0], ys[..., 0]
    for i in range(m):
        for j in range(n):
            acc = 0.0
            for t in range(s):
                acc = float(f(x[i, t * k0:(t + 1) * k0].tolist(), w[j, t * k0:(t + 1) * k0].tolist(),
                              acc, xs[t, i].tolist(), ys[t, j].tolist()))
            assert _bits(got[i, j].item()) == _bits(acc), (i, j, got[i, j].item(), acc)


@pytest.mark.parametrize('fused', [True, False])
def test_scaled_partials_sum_in_order(fused: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    """Each 128 of `k` from a zero accumulator, scaled and summed left to
    right in FP32, `y + p * (s_x * s_w)`, in one kernel or launches: bit for
    bit (any NaN as one) against the interpreter's partials combined so."""
    from compile import DESIGNS as ALL

    design = 'nv.hopper.e4m3.f32'
    _fuse(monkeypatch, design, fused)
    f, arg_types = dict(ALL)[design]()
    rng = random.Random(0)
    m, n, k = 2, 3, 256
    x, w = _draw(arg_types[0], rng, m, k), _draw(arg_types[0], rng, n, k)
    sx = torch.rand(m, 2) * 4
    sw = torch.rand(n, 2) * 4
    held = kernels.storage(design)
    got = kernels.matmul(x.cuda().to(held[0]), kernels.prepare(w.cuda(), held[1]), design,
                         scales=(sx.cuda(), sw.cuda())).cpu()
    for i in range(m):
        for j in range(n):
            acc = np.float32(0)
            for c in range(2):
                p = np.float32(f(x[i, c * 128:(c + 1) * 128].tolist(),
                                 w[j, c * 128:(c + 1) * 128].tolist(), 0.0))
                acc = acc + p * (np.float32(sx[i, c]) * np.float32(sw[j, c]))
            assert _bits(got[i, j].item()) == _bits(float(acc)), (i, j)


@pytest.mark.parametrize('m', [1, 37])
@pytest.mark.parametrize('design', [*SCALED, 'nv.hopper.e4m3.f32', 'amd.cdna3.fp8'])
def test_one_kernel_is_the_launches(design: str, m: int, monkeypatch: pytest.MonkeyPatch) -> None:
    """Three instructions or two promoted blocks, on hard-case elements (and
    scales, block-scaled), bit for bit (any NaN as one)."""
    from compile import DESIGNS as ALL

    _, args = dict(ALL)[design]()
    rng = random.Random(0)
    held, n = kernels.storage(design), 5
    if kernels.block_scaled(design):
        k0 = kernels.compiled(design)[1]
        per, s = k0 // kernels.group(design), 3
        x, w = _draw(args[0], rng, m, s * k0), _draw(args[1], rng, n, s * k0)
        scales = tuple(_rows(t).cuda().to(held[2])
                       for t in (_draw(args[3], rng, s, m, per), _draw(args[4], rng, s, n, per)))
    else:
        x, w = _draw(args[0], rng, m, 256), _draw(args[1], rng, n, 256)
        scales = ((torch.rand(m, 2) * 4).cuda(), (torch.rand(n, 2) * 4).cuda())
    a, b = x.cuda().to(held[0]), kernels.prepare(w.cuda(), held[1])
    _fuse(monkeypatch, design, True)
    fused = kernels.matmul(a, b, design, scales=scales)
    _fuse(monkeypatch, design, False)
    launched = kernels.matmul(a, b, design, scales=scales)
    assert [*map(_bits, fused.flatten().tolist())] == [*map(_bits, launched.flatten().tolist())]


@pytest.mark.parametrize('split_k', [1, 4])
def test_rows_in_blocks_change_nothing(split_k: int, monkeypatch: pytest.MonkeyPatch) -> None:
    design = 'amd.cdna2.bf16'
    _, k0 = kernels.compiled(design)
    x, w = torch.randn(8, 4 * k0).cuda(), torch.randn(16, 4 * k0).cuda()
    whole = kernels.linear(x, w, design, split_k=split_k)
    monkeypatch.setattr(kernels, '_ELEMS', 3 * 16)
    assert torch.equal(kernels.linear(x, w, design, split_k=split_k), whole)


def test_a_k_the_design_cannot_take_is_refused() -> None:
    design = 'nv.hopper.bf16.f32'
    _, k0 = kernels.compiled(design)
    x, w = torch.zeros(1, k0 + 1).cuda(), torch.zeros(1, k0 + 1).cuda()
    with pytest.raises(ValueError, match='does not split'):
        kernels.linear(x, w, design)
    with pytest.raises(ValueError, match='does not split'):
        kernels.linear(torch.zeros(1, k0).cuda(), torch.zeros(1, k0).cuda(), design, split_k=2)
