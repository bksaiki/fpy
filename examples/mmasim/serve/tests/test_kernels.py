"""
`kernels.matmul`.  Needs a GPU; skipped without one.

    pytest serve/tests
"""

import gc
import math
import random
import struct

import pytest
import torch
from core import kernels, quant

from fpy2.backend.triton import unavailable

_WHY = unavailable()
pytestmark = pytest.mark.skipif(_WHY is not None, reason=_WHY or '')


DESIGNS = [d for d in kernels.TILES if not kernels.block_scaled(d)]
SCALED = [d for d in kernels.TILES if kernels.block_scaled(d)]


def linear(x: torch.Tensor, w: torch.Tensor, design: str) -> torch.Tensor:
    """`x @ w.T` by an unscaled *design*, for `x` `[..., k]` and `w` `[n, k]`
    (as `nn.Linear` stores it): both rounded to its input formats (as torch
    casts), the FP32 result."""
    fx, fw = (quant.DTYPES[quant.context(f)] for f in kernels.formats(design)[:2])
    held_x, held_w = kernels.storage(design)
    a = x.reshape(-1, x.shape[-1]).to(fx).to(held_x)
    y = kernels.matmul(a, w.to(fw).to(held_w), design)
    return y.reshape(*x.shape[:-1], w.shape[0])


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
    got = linear(x.cuda(), w.cuda(), design).cpu()
    fx = quant.DTYPES[quant.context(kernels.formats(design)[0])]
    xb = x.to(fx).float().tolist()
    wb = w.float().tolist()
    for i in range(m):
        for j in range(n):
            want = float(f(xb[i], wb[j], 0.0))
            assert _bits(got[i, j].item()) == _bits(want), (i, j, got[i, j].item(), want)


def test_a_call_frees_what_it_allocates() -> None:
    """Nothing a call allocates outlives it with the garbage collector off,
    as a reference cycle would hold its inputs until a collection."""
    design = 'amd.cdna2.bf16'
    _, k0 = kernels.compiled(design)
    x, w = torch.randn(8, 4 * k0).cuda(), torch.randn(16, 4 * k0).cuda()
    linear(x, w, design)
    gc.disable()
    try:
        before = torch.cuda.memory_allocated()
        linear(x, w, design)
        assert torch.cuda.memory_allocated() == before
    finally:
        gc.enable()


@pytest.mark.parametrize('design', SCALED)
def test_a_chain_of_instructions_agrees_with_the_interpreter(design: str) -> None:
    """`k` as three instructions in one kernel, each accumulating onto the
    last's result: bit for bit (any NaN as one) against the interpreter
    chained so, on hard-case elements and scales."""
    from compile import DESIGNS as ALL

    f, args = dict(ALL)[design]()
    k0 = kernels.compiled(design)[1]
    per = k0 // kernels.group(design)
    rng = random.Random(0)
    m, n, s = 2, 3, 3
    x, w = _draw(args[0], rng, m, s * k0), _draw(args[1], rng, n, s * k0)
    xs, ys = _draw(args[3], rng, s, m, per), _draw(args[4], rng, s, n, per)
    held = kernels.storage(design)
    got = kernels.matmul(x.cuda().to(held[0]), w.cuda().to(held[1]), design,
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


def test_rows_in_blocks_change_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    design = 'amd.cdna2.bf16'
    _, k0 = kernels.compiled(design)
    x, w = torch.randn(8, 4 * k0).cuda(), torch.randn(16, 4 * k0).cuda()
    whole = linear(x, w, design)
    monkeypatch.setattr(kernels, '_ELEMS', 3 * 16)
    assert torch.equal(linear(x, w, design), whole)


def test_a_k_the_design_cannot_take_is_refused() -> None:
    design = 'nv.hopper.bf16.f32'
    _, k0 = kernels.compiled(design)
    x, w = torch.zeros(1, k0 + 1).cuda(), torch.zeros(1, k0 + 1).cuda()
    with pytest.raises(ValueError, match='is not a multiple'):
        linear(x, w, design)


def test_scales_not_one_per_instruction_are_refused() -> None:
    """A kernel cannot raise, so a scale list too short to cover `k` is the
    host's to refuse."""
    design = 'nv.blackwell.mx.e4m3'
    held, (_, k0) = kernels.storage(design), kernels.compiled(design)
    x, w = torch.zeros(2, 2 * k0).cuda().to(held[0]), torch.zeros(3, 2 * k0).cuda().to(held[1])
    with pytest.raises(ValueError, match='takes scales'):
        kernels.matmul(x, w, design, scales=(torch.ones(2, 1).cuda().to(held[2]),
                                             torch.ones(3, 1).cuda().to(held[2])))


def test_generated_designs_register_and_agree(registry: None) -> None:
    """Registered from the generator, compiled together in worker processes
    (their builds pickle), each runs through its kernel bit for bit (any
    NaN as one) against the generator's function; its name is its
    parameters'."""
    from dataclasses import replace

    import generator
    from generator.catalog import CATALOG

    import fpy2 as fp

    remixes = [replace(CATALOG['nv.hopper.e4m3.f32'], F=17, rho=fp.IEEEContext(8, 9 + 17, fp.RM.RTZ)),
               replace(CATALOG['amd.cdna2.bf16_1k'], L=16, k=32)]
    names = [kernels.register_design(d) for d in remixes]
    for d, name in zip(remixes, names):
        assert name == generator.name(d) and kernels.REGISTERED[name] == generator.about(d)
    assert kernels.compile_designs(names, jobs=2) == {}
    rng = random.Random(0)
    for d, name in zip(remixes, names):
        f, arg_types = generator.build(d)
        m, n, k = 2, 3, 2 * d.k
        x = _draw(arg_types[0], rng, m, k)
        w = _draw(arg_types[1], rng, n, k)
        got = linear(x.cuda(), w.cuda(), name).cpu()
        for i in range(m):
            for j in range(n):
                acc = 0.0
                for t in range(0, k, d.k):
                    acc = float(f(x[i, t:t + d.k].tolist(), w[j, t:t + d.k].tolist(), acc))
                assert _bits(got[i, j].item()) == _bits(acc), (name, i, j, got[i, j].item(), acc)
