"""
MMA-Sim's backend agrees with FPy's kernels bit for bit, on every design
both compute.  Needs a GPU; compiles every design once, cached under
pytest's cache.

    pytest -n 0 -q evaluation/gsm8k/tests/test_mmasim_matmul.py
"""

import mmasim_matmul
import pytest
import torch
from core import kernels, quant, swap

import fpy2 as fp

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a GPU')

SHAPES = [(16, 1024), (16, 3584)]
"""`(n, k)`: Qwen3.5-0.8B's shortest and longest `k`.  A column's result
depends only on `k`, so few columns suffice."""


@pytest.fixture(scope='session')
def compiled(request: pytest.FixtureRequest) -> None:
    """Every design's kernel, compiled 8 at a time, cached across sessions."""
    kernels.CACHE = request.config.cache.mkdir('kernels')
    assert kernels.compile_designs(mmasim_matmul.DESIGNS, 8) == {}


def _values(*shape: int, g: torch.Generator, top: int = 12) -> torch.Tensor:
    """BF16 values of exponents from -12 to *top*, with zeros, held in FP32."""
    x = torch.randn(*shape, generator=g, device='cuda')
    x *= torch.exp2(torch.randint(-12, top, shape, generator=g, device='cuda').float())
    x[torch.rand(shape, generator=g, device='cuda') < 0.05] = 0
    return x.to(torch.bfloat16).float()


_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _moderate(op: quant.Operand, rows: int, k: int, g: torch.Generator) -> quant.Quantized:
    """*op*'s elements of magnitude at most 16 or so, its scales 1: an FP16
    accumulator stays finite over `k`, where elements scaled to the format's
    range would overflow it."""
    fmt = op.elements
    if fmt == fp.MX_E2M1:
        e = _E2M1.cuda()[torch.randint(8, (rows, k), generator=g, device='cuda')]
        e = e * torch.where(torch.rand(rows, k, generator=g, device='cuda') < 0.5, -1.0, 1.0)
    elif fmt in quant._SUBBYTE:   # FP6: no torch dtype
        top = float(fmt.maxval())
        e = quant._subbyte(_values(rows, k, g=g, top=2).clamp(-top, top), fmt)
    else:
        e = _values(rows, k, g=g, top=2).to(quant.DTYPES[fmt]).float()
    return quant.Quantized(op, e, torch.ones(rows, 1, device='cuda'))


def _both(design: str, m: int, n: int, k: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """MMA-Sim's and the kernel's accumulator on the same quantized operands."""
    g = torch.Generator('cuda').manual_seed(seed)
    scheme = kernels.scheme(design)
    if mmasim_matmul.output(design) == torch.float16 and scheme.applied != 'instruction':
        qa, qw = _moderate(scheme.x, m, k, g), _moderate(scheme.w, n, k, g)
    else:
        qa = quant.quantize(_values(m, k, g=g), scheme.x)
        qw = quant.quantize(_values(n, k, g=g), scheme.w)
    want = swap.accumulate(design, scheme, qa, qw, qw.elements.to(kernels.storage(design)[1]))
    return mmasim_matmul.accumulate(design, scheme, qa, qw), want


@pytest.mark.parametrize('design', mmasim_matmul.DESIGNS)
@pytest.mark.parametrize('n, k', SHAPES)
def test_layer_shapes(design: str, n: int, k: int, compiled: None):
    got, want = _both(design, 3, n, k, n * k)
    assert got.dtype == want.dtype == mmasim_matmul.output(design)
    assert bool(mmasim_matmul.same(got, want).all())
    assert got.isfinite().float().mean() > 0.9 and (got != 0).float().mean() > 0.5


@pytest.mark.parametrize('design', ['nv.hopper.bf16.f32', 'amd.cdna2.bf16', 'fp64 (fma)',
                                    'nv.ampere.tf32.f32', 'nv.blackwell.mx.e4m3', 'nv.blackwell.nvfp4'])
def test_blocks_change_nothing(design: str, compiled: None, monkeypatch: pytest.MonkeyPatch):
    """Blocks of rows and columns, as `lm_head` needs, give the same bits."""
    whole, _ = _both(design, 5, 96, 128, 0)
    monkeypatch.setattr(mmasim_matmul, '_BUDGET', 2 * 32 * mmasim_matmul._bytes(design, 128))
    blocked, want = _both(design, 5, 96, 128, 0)
    assert bool(mmasim_matmul.same(blocked, whole).all() & mmasim_matmul.same(blocked, want).all())


def test_run_modes():
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(1024, 2048, bias=False),
                                torch.nn.Linear(2048, 1024, bias=False)).cuda()
    run = mmasim_matmul.patch(model)
    x = torch.randn(2, 3, 1024, device='cuda')
    out = {}
    with torch.no_grad():
        for mode in ('nv.hopper.bf16.f32', 'mmasim:nv.hopper.bf16.f32'):
            run.mode = mode
            out[mode] = model(x)
    assert bool(mmasim_matmul.same(*out.values()).all())
    run.mode = 'mmasim:nv.hopper.e4m3.f32'   # not under bf16
    with pytest.raises(ValueError):
        model(x)


def test_check():
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(1024, 2048, bias=False),
                                torch.nn.Linear(2048, 1024, bias=False)).cuda()
    run = mmasim_matmul.patch(model)
    run.mode, run.calls = 'mmasim:nv.hopper.bf16.f32', []
    with torch.no_grad():
        model(torch.randn(3, 1024, device='cuda'))
    assert len(run.calls) == 2
    run.calls[1][3].view(torch.int32)[0, 0] ^= 1
    run.calls[0][3][0, :2] = torch.nan
    assert run.check() == {'calls': 2, 'outputs': 3 * 2048 + 3 * 1024, 'differing': 3,
                           'finite': 3 * 2048 + 3 * 1024 - 2}
    assert run.calls == [] and run.mode == 'mmasim:nv.hopper.bf16.f32'


def test_fpy_calls_are_recorded():
    model = torch.nn.Sequential(torch.nn.Linear(1024, 2048, bias=False)).cuda()
    run = mmasim_matmul.patch(model)
    run.calls = []
    with torch.no_grad():
        for run.mode in ('fp32', 'nv.hopper.bf16.f32'):
            model(torch.randn(3, 1024, device='cuda'))
    ((qa, _, w, acc),) = run.calls
    assert qa.elements.shape == (3, 1024) and w is model[0].weight and acc is None


def test_release_drops_every_held_weight():
    model = torch.nn.Sequential(torch.nn.Linear(1024, 2048, bias=False)).cuda()
    run = mmasim_matmul.patch(model)
    with torch.no_grad():
        for run.mode in ('nv.hopper.bf16.f32', 'mmasim:nv.hopper.bf16.f32'):
            model(torch.randn(3, 1024, device='cuda'))
    assert run._weights and run._prepared
    run.release()
    assert not run._weights and not run._prepared
    assert run._input is None


@pytest.mark.parametrize('design', mmasim_matmul.DESIGNS)
def test_tuned_is_bit_for_bit(design: str, compiled: None):
    """Tuned (no host waits, compiled, BF16 in FP32) gives MMA-Sim's own bits."""
    released = [_both(design, 3, 16, 1024, seed)[0] for seed in (0, 1)]
    mmasim_matmul.tune()
    try:
        tuned = [_both(design, 3, 16, 1024, seed)[0] for seed in (0, 1)]
    finally:
        mmasim_matmul.tune(False)
    assert all(bool(mmasim_matmul.same(t, r).all()) for t, r in zip(tuned, released, strict=True))
    assert not bool(mmasim_matmul.same(*released).all())
