"""
MMA-Sim's backend agrees with FPy's kernels bit for bit.  Needs a GPU.

    pytest -n 0 -q evaluation/gsm8k/tests/test_mmasim_matmul.py
"""

import mmasim_matmul
import pytest
import torch
from core import kernels

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a GPU')

SHAPES = [(3584, 1024), (16, 1024), (1024, 2048), (1024, 3584), (6144, 1024), (2048, 1024),
          (512, 1024), (4096, 1024)]
"""Qwen3.5-0.8B's linear layers `(n, k)` but `lm_head`'s."""


def _bf16(*shape: int, g: torch.Generator) -> torch.Tensor:
    """BF16 values over a wide exponent range, with zeros, held in FP32."""
    x = torch.randn(*shape, generator=g, device='cuda')
    x *= torch.exp2(torch.randint(-20, 20, shape, generator=g, device='cuda').float())
    x[torch.rand(shape, generator=g, device='cuda') < 0.05] = 0
    return x.to(torch.bfloat16).float()


def _same(x: torch.Tensor, y: torch.Tensor) -> bool:
    return x.shape == y.shape and torch.equal(x.view(torch.int32), y.view(torch.int32))


@pytest.mark.parametrize('design', list(mmasim_matmul.OPERATIONS))
@pytest.mark.parametrize('n, k', SHAPES)
def test_layer_shapes(design: str, n: int, k: int):
    g = torch.Generator('cuda').manual_seed(n * k)
    a, w = _bf16(3, k, g=g), _bf16(n, k, g=g)
    dtype = kernels.storage(design)
    want = kernels.matmul(a.to(dtype[0]), w.to(dtype[1]), design)
    got = mmasim_matmul.matmul(a.to(torch.bfloat16), w.to(torch.bfloat16), design)
    assert _same(got, want)


def test_row_blocks(monkeypatch: pytest.MonkeyPatch):
    design = 'nv.hopper.bf16.f32'
    g = torch.Generator('cuda').manual_seed(0)
    a, w = _bf16(5, 64, g=g).to(torch.bfloat16), _bf16(8, 64, g=g).to(torch.bfloat16)
    whole = mmasim_matmul.matmul(a, w, design)
    monkeypatch.setattr(mmasim_matmul, '_ELEMS', 2 * 8 * 16)   # 2 rows per block
    assert _same(mmasim_matmul.matmul(a, w, design), whole)


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
    assert _same(*out.values())
    run.mode = 'mmasim:nv.ampere.bf16.f32'
    with pytest.raises(ValueError):
        model(x)
