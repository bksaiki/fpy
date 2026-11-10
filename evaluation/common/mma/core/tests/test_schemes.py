"""
Each design's scheme (`kernels.scheme`, `quant.derive`), and the element
formats it adds.

    pytest evaluation/common/mma/core/tests/test_schemes.py
"""

import pytest
import torch
from compile import DESIGNS
from core import kernels, quant

import fpy2 as fp

REFUSED = ('amd.cdna1.bf16', 'amd.cdna1.f16')
"""The designs no backend compiles."""


@pytest.mark.parametrize('design', [d for d, _ in DESIGNS if d not in REFUSED])
def test_every_design_takes_its_scheme(design: str) -> None:
    assert kernels.applicable(design, kernels.scheme(design))


@pytest.mark.parametrize('design', list(kernels.TILES))
def test_a_named_schemes_designs_keep_it(design: str) -> None:
    s = kernels.scheme(design)
    assert quant.scheme(s.name) == s and design in kernels.designs(s)


@pytest.mark.parametrize('design, name', [
    ('nv.volta.f16.f16', 'fp16'), ('nv.hopper.tf32.f32', 'tf32'), ('fp64 (fma)', 'fp64'),
    ('nv.hopper.e5m2.f32', 'e5m2-row'), ('nv.hopper.e4m3.e5m2.f16', 'e4m3.e5m2-row'),
    ('nv.blackwell.e2m1.f32', 'e2m1-row'), ('amd.cdna3.bf8', 'e5m2fnuz-row'),
    ('amd.cdna3.fp8.bf8', 'e4m3fnuz.e5m2fnuz-row'), ('nv.blackwell.mx.e5m2', 'mx-e5m2'),
    ('nv.blackwell.mx.e3m2.e2m3', 'mx-e3m2.e2m3'), ('nv.blackwell.e2m3.f16', 'e2m3-row'),
    ('nv.hopper.bf16.f32', 'bf16'), ('nv.ada.e4m3.f16', 'fp8-row'), ('amd.cdna3.fp8', 'fp8-row:fnuz'),
    ('nv.blackwell.mx.e4m3', 'mxfp8'), ('nv.blackwell.nvfp4', 'nvfp4'),
])
def test_scheme_names(design: str, name: str) -> None:
    assert kernels.scheme(design).name == name


def test_tf32_rounds_to_nearest_even() -> None:
    ulp = 2.0 ** -10
    x = torch.tensor([1 + ulp / 2, 1 + 1.5 * ulp, 1 + ulp / 2 + 2 ** -20, -(1 + 1.5 * ulp), 3.0])
    want = torch.tensor([1.0, 1 + 2 * ulp, 1 + ulp, -(1 + 2 * ulp), 3.0])
    q = quant.quantize(x[None], quant.Operand(fp.TF32))
    assert torch.equal(q.elements[0], want)


def test_e2m1_per_row() -> None:
    op = quant.Operand(fp.MX_E2M1, quant.Scaling(fp.FP32, None))
    t = torch.tensor([[3.0, -1.0, 0.25, 0.0], [0.6, 0.1, -0.3, 0.2]])
    q = quant.quantize(t, op)
    assert torch.equal(q.scales[:, 0], torch.tensor([0.5, 0.1]))
    assert q.elements.abs().amax(1).tolist() == [6.0, 6.0]
    assert set(q.elements.abs().flatten().tolist()) <= {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}


@pytest.mark.parametrize('ctx', [fp.MX_E3M2, fp.MX_E2M3])
def test_fp6_per_row(ctx: fp.Context) -> None:
    """Each row scaled to the format's largest value, every element one of its
    values: FP6 is rounded, not cast (torch has no FP6)."""
    op = quant.Operand(ctx, quant.Scaling(fp.FP32, None))
    t = torch.randn(4, 64, generator=torch.Generator().manual_seed(0))
    q = quant.quantize(t, op)
    top = float(ctx.maxval())
    assert q.elements.abs().amax(1).tolist() == [top] * 4
    assert all(float(ctx.round(x)) == x for x in q.elements.flatten().tolist())
