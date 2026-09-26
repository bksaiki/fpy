"""
`quant`: the schemes, and `torchao`'s quantizers as `quant` reads them.

    pytest serve/tests
"""

import pytest
import torch

pytest.importorskip('torchao')

import quant

import fpy2 as fp

_E4M3 = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
_E4M3 = _E4M3[torch.isfinite(_E4M3)]
_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _blocks(values: torch.Tensor, top: float, shape: tuple[int, int], block: tuple[int, int],
            factors: list[float]) -> torch.Tensor:
    """A `shape` tensor of *values* (signed), each block's largest `top`,
    and the blocks scaled by *factors* in turn, repeating."""
    g = torch.Generator().manual_seed(0)
    t = values[torch.randint(len(values), shape, generator=g)]
    t = t * torch.where(torch.rand(shape, generator=g) < 0.5, -1.0, 1.0)
    br, bc = block
    for i in range(shape[0] // br * (shape[1] // bc)):
        r, c = divmod(i, shape[1] // bc)
        blk = t[r * br:(r + 1) * br, c * bc:(c + 1) * bc]
        blk[0, 0] = top
        blk *= factors[i % len(factors)]
    return t


@pytest.mark.parametrize('name, operand, values, top, shape, block', [
    ('bf16', 'x', _E4M3, 448.0, (2, 64), (1, 64)),
    ('fp8-row', 'x', _E4M3, 448.0, (4, 64), (1, 64)),
    ('fp8-block', 'x', _E4M3, 448.0, (2, 256), (1, 128)),
    ('fp8-block', 'w', _E4M3, 448.0, (256, 256), (128, 128)),
    ('mxfp8', 'x', _E4M3, 448.0, (2, 64), (1, 32)),
    ('mxfp4', 'x', _E2M1, 6.0, (2, 64), (1, 32)),
])
def test_lossless_inputs_come_back_exactly(name: str, operand: str, values: torch.Tensor,
                                           top: float, shape: tuple[int, int],
                                           block: tuple[int, int]) -> None:
    """Blocks of the format's values, each scaled by a power of two, quantize
    losslessly: elements, scales and their layout read back exactly."""
    t = _blocks(values, top, shape, block, [2.0 ** j for j in (3, -5, 0, 7)])
    op = getattr(quant.SCHEMES[name], operand)
    assert torch.equal(quant.quantize(t, op).dequantize(), t.double())


def test_nvfp4_comes_back_exactly() -> None:
    """With a power-of-two per-tensor scale, blocks of E2M1 values scaled by
    UE4M3 values are lossless too."""
    t = _blocks(_E2M1, 6.0, (2, 64), (1, 16), [448 / 16, 3.5 / 16, 0.25 / 16, 12 / 16])
    q = quant.quantize(t, quant.SCHEMES['nvfp4'].x)
    assert q.tensor.item() == 1 / 16
    assert torch.equal(q.dequantize(), t.double())


@pytest.mark.parametrize('name', [*quant.SCHEMES, 'fp8-row:fnuz', 'fp8-block:fnuz'])
def test_elements_and_scales_are_in_the_schemes_formats(name: str) -> None:
    """On values over twelve binades, every element and scale is exactly
    representable in the FPy format the scheme names."""
    s = quant.scheme(name)
    g = torch.Generator().manual_seed(1)
    for op, shape in ((s.x, (4, 256)), (s.w, (256, 256))):
        t = torch.randn(shape, generator=g) * 2.0 ** torch.randint(-6, 6, shape, generator=g)
        q = quant.quantize(t, op)
        parts = [(q.elements, op.elements)]
        if q.scales is not None:
            parts.append((q.scales, op.scaling.fmt))
        for values, ctx in parts:
            for v in values.unique().tolist():
                assert float(ctx.round(v)) == v, (name, v)


def test_only_the_software_scaled_fp8_schemes_take_fnuz() -> None:
    assert quant.scheme('fp8-row:fnuz').w.elements == fp.S1E4M3
    for bad in ('mxfp8:fnuz', 'bf16:fnuz', 'fp8-row:e5m2'):
        with pytest.raises(ValueError):
            quant.scheme(bad)
