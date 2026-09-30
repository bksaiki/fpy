"""
`quant`: the schemes, and its quantizers, bit for bit against `torchao`'s.

    pytest serve/tests
"""

import pytest
import torch

pytest.importorskip('torchao')


from core import quant

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


def test_a_short_block_is_padded_with_zeros() -> None:
    """Rows short of a whole 128-row block quantize as the block would with
    zeros below them: still lossless."""
    t = _blocks(_E4M3, 448.0, (128, 256), (128, 128), [2.0 ** 3, 2.0 ** -5])[:64]
    assert torch.equal(quant.quantize(t, quant.SCHEMES['fp8-block'].w).dequantize(), t.double())


def test_nvfp4_comes_back_exactly() -> None:
    """With a power-of-two per-tensor scale, blocks of E2M1 values scaled by
    UE4M3 values are lossless too."""
    t = _blocks(_E2M1, 6.0, (2, 64), (1, 16), [448 / 16, 3.5 / 16, 0.25 / 16, 12 / 16])
    q = quant.quantize(t, quant.SCHEMES['nvfp4'].x)
    assert q.tensor.item() == 1 / 16
    assert torch.equal(q.dequantize(), t.double())


@pytest.mark.parametrize('name', quant.NAMES)
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


def _torchao(t: torch.Tensor, op: quant.Operand, tensor: torch.Tensor | None = None,
             amax: torch.Tensor | None = None) -> quant.Quantized:
    """*t* quantized for *op* by `torchao`'s own quantizers: the reference."""
    from torchao.prototype.mx_formats.config import ScaleCalculationMode
    from torchao.prototype.mx_formats.kernels import f4_unpacked_to_f32, unpack_uint4
    from torchao.prototype.mx_formats.mx_tensor import MXTensor
    from torchao.prototype.mx_formats.nvfp4_tensor import (
        nvfp4_quantize,
        per_tensor_amax_to_scale,
    )
    from torchao.quantization import Float8Tensor, PerBlock, PerRow
    from torchao.quantization.quantize_.common.kernel_preference import KernelPreference

    def fp4(packed: torch.Tensor) -> torch.Tensor:
        return f4_unpacked_to_f32(unpack_uint4(packed))

    s, dtype = op.scaling, quant.DTYPES[op.elements]
    if s.fmt == fp.MX_E8M0:
        q = MXTensor.to_mx(t, dtype, s.cols, ScaleCalculationMode.RCEIL)
        return quant.Quantized(op, fp4(q.qdata) if op.elements == fp.MX_E2M1 else q.qdata.float(),
                               q.scale.float())
    if s.tensor:
        if tensor is None:
            tensor = per_tensor_amax_to_scale(t.abs().amax() if amax is None else amax)
        scales, packed = nvfp4_quantize(t, s.cols, tensor)
        return quant.Quantized(op, fp4(packed), scales.float(), tensor)
    padded = torch.nn.functional.pad(t, (0, 0, 0, -t.shape[0] % s.rows))
    q = Float8Tensor.from_hp(padded, float8_dtype=dtype,
                             granularity=PerRow() if s.cols is None else PerBlock([s.rows, s.cols]),
                             hp_value_lb=torch.finfo(torch.float32).tiny,
                             kernel_preference=KernelPreference.TORCH)
    return quant.Quantized(op, q.qdata.float()[:t.shape[0]], q.scale.float())


def _hard(r: int, k: int) -> list[torch.Tensor]:
    """Finite inputs `[r, k]`: twelve binades; zero rows and blocks; FP32's
    subnormals and near its largest; every rounding tie, and the formats'
    largest values."""
    g = torch.Generator().manual_seed(0)
    binades = torch.logspace(-6, 6, k)
    out = [torch.randn(r, k, generator=g) * binades[torch.randperm(k, generator=g)] for _ in range(4)]
    zero = torch.randn(r, k, generator=g)
    zero[0], zero[1, :128] = 0.0, 0.0
    ties = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0, 448.0, 464.0, 480.0, 0.0])
    tie = ties[torch.randint(len(ties), (r, k), generator=g)]
    tie = tie * torch.where(torch.rand(r, k, generator=g) < 0.5, -1.0, 1.0)
    return [*out, zero, torch.randn(r, k, generator=g) * 1e-39, torch.randn(r, k, generator=g) * 1e37, tie]


@pytest.mark.parametrize('name', [n for n in quant.NAMES if n != 'bf16'])
@pytest.mark.parametrize('side, shape', [('x', (7, 512)), ('w', (300, 256))])
def test_quantize_is_torchao_bit_for_bit(name: str, side: str, shape: tuple[int, int]) -> None:
    """Elements, scales and per-tensor scale, every bit (signed zeros too);
    NVFP4 also with a per-row and a given per-tensor scale."""
    op = getattr(quant.scheme(name), side)

    def bits(t: torch.Tensor | None) -> torch.Tensor | None:
        return None if t is None else t.float().view(torch.int32)

    for t in _hard(*shape):
        cases: list[dict[str, torch.Tensor]] = [{}]
        if op.scaling is not None and op.scaling.tensor:
            cases += [{'amax': t.abs().amax(-1, keepdim=True).clamp(min=1.0)},
                      {'tensor': torch.tensor(0.01)}]
        for kw in cases:
            got, want = quant.quantize(t, op, **kw), _torchao(t, op, **kw)
            for a, b in ((got.elements, want.elements), (got.scales, want.scales),
                         (got.tensor, want.tensor)):
                assert (a is None) == (b is None) and (a is None or torch.equal(bits(a), bits(b)))
