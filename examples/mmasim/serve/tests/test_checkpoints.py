"""
`checkpoints`: the rules for a model's weights under a scheme, the `Run`
taking them, and the readers against their masters (only when the
checkpoints are downloaded).

    pytest serve/tests
"""

import pytest
import torch
from core import checkpoints, quant, swap

pytest.importorskip('compressed_tensors')


_FP8 = quant.SCHEMES['fp8-row']


def _checkpoint(t: torch.Tensor) -> checkpoints.Checkpoint:
    return checkpoints.Checkpoint('test', _FP8, {'layer': quant.quantize(t, _FP8.w)}, [])


def test_e2m1_unpacks_as_compressed_tensors_does() -> None:
    """Every byte, both nibbles, signed zeros included."""
    from compressed_tensors.compressors.nvfp4.helpers import unpack_fp4_from_uint8

    packed = torch.arange(256, dtype=torch.uint8).view(16, 16)
    got = checkpoints.unpack_e2m1(packed)
    want = unpack_fp4_from_uint8(packed, 16, 32, torch.float32)
    assert torch.equal(got, want) and torch.equal(got.signbit(), want.signbit())


def test_a_checkpoint_is_taken_as_it_is_converted_exactly_or_requantized_on_request() -> None:
    g = torch.Generator().manual_seed(0)
    lossy = _checkpoint(torch.randn(8, 64, generator=g))
    weights, source = checkpoints.weights_for(lossy, _FP8)
    assert source == 'checkpoint' and weights is lossy.weights
    with pytest.raises(ValueError, match='--requantize'):
        checkpoints.weights_for(lossy, swap.BF16)
    assert checkpoints.weights_for(lossy, swap.BF16, requantize=True)[1] == 'requantized'
    # E4M3 values, each row's largest 448 and so its scale 1: BF16 holds them
    e4m3 = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    t = e4m3[torch.isfinite(e4m3)][torch.randint(0, 200, (8, 64), generator=g)]
    t[:, 0] = 448.0
    weights, source = checkpoints.weights_for(_checkpoint(t), swap.BF16)
    assert source == 'converted' and torch.equal(weights['layer'].dequantize(), t.double())


def test_a_run_takes_given_weights_and_leaves_ignored_layers_to_r0(
    model: torch.nn.Module, tokens: torch.Tensor,
) -> None:
    run = swap.patch(model)
    q_proj = model.model.layers[0].self_attn.q_proj
    given = quant.quantize(torch.ones_like(q_proj.weight), _FP8.w)
    swap.give(run, model, _FP8, {'model.layers.0.self_attn.q_proj': given}, ignore=['lm_head'])
    run.scheme, run.mode = _FP8, 'fp8-row-exact'
    assert run.weight(q_proj.weight) is given
    x = torch.randn(4, model.lm_head.in_features, device='cuda')
    assert torch.equal(model.lm_head(x), torch.nn.functional.linear(x, model.lm_head.weight))
    run.scheme, run.mode, run.given, run.ignore = swap.BF16, 'fp32', None, set()


@pytest.mark.parametrize('name, scheme, bound', [
    ('RedHatAI/Qwen3-0.6B-FP8-dynamic', 'fp8-row', 2.0 ** -4),
    ('RedHatAI/Qwen3-0.6B-FP8-BLOCK', 'fp8-block', 2.0 ** -4),
    ('kaitchup/Qwen3-0.6B-NVFP4', 'nvfp4', 2.0 ** -3),
])
def test_a_checkpoint_reads_as_its_master_quantized(name: str, scheme: str, bound: float) -> None:
    """Its scheme from its config; every quantized layer's elements and
    scales in the scheme's formats and, dequantized, within the scheme's
    quantization error of its master, Qwen3-0.6B (a wrong packing or scale
    convention would be far off); `lm_head` left unquantized."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        snapshot_download(name, allow_patterns=['*.json', '*.safetensors'], local_files_only=True)
    except LocalEntryNotFoundError:
        pytest.skip(f'{name} is not downloaded')
    model, ckpt = checkpoints.load(name)
    assert ckpt.scheme.name == scheme and ckpt.ignore == ['lm_head']
    assert len(ckpt.inputs) == (len(ckpt.weights) if scheme == 'nvfp4' else 0)
    masters = checkpoints.master_weights('Qwen/Qwen3-0.6B')
    layers = dict(model.named_modules())
    for layer, q in ckpt.weights.items():
        v, w = q.dequantize(), masters[layer].double()
        assert (v.cpu() - w).norm() / w.norm() < bound, layer
        assert torch.equal(layers[layer].weight, v.float())
    q = next(iter(ckpt.weights.values()))
    for values, ctx in ((q.elements, q.operand.elements), (q.scales, q.operand.scaling.fmt)):
        assert all(float(ctx.round(x)) == x for x in values.unique().tolist())
