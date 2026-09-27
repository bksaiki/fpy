"""
Quantized checkpoints (`compressed-tensors`), and the weights a model is
evaluated with under a scheme.

:func:`load` builds a checkpoint's model from its own config and tensors:
each quantized linear layer's weight is its dequantized values in FP32 (the
weight R0 runs), and its stored elements and scales are kept in the
checkpoint's scheme, with NVFP4's static per-tensor activation scales.
:func:`weights_for` applies the rules of `docs/todos/mmasim-quantized.md`: a
checkpoint in the scheme as it is; one in another scheme converted if that
is exact, else only when asked to requantize; a master left to RTN.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import quant
import torch
from compressed_tensors.compressors.nvfp4.helpers import unpack_fp4_from_uint8


@dataclass
class Checkpoint:
    name: str
    scheme: quant.Scheme
    weights: dict[str, quant.Quantized]
    """Each quantized linear layer's weight, by layer name."""
    ignore: list[str]
    """The linear layers it leaves unquantized."""
    inputs: dict[str, torch.Tensor] = field(default_factory=dict)
    """NVFP4's static per-tensor activation scales, by layer name (their
    stored reciprocals inverted, as `torchao` takes them)."""


def _scheme(qc: dict[str, Any]) -> quant.Scheme:
    """The scheme a `compressed-tensors` quantization config describes."""
    group = next(iter(qc['config_groups'].values()))
    w = group['weights']
    if w['num_bits'] == 4 and w.get('group_size') == 16:
        return quant.SCHEMES['nvfp4']
    if w['num_bits'] == 8 and w['type'] == 'float' and w['strategy'] == 'channel':
        return quant.SCHEMES['fp8-row']
    if w['num_bits'] == 8 and w['strategy'] == 'block' and w['block_structure'] == [128, 128]:
        return quant.SCHEMES['fp8-block']
    raise ValueError(f'no scheme for weights {w}')


def _weight(scheme: quant.Scheme, t: dict[str, torch.Tensor], name: str) -> quant.Quantized:
    """Layer *name*'s stored weight in *scheme*, on the GPU."""
    if scheme.name == 'nvfp4':
        n, half = t[f'{name}.weight_packed'].shape
        elements = unpack_fp4_from_uint8(t[f'{name}.weight_packed'].cuda(), n, 2 * half,
                                         torch.float32)
        return quant.Quantized(scheme.w, elements, t[f'{name}.weight_scale'].cuda().float(),
                               1 / t[f'{name}.weight_global_scale'].cuda().float().reshape(()))
    return quant.Quantized(scheme.w, t[f'{name}.weight'].cuda().float(),
                           t[f'{name}.weight_scale'].cuda().float())


def is_checkpoint(name: str) -> bool:
    from transformers import AutoConfig

    return getattr(AutoConfig.from_pretrained(name), 'quantization_config', None) is not None


def load(name: str) -> tuple[torch.nn.Module, Checkpoint]:
    """*name*'s model, in FP32 on the GPU, its quantized weights dequantized,
    and the checkpoint."""
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForCausalLM

    path = Path(snapshot_download(name, allow_patterns=['*.json', '*.safetensors']))
    qc = json.loads((path / 'config.json').read_text())['quantization_config']
    scheme = _scheme(qc)
    cfg = AutoConfig.from_pretrained(path)
    del cfg.quantization_config
    model = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32)
    tensors = {k: v for f in sorted(path.glob('*.safetensors')) for k, v in load_file(f).items()}
    ckpt = Checkpoint(name, scheme, {}, qc.get('ignore', []))
    state: dict[str, torch.Tensor] = {}
    for layer, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear) or f'{layer}.weight_scale' not in tensors:
            continue
        q = ckpt.weights[layer] = _weight(scheme, tensors, layer)
        state[f'{layer}.weight'] = q.dequantize().float()
        if f'{layer}.input_global_scale' in tensors:
            ckpt.inputs[layer] = 1 / tensors[f'{layer}.input_global_scale'].cuda().float().reshape(())
    stored = {f'{p}.{s}' for p in ckpt.weights for s in (
        'weight', 'weight_packed', 'weight_scale', 'weight_global_scale', 'input_global_scale')}
    state.update({k: v.float() for k, v in tensors.items() if k not in stored})
    missing, unexpected = model.load_state_dict(state, strict=False)
    tied = {'lm_head.weight'} if cfg.tie_word_embeddings else set()
    if unexpected or set(missing) - tied:
        raise ValueError(f'{name}: unexpected {unexpected}, missing {missing}')
    model.tie_weights()
    return model.cuda().eval(), ckpt


def weights_for(ckpt: Checkpoint, scheme: quant.Scheme,
                requantize: bool = False) -> tuple[dict[str, quant.Quantized], str]:
    """*ckpt*'s weights under *scheme*, by layer name, and their source:
    `checkpoint` (in *scheme* already), `converted` (exactly), or
    `requantized` (RTN from its dequantized weights, lossy; only if
    *requantize*)."""
    if ckpt.scheme.name == scheme.name:
        return ckpt.weights, 'checkpoint'
    out, lossy = {}, 0
    for layer, q in ckpt.weights.items():
        v = q.dequantize()
        out[layer] = r = quant.quantize(v.float(), scheme.w)
        lossy += not torch.equal(r.dequantize(), v)
    if lossy and not requantize:
        raise ValueError(f'{ckpt.name} is {ckpt.scheme.name}; {scheme.name} would requantize '
                         f'{lossy} of its layers lossily (--requantize)')
    return out, 'requantized' if lossy else 'converted'


def base_model(name: str) -> str | None:
    """The model *name*'s card says it was made from, if it says."""
    from huggingface_hub import HfApi

    card = HfApi().model_info(name).card_data
    base = card.get('base_model') if card else None
    return base[0] if isinstance(base, list) else base


def master_weights(name: str) -> dict[str, torch.Tensor]:
    """The unquantized model *name*'s linear weights, by layer name, in FP32
    on the host."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)
    return {n: m.weight.detach() for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)}
