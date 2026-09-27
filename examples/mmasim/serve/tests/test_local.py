"""
`local` and `workloads` on a small random Qwen3.  Needs a GPU and
`transformers`; skipped without either.

    pytest serve/tests
"""

import math
from functools import partial

import pytest
import torch

from fpy2.backend.triton import unavailable

_WHY = unavailable()
pytestmark = pytest.mark.skipif(_WHY is not None, reason=_WHY or '')

import kernels
import local
import metrics
import quant
import swap
import workloads

_WORDS = ['<|im_start|>', 'user', '\n', 'Hi', '<|im_end|>', '\n',
          '<|im_start|>', 'assistant', '\n', '<think>', '\n\n', '</think>', '\n\n',
          'Hel', 'lo', '<|im_end|>']
"""A chat-template sequence, one word a token, as Qwen's template writes it."""


class _Tok:
    """A tokenizer over :data:`_WORDS`, token `i` its `i`th word."""

    def convert_tokens_to_ids(self, t: str) -> int:
        return _WORDS.index(t)

    def decode(self, ids: list[int]) -> str:
        return ''.join(_WORDS[i] for i in ids)


def _seqs(tokens: torch.Tensor) -> list[workloads.Sequence]:
    ids = tokens[0].tolist()
    half = len(ids) // 2
    return [workloads.Sequence(ids, {'role': ['user'] * half + ['assistant'] * (len(ids) - half)})]


def _own_inputs(model: torch.nn.Module, run: swap.Run, tokens: torch.Tensor, design: str,
                names: list[str]) -> dict[str, metrics.Stats]:
    """*design*'s local metrics *names* with the model run through it, each
    linear layer on its own inputs."""
    stats: dict[str, metrics.Stats] = {}

    def record(name: str, layer: torch.nn.Linear, inputs: tuple[torch.Tensor],
               y: torch.Tensor) -> None:
        x = inputs[0].reshape(-1, inputs[0].shape[-1])
        metrics.local(stats.setdefault(name, metrics.Stats()), names,
                      run.quantize(x, layer.weight), run.weight(layer.weight),
                      y.reshape(-1, y.shape[-1]), (x, layer.weight))

    handles = [m.register_forward_hook(partial(record, n)) for n, m in model.named_modules()
               if isinstance(m, torch.nn.Linear)]
    run.mode = design
    try:
        with torch.no_grad():
            model(tokens)
    finally:
        run.mode = 'fp32'
        for h in handles:
            h.remove()
    return stats


def test_cached_inputs_give_the_model_run_where_inputs_agree(
    model: torch.nn.Module, tokens: torch.Tensor,
) -> None:
    """One input per distinct tensor; where no linear layer comes before
    (block 0's `q/k/v_proj`), a design evaluated on cached inputs has the
    local metrics of the whole model run through it; and a design registered
    under a new name evaluates as the original."""
    from compile import DESIGNS

    design = 'amd.cdna2.bf16'
    run = swap.patch(model)
    acts = local.capture(model, run, _seqs(tokens))
    assert len(acts.inputs) == 4 * 2 + 1
    first = [f'model.layers.0.self_attn.{p}_proj' for p in 'qkv']
    assert len({acts.index[n] for n in first}) == 1

    # unquantized inputs are FP32 there, BF16 here
    same = [m for m in metrics.METRICS if m != 'quantization']
    got = local.evaluate(model, run, acts, design)['quantized']['all']
    want = _own_inputs(model, run, tokens, design, same)
    for name in first:
        assert got[name].report(same) == want[name].report(same)

    if 'test.copy' not in kernels.TILES:
        kernels.register('test.copy', dict(DESIGNS)[design], *kernels.TILES[design])
    copy = local.evaluate(model, run, acts, 'test.copy')['quantized']['all']
    assert all(copy[n].report(metrics.METRICS) == s.report(metrics.METRICS) for n, s in got.items())


def test_groups_partition_the_rows(model: torch.nn.Module, tokens: torch.Tensor) -> None:
    """Split by a tag, the groups hold every sampled row once: their counts
    and maxima are the whole's exactly, their sums up to rounding."""
    run = swap.patch(model)
    acts = local.capture(model, run, _seqs(tokens), tokens=10)
    assert len(acts.tags['role']) == acts.inputs[0].shape[0] == 10
    whole = local.evaluate(model, run, acts, 'amd.cdna2.bf16')['quantized']['all']
    parts = local.evaluate(model, run, acts, 'amd.cdna2.bf16', by='role')['quantized']
    assert set(parts) == {'user', 'assistant'}
    for name, s in whole.items():
        pooled = sum((g[name] for g in parts.values()), metrics.Stats())
        assert (pooled.n, pooled.rounded) == (s.n, s.rounded)
        assert (pooled.backward_max, pooled.ulp_max) == (s.backward_max, s.ulp_max)
        assert math.isclose(pooled.err, s.err, rel_tol=1e-9)


@pytest.mark.parametrize('name', ['fp8-row', 'fp8-block', 'nvfp4'])
def test_a_scheme_measures_its_quantization_apart_from_the_design(
    name: str, model: torch.nn.Module, tokens: torch.Tensor,
) -> None:
    """Under a quantizing scheme the quantization errs, and against the
    unquantized operands the design's output errs by more than against the
    quantized ones; under `bf16`, on BF16 inputs and weights, quantizing
    costs nothing and the two references agree; a design the scheme does not
    apply to is refused, as is splitting `k`'s own blocks further."""
    run = swap.patch(model)
    scheme = run.scheme = quant.SCHEMES[name]
    acts = local.capture(model, run, _seqs(tokens))
    design = kernels.designs(scheme)[0]
    own, both = (sum(r['all'].values(), metrics.Stats()).report(metrics.METRICS)
                 for r in local.evaluate(model, run, acts, design).values())
    assert own['quantization'] > 0 and own['rounded'] < 1
    assert both['normwise'] > own['normwise'] > 0
    with pytest.raises(ValueError, match='does not take'):
        local.evaluate(model, run, acts, 'amd.cdna2.bf16')
    if scheme.applied != 'epilogue':
        run.split_k = 2
        with pytest.raises(ValueError, match='its own blocks'):
            local.evaluate(model, run, acts, design)
        run.split_k = 1
    run.scheme = swap.BF16
    own, both = (sum(r['all'].values(), metrics.Stats()).report(metrics.METRICS)
                 for r in local.evaluate(model, run, local.capture(model, run, _seqs(tokens)),
                                         'amd.cdna2.bf16').values())
    assert own.pop('quantization') == 0 and own.pop('rounded') > 0
    assert own == {m: v for m, v in both.items() if m in own}


def test_sample_is_fixed_and_in_order() -> None:
    seqs = [workloads.Sequence([0] * n, {}) for n in (5, 0, 7)]
    keep = local.sample(seqs, 6)
    assert keep == local.sample(seqs, 6) and sum(map(len, keep)) == 6 and keep[1] == []
    assert all(k == sorted(k) and all(0 <= p < len(s.ids) for p in k) for k, s in zip(keep, seqs))
    assert local.sample(seqs, None) == [list(range(5)), [], list(range(7))]


def test_roles_follow_the_chat_template() -> None:
    """Markers, role headers and the empty think block are `template`; a
    message's content is its role's, up to its end marker."""
    ids = [_WORDS.index(w) if w.startswith('<') else i for i, w in enumerate(_WORDS)]
    assert workloads.roles(ids, _Tok()) == [
        'template', 'template', 'template', 'user', 'template', 'template',
        'template', 'template', 'template', 'template', 'template', 'template', 'template',
        'assistant', 'assistant', 'template']
