"""
`local` and `workloads` on a small random Qwen3.  Needs a GPU and
`transformers`; skipped without either.

    pytest serve/tests
"""

import math

import pytest
import torch

from fpy2.backend.triton import unavailable

_WHY = unavailable()
pytestmark = pytest.mark.skipif(_WHY is not None, reason=_WHY or '')

import kernels
import layers
import local
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

    got = local.evaluate(model, acts, design)['all']
    want = layers.evaluate(model, run, [tokens], [design], local.METRICS)[design]
    for name in first:
        assert got[name].report(local.METRICS) == want[name].report(local.METRICS)

    if 'test.copy' not in kernels.BF16_DESIGNS:
        kernels.register('test.copy', dict(DESIGNS)[design], *kernels.BF16_DESIGNS[design])
    copy = local.evaluate(model, acts, 'test.copy')['all']
    assert all(copy[n].report(local.METRICS) == s.report(local.METRICS) for n, s in got.items())


def test_groups_partition_the_rows(model: torch.nn.Module, tokens: torch.Tensor) -> None:
    """Split by a tag, the groups hold every sampled row once: their counts
    and maxima are the whole's exactly, their sums up to rounding."""
    run = swap.patch(model)
    acts = local.capture(model, run, _seqs(tokens), tokens=10)
    assert len(acts.tags['role']) == acts.inputs[0].shape[0] == 10
    whole = local.evaluate(model, acts, 'amd.cdna2.bf16')['all']
    parts = local.evaluate(model, acts, 'amd.cdna2.bf16', by='role')
    assert set(parts) == {'user', 'assistant'}
    for name, s in whole.items():
        pooled = sum((g[name] for g in parts.values()), layers.Stats())
        assert (pooled.n, pooled.rounded) == (s.n, s.rounded)
        assert (pooled.backward_max, pooled.ulp_max) == (s.backward_max, s.ulp_max)
        assert math.isclose(pooled.err, s.err, rel_tol=1e-9)


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
