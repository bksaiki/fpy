"""
Batched greedy generation that prefills the prompts' shared prefix once.

Qwen3.5's linear-attention layers decay their state at padding inside a
sequence, so prompts are not padded while prefilling.  The shared prefix is
prefilled once; each prompt's rest, alone, from a copy of its cache.  The
caches then merge into one batch, attention keys and values left-padded and
masked, to decode together.
"""

import copy
import time
from collections.abc import Collection
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import DynamicCache
from transformers.cache_utils import LinearAttentionCacheLayerMixin


@dataclass
class Generated:
    tokens: list[list[int]]
    """Each prompt's tokens, through its first stop token."""
    prefill: float
    """Seconds to each prompt's first token."""
    decode: float
    """Seconds for the rest."""


def _clock() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def shared(prompts: list[list[int]]) -> int:
    """The length of *prompts*' common prefix, leaving each at least one
    token of its own."""
    n = min(len(p) for p in prompts) - 1
    return next((i for i in range(n) if len({p[i] for p in prompts}) > 1), n)


def _forward(model: torch.nn.Module, tokens: torch.Tensor, cache: DynamicCache,
             positions: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """The next token after each row of *tokens* `[b, t]`."""
    out = model(tokens, attention_mask=mask, position_ids=positions, past_key_values=cache,
                use_cache=True, logits_to_keep=1)
    return out.logits[:, -1].argmax(-1)


def _merge(caches: list[DynamicCache], lengths: list[int]) -> DynamicCache:
    """*caches*, of one sequence each, of *lengths*, as one batch: keys and
    values left-padded with zeros."""
    merged, longest = caches[0], max(lengths)
    for i, layer in enumerate(merged.layers):
        parts = [c.layers[i] for c in caches]
        if isinstance(layer, LinearAttentionCacheLayerMixin):
            for s in range(layer.number_of_states):
                layer.conv_states[s] = torch.cat([p.conv_states[s] for p in parts])
                layer.recurrent_states[s] = torch.cat([p.recurrent_states[s] for p in parts])
        else:
            pad = [(0, 0, longest - n, 0) for n in lengths]
            layer.keys = torch.cat([F.pad(p.keys, d) for p, d in zip(parts, pad)])
            layer.values = torch.cat([F.pad(p.values, d) for p, d in zip(parts, pad)])
    return merged


@torch.no_grad()
def generate(model: torch.nn.Module, prompts: list[list[int]], max_new: int,
             eos: Collection[int]) -> Generated:
    """Greedy tokens after each of *prompts*: up to *max_new*, through the
    first of *eos*; as :func:`core.generate.greedy` makes them one at a
    time."""
    dev = model.device
    start = _clock()
    n = shared(prompts)
    prefix = DynamicCache(config=model.config)
    if n > 0:
        _forward(model, torch.tensor([prompts[0][:n]], device=dev), prefix,
                 torch.arange(n, device=dev)[None])
    caches, first = [], []
    for p in prompts:
        cache = copy.deepcopy(prefix)
        first.append(_forward(model, torch.tensor([p[n:]], device=dev), cache,
                              torch.arange(n, len(p), device=dev)[None]))
        caches.append(cache)
    lengths = [len(p) for p in prompts]
    cache = _merge(caches, lengths)
    mask = torch.tensor([[0] * (max(lengths) - k) + [1] * k for k in lengths], device=dev)
    positions = torch.tensor(lengths, device=dev)[:, None]
    x = torch.cat(first)
    prefilled = _clock()

    tokens: list[list[int]] = [[] for _ in prompts]
    live = set(range(len(prompts)))
    for step in range(max_new):
        for i, t in enumerate(x.tolist()):
            if i in live:
                tokens[i].append(t)
                if t in eos:
                    live.discard(i)
        if not live or step == max_new - 1:
            break
        mask = F.pad(mask, (0, 1), value=1)
        x = _forward(model, x[:, None], cache, positions, mask)
        positions = positions + 1
    return Generated(tokens, prefilled - start, _clock() - prefilled)
