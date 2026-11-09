"""
Batched greedy generation that prefills the prompts' shared prefix once.

Qwen3.5's linear-attention layers decay their state at padding inside a
sequence, so prompts are not padded while prefilling.  The shared prefix is
prefilled once; each prompt's rest, alone, from a copy of its cache.  The
caches then merge into one batch, attention keys and values left-padded and
masked, to decode together.
"""

import contextlib
import copy
import time
from collections.abc import Callable, Collection
from contextlib import AbstractContextManager
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
    prefix: float
    """Of :attr:`prefill`, seconds for the shared prefix."""
    questions: list[float]
    """Of :attr:`prefill`, seconds for each prompt's rest, its cache copy included."""
    steps: list[float]
    """Seconds for each decode step."""
    running: list[int]
    """Prompts still generating at each decode step."""


def clock() -> float:
    """Seconds, once the GPU is idle."""
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


@dataclass
class State:
    """A batch ready to decode one more token per row."""

    cache: DynamicCache
    mask: torch.Tensor
    positions: torch.Tensor
    x: torch.Tensor
    """Each row's next token."""

    def step(self, model: torch.nn.Module) -> None:
        """Decodes :attr:`x`; :attr:`x` becomes the token after it."""
        self.mask = F.pad(self.mask, (0, 1), value=1)
        self.x = _forward(model, self.x[:, None], self.cache, self.positions, self.mask)
        self.positions = self.positions + 1


def prefix(model: torch.nn.Module, prompts: list[list[int]]) -> tuple[DynamicCache, int]:
    """*prompts*' shared prefix (:func:`shared`), prefilled, and its length."""
    dev = model.device
    n = shared(prompts)
    cache = DynamicCache(config=model.config)
    if n > 0:
        _forward(model, torch.tensor([prompts[0][:n]], device=dev), cache,
                 torch.arange(n, device=dev)[None])
    return cache, n


def question(model: torch.nn.Module, prefix: DynamicCache, n: int,
             prompt: list[int]) -> tuple[DynamicCache, torch.Tensor]:
    """*prompt*'s rest after the *n*-token *prefix*, prefilled alone from a
    copy of its cache; and the token after it."""
    dev = model.device
    cache = copy.deepcopy(prefix)
    return cache, _forward(model, torch.tensor([prompt[n:]], device=dev), cache,
                           torch.arange(n, len(prompt), device=dev)[None])


def batch(prompts: list[list[int]], caches: list[DynamicCache], first: list[torch.Tensor]) -> State:
    """*prompts*, each prefilled into its own of *caches* with its next token
    in *first*, merged to decode together."""
    lengths = [len(p) for p in prompts]
    dev = first[0].device
    mask = torch.tensor([[0] * (max(lengths) - k) + [1] * k for k in lengths], device=dev)
    return State(_merge(caches, lengths), mask, torch.tensor(lengths, device=dev)[:, None],
                 torch.cat(first))


@torch.no_grad()
def generate(model: torch.nn.Module, prompts: list[list[int]], max_new: int,
             eos: Collection[int],
             decoding: Callable[[], AbstractContextManager] = contextlib.nullcontext) -> Generated:
    """Greedy tokens after each of *prompts*: up to *max_new*, through the
    first of *eos*; as :func:`core.generate.greedy` makes them one at a
    time.  Decode runs within *decoding*."""
    start = clock()
    cache, n = prefix(model, prompts)
    prefixed = clock()
    caches, first, questions = [], [], []
    for p in prompts:
        t = clock()
        c, x = question(model, cache, n, p)
        caches.append(c)
        first.append(x)
        questions.append(clock() - t)
    state = batch(prompts, caches, first)
    prefilled = clock()

    tokens: list[list[int]] = [[] for _ in prompts]
    live = set(range(len(prompts)))
    steps: list[float] = []
    running: list[int] = []
    with decoding():
        for step in range(max_new):
            for i, t in enumerate(state.x.tolist()):
                if i in live:
                    tokens[i].append(t)
                    if t in eos:
                        live.discard(i)
            if not live or step == max_new - 1:
                break
            t = clock()
            state.step(model)
            steps.append(clock() - t)
            running.append(len(live))
    return Generated(tokens, prefilled - start, clock() - prefilled, prefixed - start,
                     questions, steps, running)
