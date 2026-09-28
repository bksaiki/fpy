"""
Greedy generation with the chat template, thinking off.
"""

from collections.abc import Collection, Iterator
from typing import Any

import torch


def stop_tokens(model: torch.nn.Module, tok: Any) -> set[int]:
    """The tokenizer's EOS (the chat template's end of turn) and the model's
    end of text."""
    ids = model.generation_config.eos_token_id
    return {tok.eos_token_id, *(ids if isinstance(ids, list) else [ids])}


def encode(tok: Any, messages: list[dict[str, str]]) -> torch.Tensor:
    """*messages* under *tok*'s chat template, thinking off, ready for the
    reply: `[1, t]` on the GPU."""
    return tok.apply_chat_template(
        messages, add_generation_prompt=True, enable_thinking=False,
        return_dict=True, return_tensors='pt')['input_ids'].cuda()


@torch.no_grad()
def stream(
    model: torch.nn.Module, prompt: torch.Tensor, max_new: int, eos: Collection[int],
) -> Iterator[int]:
    """Greedy tokens after *prompt* `[1, t]` as they are made: up to *max_new*,
    through the first of *eos*."""
    from transformers import DynamicCache

    cache = DynamicCache(config=model.config)
    x = prompt
    for _ in range(max_new):
        out = model(x, past_key_values=cache, use_cache=True, logits_to_keep=1)
        t = int(out.logits[0, -1].argmax())
        yield t
        if t in eos:
            return
        x = prompt.new_tensor([[t]])


def greedy(
    model: torch.nn.Module, prompt: torch.Tensor, max_new: int, eos: Collection[int],
    ref: list[int] | None = None,
) -> list[int]:
    """:func:`stream`'s tokens, and with *ref* through the first that differs
    from it."""
    out: list[int] = []
    for t in stream(model, prompt, max_new, eos):
        out.append(t)
        if ref is not None and t != ref[len(out) - 1]:
            break
    return out


def divergence(ref: list[int], got: list[int]) -> int | None:
    """The first position where *got* departs from *ref*, or `None`."""
    return next((i for i, (a, b) in enumerate(zip(ref, got)) if a != b), None)
