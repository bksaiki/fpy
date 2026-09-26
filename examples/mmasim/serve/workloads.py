"""
The token sequences local metrics are captured on, every token tagged.

- `wikitext`: the first tokens of WikiText-2's test split, one sequence of
  prose.
- `mtbench`: MT-Bench's 80 two-turn conversations held as a user would with
  the model: its chat template, thinking off, each reply R0's greedy decode.
  A conversation's sequence is its last turn as the model processes it: the
  first exchange as history, the second question, the reply.  Generated once
  and cached as JSON.

Tags: `category` (MT-Bench's, or `wikitext`); `role`, from the chat
template's structure: `user`, `assistant`, `template` (its markers, role
headers and empty think block), or `text` for WikiText.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import decode
import perplexity
import swap
import torch

MAX_NEW = 512
"""Tokens per MT-Bench reply at most."""


@dataclass
class Sequence:
    ids: list[int]
    tags: dict[str, list[str]]
    """Per tag, each token's label."""


def wikitext(model: str, tokens: int) -> list[Sequence]:
    ids = perplexity.wikitext(model)[0, :tokens].tolist()
    n = len(ids)
    return [Sequence(ids, {'category': ['wikitext'] * n, 'role': ['text'] * n})]


def roles(ids: list[int], tok: Any) -> list[str]:
    """Each token's role in a chat-template sequence (`<|im_start|>role\\n
    ... <|im_end|>`, as Qwen's template writes it)."""
    start, end, think, unthink = (tok.convert_tokens_to_ids(t) for t in (
        '<|im_start|>', '<|im_end|>', '<think>', '</think>'))
    out: list[str] = []
    role, header, name, thinking, closed = 'template', False, '', False, False
    for i in ids:
        if i == start:
            header, name, tag = True, '', 'template'
        elif header:
            name += tok.decode([i])
            header, role, tag = '\n' not in name, name.strip(), 'template'
        elif i == end:
            role = tag = 'template'
        elif i == think or thinking:
            thinking, closed, tag = i != unthink, i == unthink, 'template'
        elif closed and not tok.decode([i]).strip():
            tag = 'template'
        else:
            closed, tag = False, role
        out.append(tag)
    return out


def mtbench(model: torch.nn.Module, run: swap.Run, tok: Any, path: Path) -> list[Sequence]:
    """The MT-Bench sequences for *model* (`swap.patch`ed as *run*), from
    *path* if it holds them, else decoded under `fp32` and written there."""
    name = model.config.name_or_path
    cached = json.loads(path.read_text()) if path.exists() else None
    if cached is None or cached['model'] != name or cached['max_new'] != MAX_NEW:
        import datasets

        eos = decode.stop_tokens(model, tok)
        run.mode = 'fp32'
        conversations = []
        for row in datasets.load_dataset('HuggingFaceH4/mt_bench_prompts', split='train'):
            messages: list[dict[str, str]] = []
            for question in row['prompt']:
                messages.append({'role': 'user', 'content': question})
                prompt = decode.encode(tok, messages)
                reply = list(decode.stream(model, prompt, MAX_NEW, eos))
                messages.append({'role': 'assistant',
                                 'content': tok.decode(reply, skip_special_tokens=True)})
            conversations.append({'category': row['category'],
                                  'ids': prompt[0].tolist() + reply})
        cached = {'model': name, 'max_new': MAX_NEW, 'conversations': conversations}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cached))
    return [Sequence(c['ids'], {'category': [c['category']] * len(c['ids']),
                                'role': roles(c['ids'], tok)})
            for c in cached['conversations']]
