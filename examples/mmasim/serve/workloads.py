"""
The token sequences local metrics are captured on, every token tagged.

- `wikitext`: the first tokens of WikiText-2's test split, one sequence of
  prose (the quantization-calibration convention).
- `mtbench`: MT-Bench's 80 two-turn conversations (10 each of writing,
  roleplay, reasoning, math, coding, extraction, STEM and humanities) held as
  a user would with the model: its chat template, thinking off, each reply
  R0's greedy decode.  A conversation's sequence is its last turn as the
  model processes it -- the first exchange as history, the second question,
  the reply -- so every token is one the session has the model read or write.
  Generated once (~30 min on Qwen3-0.6B) and cached as JSON.

Tags: `workload`; `category` (MT-Bench's, or `wikitext`); `role`, from the
chat template's structure: `user`, `assistant`, `template` (its markers and
role headers), or `text` for WikiText.
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
    return [Sequence(ids, {'workload': ['wikitext'] * n, 'category': ['wikitext'] * n,
                           'role': ['text'] * n})]


def roles(ids: list[int], tok: Any) -> list[str]:
    """Each token's role in a chat-template sequence (`<|im_start|>role\\n
    ... <|im_end|>`, as Qwen's template writes it)."""
    start, end = tok.convert_tokens_to_ids('<|im_start|>'), tok.convert_tokens_to_ids('<|im_end|>')
    out: list[str] = []
    role, header, name = 'template', False, ''
    for i in ids:
        if i == start:
            header, name, tag = True, '', 'template'
        elif header:
            name += tok.decode([i])
            header, role, tag = '\n' not in name, name.strip(), 'template'
        elif i == end:
            role = tag = 'template'
        else:
            tag = role
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
                prompt = tok.apply_chat_template(
                    messages, add_generation_prompt=True, enable_thinking=False,
                    return_dict=True, return_tensors='pt')['input_ids']
                reply = list(decode.stream(model, prompt.cuda(), MAX_NEW, eos))
                messages.append({'role': 'assistant',
                                 'content': tok.decode(reply, skip_special_tokens=True)})
            conversations.append({'category': row['category'],
                                  'ids': prompt[0].tolist() + reply})
        cached = {'model': name, 'max_new': MAX_NEW, 'conversations': conversations}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cached))
    return [Sequence(c['ids'], {'workload': ['mtbench'] * len(c['ids']),
                                'category': [c['category']] * len(c['ids']),
                                'role': roles(c['ids'], tok)})
            for c in cached['conversations']]
