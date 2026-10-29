"""
The evaluation's inputs: WikiText-2's test split (`wikitext_ids`), cut into
`segments`, and `pick`'s seeded subsets.
"""

import random

import torch

CONTEXT = 2048
"""Tokens per WikiText-2 segment (the GPTQ convention)."""


def pick(total: int, n: int | None, seed: int = 0) -> list[int]:
    """*n* of `range(total)` at random by *seed*, sorted: all of them if *n*
    is `None` or no fewer."""
    if n is None or n >= total:
        return list(range(total))
    return sorted(random.Random(seed).sample(range(total), n))


def wikitext_ids(model: str) -> torch.Tensor:
    """WikiText-2's test split, joined as the Hugging Face perplexity guide
    does and tokenized for *model*: `[1, t]` on the GPU."""
    import datasets
    from transformers import AutoTokenizer

    text = '\n\n'.join(datasets.load_dataset(
        'Salesforce/wikitext', 'wikitext-2-raw-v1', split='test')['text'])
    return AutoTokenizer.from_pretrained(model)(text, return_tensors='pt').input_ids.cuda()


def segments(ids: torch.Tensor, context: int = CONTEXT) -> list[torch.Tensor]:
    """*ids* `[1, t]` as whole segments of *context* tokens."""
    return [ids[:, s:s + context] for s in range(0, ids.shape[1] - context + 1, context)]
