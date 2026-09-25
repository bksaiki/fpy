"""
Greedy decode, and where each run's output departs from R0's.

The divergence index (Yuan et al. 2025) of a prompt is the first generated
position at which a run's greedy token differs from R0's; a run that matches
R0 to its end never diverges.  Prompts are a fixed random sample of MATH-500
under Qwen3's chat template, thinking off, with the model card's instruction
for math; decoding is up to 2,048 new tokens, as Yuan et al. do for
non-reasoning models.  A run other than R0 stops at its first departure, so
its cost is its divergence index rather than R0's length.

Each run's tokens go to `<out>/<run>.json` and are reused by a later call
with the same settings.

    python serve/decode.py -o dec                      # every run
    python serve/decode.py -o dec -r amd.cdna2.bf16 --prompts 10
"""

import argparse
import json
import random
import statistics
import sys
from collections.abc import Collection, Iterator
from pathlib import Path
from typing import Any

import swap
import torch

INSTRUCTION = 'Please reason step by step, and put your final answer within \\boxed{}.'


def stop_tokens(model: torch.nn.Module, tok: Any) -> set[int]:
    """The chat template's end of turn (the tokenizer's EOS) and the model's
    end of text: Qwen3.5 has no `generation_config.json` to give both."""
    ids = model.generation_config.eos_token_id
    return {tok.eos_token_id, *(ids if isinstance(ids, list) else [ids])}


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
        t = int(model(x, past_key_values=cache, use_cache=True).logits[0, -1].argmax())
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


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('--model', default=swap.MODEL)
    ap.add_argument('-o', '--out', required=True, help='directory for each run\'s JSON')
    ap.add_argument('-r', '--runs', nargs='*', default=[m for m in swap.MODES if m != 'fp32'],
                    help='runs besides fp32 (default: bf16-exact and every design)')
    ap.add_argument('--prompts', type=int, default=100, help='MATH-500 problems, a fixed random sample')
    ap.add_argument('--max-new', type=int, default=2048)
    ap.add_argument('--split-k', type=int, default=1)
    ap.add_argument('--combine', choices=['linear', 'tree'], default='linear')
    args = ap.parse_args(argv)

    import datasets
    from transformers import AutoTokenizer

    settings = {'model': args.model, 'prompts': args.prompts, 'max_new': args.max_new,
                'split_k': args.split_k, 'combine': args.combine}
    problems = datasets.load_dataset('HuggingFaceH4/MATH-500', split='test')['problem']
    picked = sorted(random.Random(0).sample(range(len(problems)), args.prompts))
    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = [tok.apply_chat_template(
        [{'role': 'user', 'content': f'{problems[i]}\n{INSTRUCTION}'}],
        add_generation_prompt=True, enable_thinking=False, return_dict=True,
        return_tensors='pt')['input_ids'] for i in picked]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = run = None
    runs: dict[str, list[list[int]]] = {}
    for mode in ('fp32', *[m for m in args.runs if m != 'fp32']):
        path = out / f'{mode}.json'
        if path.exists() and (cached := json.loads(path.read_text()))['settings'] == settings:
            runs[mode] = cached['tokens']
            continue
        if model is None:
            model, run = swap.load(args.model)
            run.split_k, run.combine = args.split_k, args.combine
            eos = stop_tokens(model, tok)
        run.mode = mode
        ref = runs.get('fp32')
        runs[mode] = []
        for i, p in enumerate(prompts):
            runs[mode].append(greedy(model, p.cuda(), args.max_new, eos, ref and ref[i]))
            print(f'{mode} prompt {i + 1}', file=sys.stderr, flush=True)
        run.mode = 'fp32'
        path.write_text(json.dumps({'settings': settings, 'tokens': runs[mode]}))

    ref = runs['fp32']
    print(f'R0: {len(ref)} prompts, mean length {statistics.mean(map(len, ref)):.0f} tokens, '
          f'{sum(len(r) == args.max_new for r in ref)} at the limit')
    print(f'{"run":20} {"diverged":>16} {"mean index":>11} {"median":>7}')
    for mode, got in runs.items():
        idx: list[Any] = [divergence(r, g) for r, g in zip(ref, got)]
        hit = [i for i in idx if i is not None]
        mean, med = (f'{statistics.mean(hit):.0f}', f'{statistics.median(hit):.0f}') if hit else ('-', '-')
        print(f'{mode:20} {len(hit):5} ({len(hit) / len(ref):6.1%}) {mean:>11} {med:>7}')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
