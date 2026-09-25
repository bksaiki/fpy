"""
Zero-shot accuracy on the standard suite, and each run's flips against R0.

`lm-evaluation-harness` runs PIQA, ARC-Easy, ARC-Challenge, HellaSwag,
WinoGrande and LAMBADA (OpenAI) zero-shot for each run, on one model whose
linear layers `swap.patch` switches.  A flip (Dutta et al. 2024) is an item
whose correctness differs from R0's, either way; counted for each per-item
metric the task logs (`acc`, `acc_norm`), over the items both runs have.

Each run's results go to `<out>/<run>.json` and are reused by a later call
with the same settings, so the suite can be run a design at a time.

    python serve/zeroshot.py -o zs                        # every run
    python serve/zeroshot.py -o zs-full -r bf16-exact --hellaswag 0
    python serve/zeroshot.py -o smoke --limit 10
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import swap
import torch

MODEL = 'Qwen/Qwen3-0.6B'
TASKS = ('piqa', 'arc_easy', 'arc_challenge', 'hellaswag', 'winogrande', 'lambada_openai')
METRICS = ('acc', 'acc_norm')

Items = dict[str, dict[str, float]]
"""Per item (`doc_id`, as a string): each of :data:`METRICS` it logs."""


def hellaswag_subset(n: int, seed: int = 0) -> list[int]:
    """A fixed random *n* of HellaSwag's validation items, by index."""
    from lm_eval.tasks import TaskManager, get_task_dict

    total = len(get_task_dict(['hellaswag'], TaskManager())['hellaswag'].eval_docs)
    return sorted(random.Random(seed).sample(range(total), n))


def evaluate(lm: Any, run: swap.Run, mode: str, **kw: Any) -> dict[str, Any]:
    """The suite under *mode*: the harness's `results` per task, and `items`
    per task (:data:`Items`).  *kw* goes to `lm_eval.simple_evaluate`."""
    import lm_eval

    run.mode = mode
    try:
        out = lm_eval.simple_evaluate(model=lm, tasks=list(TASKS), log_samples=True, **kw)
    finally:
        run.mode = 'fp32'
    return {
        'results': {t: out['results'][t] for t in TASKS},
        'items': {t: {str(s['doc_id']): {m: float(s[m]) for m in METRICS if m in s}
                      for s in out['samples'][t]} for t in TASKS},
    }


def flips(ref: Items, got: Items) -> dict[str, tuple[int, int]]:
    """Per metric: the items whose value differs, and the items compared."""
    common = ref.keys() & got.keys()
    counts: dict[str, tuple[int, int]] = {}
    for m in METRICS:
        both = [d for d in common if m in ref[d] and m in got[d]]
        if both:
            counts[m] = (sum(ref[d][m] != got[d][m] for d in both), len(both))
    return counts


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('-o', '--out', required=True, help='directory for each run\'s JSON')
    ap.add_argument('-r', '--runs', nargs='*', default=[m for m in swap.MODES if m != 'fp32'],
                    help='runs besides fp32 (default: bf16-exact and every design)')
    ap.add_argument('--hellaswag', type=int, default=2000,
                    help='HellaSwag items, a fixed random subset (0: all)')
    ap.add_argument('--limit', type=int, default=None,
                    help='only the first this many items of every task (a smoke run)')
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--split-k', type=int, default=1)
    ap.add_argument('--combine', choices=['linear', 'tree'], default='linear')
    args = ap.parse_args(argv)

    from lm_eval.models.huggingface import HFLM
    from transformers import AutoModelForCausalLM, AutoTokenizer

    settings = {'model': MODEL, 'limit': args.limit, 'hellaswag': args.hellaswag,
                'batch_size': args.batch_size, 'split_k': args.split_k, 'combine': args.combine}
    samples = None
    if args.limit is None and args.hellaswag:
        samples = {'hellaswag': hellaswag_subset(args.hellaswag)}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lm = run = None
    runs: dict[str, dict[str, Any]] = {}
    for mode in ('fp32', *[m for m in args.runs if m != 'fp32']):
        path = out / f'{mode}.json'
        if path.exists() and (cached := json.loads(path.read_text()))['settings'] == settings:
            runs[mode] = cached
            continue
        if lm is None:
            model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).cuda().eval()
            run = swap.patch(model)
            run.split_k, run.combine = args.split_k, args.combine
            lm = HFLM(pretrained=model, tokenizer=AutoTokenizer.from_pretrained(MODEL),
                      batch_size=args.batch_size)
        runs[mode] = {'settings': settings,
                      **evaluate(lm, run, mode, limit=args.limit, samples=samples)}
        path.write_text(json.dumps(runs[mode]))

    ref = runs['fp32']['items']
    for t in TASKS:
        print(f'\n{t}')
        for mode, r in runs.items():
            res, f = r['results'][t], flips(ref[t], r['items'][t])
            cells = [f'{m} {res[f"{m},none"]:7.2%} ±{res[f"{m}_stderr,none"]:.2%} '
                     f'flips {f[m][0]:4} ({f[m][0] / f[m][1]:5.2%})' for m in f]
            print(f'  {mode:20} ' + '   '.join(cells))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
