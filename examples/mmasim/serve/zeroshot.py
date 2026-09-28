"""
Zero-shot accuracy on the standard suite, and each run's flips against R0.

`lm-evaluation-harness` runs PIQA, ARC-Easy, ARC-Challenge, HellaSwag,
WinoGrande and LAMBADA (OpenAI) zero-shot for each run, on one model whose
linear layers `swap.patch` switches.  A flip (Dutta et al. 2024) is an item
whose correctness differs from R0's, either way; counted for each per-item
metric the task logs (`acc`, `acc_norm`), over the items both runs have.
Paired per item (:func:`against`), each run against R0 and each design
against the scheme's exact run, per task and macro-averaged over them, each
with its standard error: Δ acc, and the continuous :data:`SCORES` from each
choice's log-likelihood.  The
macro Δ acc's and Δ log-likelihood's p-values are Holm-adjusted over the runs
sharing a reference.  All in `<out>/paired.json`.

Each run's results go to `<out>/<run>.json` and are reused by a later call
with the same settings, so the suite can be run a design at a time.

    python serve/zeroshot.py -o zs                        # every run
    python serve/zeroshot.py -o zs-full -r bf16-exact --items 0
    python serve/zeroshot.py -o smoke --items 10
"""

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import torch
from core import cli, stats, swap, workloads

TASKS = ('piqa', 'arc_easy', 'arc_challenge', 'hellaswag', 'winogrande', 'lambada_openai')
METRICS = ('acc', 'acc_norm')

SCORES = ('ll', 'margin', 'kl')
"""Per item, against a reference: Δ of the correct choice's log-likelihood;
Δ of its margin over the best wrong one; KL(ref || run) of the softmax over
the choices.  The last two need two choices (not LAMBADA)."""

Items = dict[str, dict[str, Any]]
"""Per item (`doc_id`, as a string): each of :data:`METRICS` it logs, each
choice's log-likelihood `lls`, and the correct one's index `gold`."""


def sizes() -> dict[str, int]:
    """Each of :data:`TASKS`' item count."""
    from lm_eval.tasks import TaskManager, get_task_dict

    tasks = get_task_dict(list(TASKS), TaskManager())
    return {t: len(tasks[t].eval_docs) for t in TASKS}


def subsets(counts: dict[str, int], n: int | None, seed: int = 0) -> dict[str, list[int]]:
    """*n* items of each task in *counts* that has more (`workloads.pick`), by
    index."""
    return {t: workloads.pick(k, n, seed) for t, k in counts.items() if n is not None and n < k}


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
        'items': {t: {str(s['doc_id']): {**{m: float(s[m]) for m in METRICS if m in s},
                                          'lls': [float(r[0]) for r in s['filtered_resps']],
                                          'gold': _gold(t, s)}
                      for s in out['samples'][t]} for t in TASKS},
    }


def _gold(task: str, sample: dict[str, Any]) -> int:
    """The correct choice's index in *sample*: WinoGrande's from its answer,
    LAMBADA's only one."""
    if task == 'winogrande':
        return int(sample['doc']['answer']) - 1
    return sample['target'] if isinstance(sample['target'], int) else 0


def _scores(ref: dict[str, Any], got: dict[str, Any]) -> dict[str, float]:
    """One item's :data:`SCORES`, *got* against *ref*."""
    r, g, k = torch.tensor(ref['lls']), torch.tensor(got['lls']), ref['gold']
    out = {'ll': float(g[k] - r[k])}
    if len(r) > 1:
        def margin(x: torch.Tensor) -> float:
            return float(x[k] - torch.cat([x[:k], x[k + 1:]]).max())

        lr, lg = r.log_softmax(0), g.log_softmax(0)
        out |= {'margin': margin(g) - margin(r), 'kl': float((lr.exp() * (lr - lg)).sum())}
    return out


def _both(ref: Items, got: Items) -> dict[str, list[str]]:
    """Per metric of :data:`METRICS`, the items both *ref* and *got* log it
    for (none left out)."""
    common = sorted(ref.keys() & got.keys())
    both = {m: [d for d in common if m in ref[d] and m in got[d]] for m in METRICS}
    return {m: ds for m, ds in both.items() if ds}


def flips(ref: Items, got: Items) -> dict[str, tuple[int, int]]:
    """Per metric: the items whose value differs, and the items compared."""
    return {m: (sum(ref[d][m] != got[d][m] for d in ds), len(ds))
            for m, ds in _both(ref, got).items()}


def delta(ref: Items, got: Items) -> dict[str, tuple[float, float]]:
    """Over the items both have, the mean and standard error of each metric's
    `got - ref`, and of each of :data:`SCORES`."""
    out = {m: stats.paired([got[d][m] for d in ds], [ref[d][m] for d in ds])
           for m, ds in _both(ref, got).items()}
    per = [_scores(ref[d], got[d]) for d in sorted(ref.keys() & got.keys())]
    for m in SCORES:
        if xs := [x[m] for x in per if m in x]:
            out[m] = stats.paired(xs)
    return out


def against(items: dict[str, dict[str, Items]], exact: str) -> dict[str, dict[str, Any]]:
    """Paired per item, per reference (R0, and *exact* for the designs) and
    run: :func:`delta` per task, its macro average over tasks with standard
    error, and the macro Δ acc's and Δ log-likelihood's p-values,
    Holm-adjusted over the runs sharing the reference.  *items* is each run's
    items per task."""
    out: dict[str, dict[str, Any]] = {}
    for ref in ('fp32', exact):
        rows: dict[str, dict[str, Any]] = {}
        for mode, per in items.items():
            if ref not in items or mode in ('fp32', ref):
                continue
            tasks = {t: delta(items[ref][t], per[t]) for t in per}
            macro = {}
            for m in (*METRICS, *SCORES):
                if ds := [d[m] for d in tasks.values() if m in d]:
                    macro[m] = (statistics.fmean(x for x, _ in ds),
                                math.sqrt(sum(se * se for _, se in ds)) / len(ds))
            rows[mode] = {'tasks': tasks, 'macro': macro,
                          'p': {m: stats.p_value(*macro[m]) for m in ('acc', 'll')}}
        for m in ('acc', 'll'):
            for r, p in zip(rows.values(), stats.holm([r['p'][m] for r in rows.values()])):
                r.setdefault('p_holm', {})[m] = p
        if rows:
            out[ref] = rows
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    cli.add_args(ap)
    cli.add_scheme_args(ap)
    cli.add_runs(ap)
    ap.add_argument('-o', '--out', required=True, help='directory for each run\'s JSON')
    ap.add_argument('--items', type=int, default=500,
                    help='items per task at most, at random (0: all)')
    ap.add_argument('--batch-size', type=int, default=16)
    args = ap.parse_args(argv)

    from lm_eval.models.huggingface import HFLM
    from transformers import AutoTokenizer

    modes = cli.runs(ap, args)
    settings = {'model': args.model, 'items': args.items, 'seed': args.seed, 'logged': 'lls',
                'batch_size': args.batch_size}
    samples = subsets(sizes(), args.items or None, args.seed) or None

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lm = run = None
    runs: dict[str, dict[str, Any]] = {}
    for mode in ('fp32', *modes):
        path = out / f'{mode}.json'
        want = settings if mode == 'fp32' else {
            **settings, 'scheme': args.scheme.name, 'requantize': args.requantize,
            'split_k': args.split_k, 'combine': args.combine}
        if cached := cli.cached(path, want):
            runs[mode] = cached
            continue
        if lm is None:
            model, run, _ = cli.load(args)
            lm = HFLM(pretrained=model, tokenizer=AutoTokenizer.from_pretrained(args.model),
                      batch_size=args.batch_size)
        runs[mode] = {'settings': want,
                      **evaluate(lm, run, mode, samples=samples)}
        path.write_text(json.dumps(runs[mode]))

    ref = runs['fp32']['items']
    for t in TASKS:
        print(f'\n{t}')
        for mode, r in runs.items():
            res, f = r['results'][t], flips(ref[t], r['items'][t])
            cells = [f'{m} {res[f"{m},none"]:7.2%} ±{res[f"{m}_stderr,none"]:.2%} '
                     f'flips {f[m][0]:4} ({f[m][0] / f[m][1]:5.2%})' for m in f]
            print(f'  {mode:20} ' + '   '.join(cells))

    paired = against({mode: r['items'] for mode, r in runs.items()}, f'{args.scheme.name}-exact')
    (out / 'paired.json').write_text(json.dumps(paired, indent=2))
    for ref, rows in paired.items():
        for m, show in (('acc', '{:+7.2%} ±{:.2%}'), ('ll', '{:+.1e} ±{:.0e}')):
            print(f'\npaired per item, Δ {m} against {ref}\n{"run":20} '
                  + ' '.join(f'{t[:10]:>16}' for t in TASKS) + f' {"macro":>16} {"Holm p":>7}')
            for mode, r in rows.items():
                cells = [r['tasks'][t][m] for t in TASKS] + [r['macro'][m]]
                print(f'{mode:20} ' + ' '.join(f'{show.format(d, se):>16}' for d, se in cells)
                      + f' {r["p_holm"][m]:7.3f}')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
