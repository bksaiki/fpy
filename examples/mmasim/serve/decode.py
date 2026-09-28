"""
Greedy decode, and where each run departs from R0's output.

R0 decodes greedily: a seeded random sample of MATH-500 under the model's
chat template, thinking off, with Qwen's instruction for math, up to 2,048
new tokens, as Yuan et al. 2025 do for non-reasoning models; cached as
`<out>/fp32.json` for the same settings.  Every run is then teacher-forced
on each prompt and R0's reply, one prefill (`scoring.evaluate`), and
compared position by position with R0 forced so, and a design with the
scheme's exact run too.  A prompt's divergence index is its first
disagreement, Yuan et al.'s index computed on prefill: at a near-tie,
prefill's and decoding's attention round differently enough to decide
differently, so the two can differ prompt by prompt.  R0's first miss of
its own tokens is recorded too.  Reported: the fraction diverged, the
median index of those that do (a bootstrap interval over prompts), and the
paired per-prompt statistics of `scoring.against`; all in
`<out>/forced.json`.

    python serve/decode.py -o dec                      # every run
    python serve/decode.py -o dec -r amd.cdna2.bf16 --prompts 10
    python serve/decode.py -o dec-fp8 --scheme fp8-row
"""

import argparse
import json
import math
import statistics
import sys
from collections.abc import Sequence
from pathlib import Path

import torch
from core import checkpoints, cli, generate, metrics, scoring, workloads


def _median(firsts: Sequence[int | None]) -> float:
    """The median of the indices that are not `None` (NaN if none are)."""
    hit = [f for f in firsts if f is not None]
    return statistics.median(hit) if hit else math.nan


def divergences(totals: dict[str, scoring.Totals]) -> dict[str, dict[str, dict[str, float]]]:
    """Per reference and run (`scoring.evaluate`'s comparisons, a run
    against itself left out): the fraction of prompts diverged with its
    standard error, and the median index of those that do with a bootstrap
    95% interval over prompts."""
    out: dict[str, dict[str, dict[str, float]]] = {}
    for ref in totals:
        rows = {}
        for mode, t in totals.items():
            if mode == ref or ref not in t.first:
                continue
            firsts = t.first[ref]
            frac, se = metrics.paired([float(f is not None) for f in firsts])
            lo, hi = metrics.bootstrap(firsts, _median)
            rows[mode] = {'diverged': frac, 'diverged_se': se, 'median': _median(firsts),
                          'median_lo': lo, 'median_hi': hi}
        if rows:
            out[ref] = rows
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    cli.add_args(ap)
    cli.add_scheme_args(ap)
    ap.add_argument('-o', '--out', required=True, help='directory for R0\'s tokens and the results')
    ap.add_argument('-r', '--runs', nargs='*',
                    help="runs besides fp32 (default: the scheme's exact run and every design)")
    ap.add_argument('--prompts', type=int, default=30, help='MATH-500 problems, at random')
    ap.add_argument('--max-new', type=int, default=2048)
    args = ap.parse_args(argv)

    from transformers import AutoTokenizer

    modes = cli.runs(ap, args)
    settings = {'model': args.model, 'prompts': args.prompts, 'seed': args.seed,
                'max_new': args.max_new}
    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = workloads.math500(tok, args.prompts, args.seed)
    model, run, about = checkpoints.for_scheme(
        args.model, args.scheme, requantize=args.requantize, master=args.master,
        split_k=args.split_k, combine=args.combine)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'fp32.json'
    if path.exists() and (cached := json.loads(path.read_text()))['settings'] == settings:
        ref = cached['tokens']
    else:
        eos = generate.stop_tokens(model, tok)
        ref = []
        for i, p in enumerate(prompts):
            ref.append(generate.greedy(model, p, args.max_new, eos))
            print(f'fp32 prompt {i + 1}', file=sys.stderr, flush=True)
        path.write_text(json.dumps({'settings': settings, 'tokens': ref}))

    seqs = [torch.cat([p, p.new_tensor([r])], -1) for p, r in zip(prompts, ref)]
    totals = scoring.evaluate(model, run, seqs, modes, starts=[p.shape[1] for p in prompts],
                                 progress=True)
    floor = totals['fp32'].first_miss
    results = {
        'settings': {**settings, 'scheme': args.scheme.name, 'split_k': args.split_k,
                     'combine': args.combine, **about},
        'floor': {'missed': sum(f is not None for f in floor) / len(floor), 'first': floor},
        'divergence': divergences(totals), 'paired': scoring.against(totals),
        'per_prompt': {mode: {'nll': t.nll_seg, **t.vs, 'first': t.first}
                       for mode, t in totals.items()},
    }
    (out / 'forced.json').write_text(json.dumps(results, indent=2))

    print(f'R0: {len(ref)} prompts, mean length {statistics.mean(map(len, ref)):.0f} tokens, '
          f'{sum(len(r) == args.max_new for r in ref)} at the limit; forced, it misses its own '
          f'tokens on {results["floor"]["missed"]:.1%} (median first miss {_median(floor):.0f})')
    for name, rows in results['divergence'].items():
        paired = results['paired'][name]
        print(f'\nagainst {name}, forced\n{"run":20} {"diverged":>16} {"median index":>20} '
              f'{"disagree":>18} {"KL":>20}')
        for mode, r in rows.items():
            q = paired[mode]
            print(f'{mode:20} {r["diverged"]:6.1%} ±{r["diverged_se"]:5.1%}  '
                  f'{r["median"]:6.0f} [{r["median_lo"]:.0f}, {r["median_hi"]:.0f}]  '
                  f'{q["disagree"]:7.3%} ±{q["disagree_se"]:.3%} '
                  f'{q["kl"]:10.3e} ±{q["kl_se"]:.1e}')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
