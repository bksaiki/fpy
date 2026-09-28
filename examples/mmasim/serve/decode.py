"""
Greedy decode, and where each run departs from R0's output.

R0 decodes greedily: a seeded random sample of MATH-500 under the model's
chat template, thinking off, with Qwen's instruction for math, up to 2,048
new tokens, as Yuan et al. 2025 do for non-reasoning models; cached as
`<out>/fp32.json` for the same settings.  Every run is then teacher-forced
on each prompt and R0's reply, one prefill (`scoring.evaluate`), and
compared position by position with R0 forced so, and a design with the
scheme's exact run too.  A prompt's divergence index is its first
disagreement, Yuan et al.'s index computed on prefill (it can differ from
decoding's at near-ties); R0's first miss of its own tokens is recorded too.
Reported: `scoring.against`'s paired per-prompt statistics, the fraction
diverged and the median index among them; all in `<out>/forced.json`.

    python serve/decode.py -o dec                      # every run
    python serve/decode.py -o dec -r amd.cdna2.bf16 --prompts 10
    python serve/decode.py -o dec-fp8 --scheme fp8-row
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch
from core import cli, generate, scoring, workloads


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    cli.add_args(ap)
    cli.add_scheme_args(ap)
    cli.add_runs(ap)
    ap.add_argument('-o', '--out', required=True, help='directory for R0\'s tokens and the results')
    ap.add_argument('--prompts', type=int, default=30,
                    help='MATH-500 problems, at random (0: all)')
    ap.add_argument('--max-new', type=int, default=2048)
    args = ap.parse_args(argv)

    from transformers import AutoTokenizer

    modes = cli.runs(ap, args)
    settings = {'model': args.model, 'prompts': args.prompts, 'seed': args.seed,
                'max_new': args.max_new}
    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = workloads.math500(tok, args.prompts or None, args.seed)
    model, run, about = cli.load(args)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'fp32.json'
    if cached := cli.cached(path, settings):
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
        'paired': scoring.against(totals),
        'per_prompt': {mode: t.per_sequence() for mode, t in totals.items()},
    }
    (out / 'forced.json').write_text(json.dumps(results, indent=2))

    print(f'R0: {len(ref)} prompts, mean length {statistics.mean(map(len, ref)):.0f} tokens, '
          f'{sum(len(r) == args.max_new for r in ref)} at the limit; forced, it misses its own '
          f'tokens on {results["floor"]["missed"]:.1%} '
          f'(median first miss {scoring.median_first(floor):.0f})')
    for name, rows in results['paired'].items():
        print(f'\nagainst {name}, forced\n{"run":20} {"diverged":>16} {"median index":>20} '
              f'{"disagree":>18} {"KL":>20}')
        for mode, r in rows.items():
            print(f'{mode:20} {r["diverged"]:6.1%} ±{r["diverged_se"]:5.1%}  '
                  f'{r["median"]:6.0f} [{r["median_lo"]:.0f}, {r["median_hi"]:.0f}]  '
                  f'{r["disagree"]:7.3%} ±{r["disagree_se"]:.3%} '
                  f'{r["kl"]:10.3e} ±{r["kl_se"]:.1e}')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
