"""
WikiText-2 perplexity, and each matmul's distance from the FP32 baseline.

The test split, joined and tokenized as the Hugging Face perplexity guide
does, cut into 2048-token segments (the GPTQ convention; the remainder
dropped), and scored by `scoring`: per token, NLL, KL(R0 || matmul), top-1 and
RMS Δp as llama.cpp's `perplexity --kl-divergence` reports them; paired over
segments against R0 and the scheme's exact matmul (`scoring.against`).

    python serve/perplexity.py                          # every matmul
    python serve/perplexity.py --matmuls amd.cdna2.bf16 --segments 4
    python serve/perplexity.py --split-k 4 --combine tree -o tree.json
    python serve/perplexity.py --scheme fp8-row --segments 0    # all 146
"""

import argparse
import json
import sys

from core import cli, scoring, workloads


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    cli.add_args(ap)
    cli.add_scheme_args(ap)
    cli.add_runs(ap)
    ap.add_argument('--segments', type=int, default=50,
                    help='this many segments at random, 0 for all (default: %(default)s)')
    ap.add_argument('-o', '--out', default=None, help='write the results as JSON here')
    args = cli.parse(ap, argv)

    runs = cli.runs(ap, args)
    segs = workloads.segments(workloads.wikitext_ids(args.model))
    segs = [segs[i] for i in workloads.pick(len(segs), args.segments or None, args.seed)]
    model, run, about = cli.load(args, designs=runs)

    totals = scoring.evaluate(model, run, segs, runs, progress=True)
    results = {
        'model': args.model, 'scheme': args.scheme.name, **about,
        'context': workloads.CONTEXT, 'segments': len(segs), 'seed': args.seed,
        'split_k': args.split_k, 'combine': args.combine,
        'runs': {mode: t.report() for mode, t in totals.items()},
        'paired': scoring.against(totals),
        'per_segment': {mode: t.per_sequence() for mode, t in totals.items()},
    }
    print(f'{"run":20} {"ppl":>14} {"KL":>20} {"top-1":>16} {"RMS dp":>8} {"s":>7}')
    for mode, r in results['runs'].items():
        print(f'{mode:20} {r["ppl"]:8.4f} ±{r["ppl_se"]:.3f} '
              f'{r["kl"]:10.3e} ±{r["kl_se"]:.1e} {r["top1"]:8.4%} ±{r["top1_se"]:.3%} '
              f'{r["rms_dp"]:8.3%} {r["seconds"]:7.0f}')
    for ref, rows in results['paired'].items():
        print(f'\npaired over {len(segs)} segments, against {ref}\n'
              f'{"run":20} {"Δ NLL":>22} {"Holm p":>7} {"KL":>20} {"top-1 disagree":>18}')
        for mode, r in rows.items():
            print(f'{mode:20} {r["dnll"]:+10.3e} ±{r["dnll_se"]:.1e} {r["p_holm"]:7.3f} '
                  f'{r["kl"]:10.3e} ±{r["kl_se"]:.1e} {r["disagree"]:8.4%} ±{r["disagree_se"]:.3%}')
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(results, f, indent=2)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
