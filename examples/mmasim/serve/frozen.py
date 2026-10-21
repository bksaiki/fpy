"""
WikiText-2 distance of each matmul from the scheme's exact run, frozen to it.

Each segment runs the exact run (R1) once, recording every linear layer's
input; each matmul then runs frozen to it (`swap.Frozen`): its arithmetic's
error propagates, but no activation rounds differently from R1's, so a
design's distance is its own and not the scheme's re-drawn quantization.
Per matmul, KL(R1 || run) and top-1 disagreement over segments, and the
share of activation elements (and scales) its own run would round
differently; those in `--free` also as they run.  Segments as in
`perplexity.py`.

    python serve/frozen.py --scheme fp8-row
    python serve/frozen.py --scheme fp8-row --matmuls nv.hopper.e4m3.f32 --free nv.hopper.e4m3.f32
"""

import argparse
import sys
from pathlib import Path

from core import cli, outputs, scoring, workloads


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    cli.add_args(ap)
    cli.add_scheme_args(ap)
    cli.add_runs(ap)
    ap.add_argument('--free', nargs='*', default=[], metavar='MATMUL',
                    help='matmuls also run as they are (default: none)')
    ap.add_argument('--segments', type=int, default=10,
                    help='this many segments at random, 0 for all (default: %(default)s)')
    ap.add_argument('-o', '--out', type=Path, default=None,
                    help='write the results as JSON here, and a table beside it (.csv)')
    args = cli.parse(ap, argv)

    runs = cli.runs(ap, args)
    segs = workloads.segments(workloads.wikitext_ids(args.model))
    segs = [segs[i] for i in workloads.pick(len(segs), args.segments or None, args.seed)]
    model, run, about = cli.load(args, designs=runs)

    got = scoring.frozen(model, run, segs, runs, set(args.free), progress=True)
    rows = []
    for mode, d in got['frozen'].items():
        row = {'design': mode, **d.report()}
        if mode in got['free']:
            row |= {f'free_{k}': v for k, v in got['free'][mode].report().items()}
        rows.append(row)
    print(f'{"run":24} {"KL vs R1, frozen":>22} {"top-1":>9} {"re-rounded":>10} {"KL vs R1, free":>22}')
    for r in rows:
        free = f'{r["free_kl"]:10.3e} ±{r["free_kl_se"]:.1e}' if 'free_kl' in r else ''
        print(f'{r["design"]:24} {r["kl"]:10.3e} ±{r["kl_se"]:.1e} {r["disagree"]:9.4%} '
              f'{r["rerounded"]:10.4%} {free}')
    if args.out:
        outputs.write_json(args.out, {
            'model': args.model, 'scheme': args.scheme.name, **about,
            'context': workloads.CONTEXT, 'segments': len(segs), 'seed': args.seed,
            'split_k': args.split_k, 'combine': args.combine,
            'runs': rows, 'per_segment': {k: {m: vars(d) for m, d in v.items()} for k, v in got.items()},
            'provenance': cli.provenance(args)})
        outputs.write_csv(args.out.with_suffix('.csv'), rows)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
