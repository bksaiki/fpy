"""
Each matmul's distance from the scheme's exact run, frozen to it.

Each segment runs the exact run (R1) once, recording every linear layer's
input; each matmul then runs frozen to it (`swap.Frozen`), so only its
arithmetic differs.  Per matmul: KL(R1 || run), top-1 disagreement, the
share of activation elements (and scales) its own run would round
differently, its seconds and its kernels' compile seconds; `--free`
matmuls also as deployed.  Over WikiText-2 segments, every token scored.

    python serve/frozen.py --scheme fp8-row
    python serve/frozen.py --scheme fp8-row --matmuls nv.hopper.e4m3.f32 --free nv.hopper.e4m3.f32
"""

import argparse
import sys
from pathlib import Path

from core import cli, kernels, outputs, scoring, workloads


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    cli.add_args(ap)
    cli.add_scheme_args(ap)
    cli.add_runs(ap)
    ap.add_argument('--free', nargs='*', default=[], metavar='MATMUL',
                    help='matmuls run as they are, frozen too if in --matmuls (default: none)')
    ap.add_argument('--segments', type=int, default=10,
                    help='this many WikiText-2 segments at random, 0 for all (default: %(default)s)')
    ap.add_argument('-o', '--out', type=Path, default=None,
                    help='write the results as JSON here, and a table beside it (.csv)')
    args = cli.parse(ap, argv)

    runs = cli.runs(ap, args)
    model, run = cli.load(args)
    segs = workloads.segments(workloads.wikitext_ids(args.model))
    segs = [segs[i] for i in workloads.pick(len(segs), args.segments or None, args.seed)]

    got = scoring.frozen(model, run, segs, runs, set(args.free), seed=args.seed, progress=True)
    rows = []
    frozen_runs, free_runs = got[scoring.Kind.FROZEN], got[scoring.Kind.FREE]
    for mode in dict.fromkeys([*frozen_runs, *free_runs]):
        row = {'design': mode, **(frozen_runs[mode].report() if mode in frozen_runs else {})}
        if mode in free_runs:
            row |= {f'free_{k}': v for k, v in free_runs[mode].report().items()}
        if (t := kernels.compile_seconds(mode)) is not None:
            row['compile_seconds'] = t
        rows.append(row)
    print(f'{"run":24} {"KL vs R1, frozen":>22} {"top-1":>9} {"re-rounded":>10} {"KL vs R1, free":>22}')
    for r in rows:
        free = f'{r["free_kl"]:10.3e} ±{r["free_kl_se"]:.1e}' if 'free_kl' in r else ''
        held = (f'{r["kl"]:10.3e} ±{r["kl_se"]:.1e} {r["disagree"]:9.4%} {r["rerounded"]:10.4%}'
                if 'kl' in r else ' ' * 41)
        print(f'{r["design"]:24} {held} {free}')
    if args.out:
        outputs.write_json(args.out, {
            'model': args.model, 'scheme': args.scheme.name, 'segments': len(segs), 'seed': args.seed,
            'runs': rows, 'per_segment': {k: {m: vars(d) for m, d in v.items()} for k, v in got.items()},
            'provenance': cli.provenance(args)})
        outputs.write_csv(args.out.with_suffix('.csv'), rows)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
