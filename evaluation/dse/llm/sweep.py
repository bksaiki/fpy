"""
The grid evaluation: each of `config.GRID_ANCHORS` over `F`, and with its join swapped.

Each design runs through `frozen.py`, a pass per scheme over the same
segments: frozen (amplified) and free (observed), against the scheme's exact
run, with R1 an ulp up or down at random as the floor.  The table,
`<results>/sweep/sweep-grid-<model>.csv` (`plot_grid.py`), grows: a rerun
keeps its `ok` rows whose design is unchanged and runs the rest.
`--manifest` instead compiles the grid, launches each design as it will run
(at each of `--model`'s layer widths), and writes `manifest.csv`: a row per
design, its parameters, name, status and compile time.

    python sweep.py --manifest --model Qwen/Qwen3.5-0.8B   # from evaluation/dse/llm
    python sweep.py --model Qwen/Qwen3.5-0.8B
    python sweep.py --list                                 # the points, and stop
"""

import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

_EVAL = Path(__file__).resolve().parents[2]
# this directory, for `cli`; `common/mma`, for `core` and `generator`; `examples/mmasim`,
# for `models`; `evaluation` last, for `config`
sys.path[:0] = [str(Path(__file__).resolve().parent), str(_EVAL / 'common' / 'mma'),
                str(_EVAL.parent / 'examples' / 'mmasim')]
sys.path.append(str(_EVAL))

import cli
import config
import generator
import torch
from core import kernels, outputs, quant, swap


def register() -> None:
    """Every grid design, as `frozen.py --register` calls it."""
    for d in dict.fromkeys(p.design for p in config.grid()):
        if generator.name(d) not in kernels.TILES:
            kernels.register_design(d)


def _launch(pts: list[config.Point], names: dict[generator.Design, str], model: str) -> dict[str, str]:
    """Each of *pts*' designs launched at each of *model*'s layer widths; the
    ones that fail, and why.  Stops at an error that leaves the GPU unusable,
    which every later design would hit."""
    failed: dict[str, str] = {}
    for scheme in sorted({config.SCHEMES[p.anchor] for p in pts}):
        s = quant.scheme(scheme)
        net, run = swap.load(model, s)
        ks = cli.layer_ks(net, run)
        del net, run
        torch.cuda.empty_cache()
        mine = list(dict.fromkeys(names[p.design] for p in pts if config.SCHEMES[p.anchor] == scheme))
        for name in mine:
            for k in ks:
                qx, qw = (quant.quantize(torch.randn(r, k, device='cuda'), op) for r, op in ((64, s.x), (128, s.w)))
                try:
                    swap.gemm(name, s, qx, qw, qw.elements.to(kernels.storage(name)[1]))
                    torch.cuda.synchronize()
                except Exception as ex:  # noqa: BLE001 -- a failure is a result
                    failed[name] = f'{type(ex).__name__} at launch, k = {k}: {_last_line(ex)}'
                    try:
                        torch.cuda.synchronize()
                    except RuntimeError as dead:
                        raise SystemExit(f'{name} left the GPU unusable: {_last_line(dead)}') from ex
                    break
    return failed


def _last_line(ex: BaseException) -> str:
    """*ex*'s message's last line, cut to 200 characters."""
    return (str(ex).strip().splitlines() or [''])[-1][:200]


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('--model', choices=swap.MODELS, default=config.MODEL,
                    help='the model (default: %(default)s)')
    ap.add_argument('--results', type=Path, default=cli.RESULTS,
                    help="the outputs' root (default: %(default)s)")
    ap.add_argument('-j', '--jobs', type=int, default=8,
                    help='processes compiling the designs (default: %(default)s)')
    ap.add_argument('--segments', type=int, default=config.LLM_SEGMENTS,
                    help='WikiText-2 segments (default: %(default)s)')
    ap.add_argument('--seed', type=int, default=config.LLM_SEED,
                    help="the segments' and random rounding's seed (default: %(default)s)")
    ap.add_argument('--manifest', action='store_true',
                    help='compile and launch the grid as --model runs it, and write manifest.csv')
    ap.add_argument('--list', action='store_true', help='print the points, and stop')
    ap.add_argument('--smoke', action='store_true', help="the grid's first 2 designs, 1 segment")
    args = ap.parse_args(argv)
    if args.smoke:
        args.segments = 1

    pts = config.grid()[:2] if args.smoke else config.grid()
    tag = args.model.rsplit('/', 1)[-1]
    out = args.results / 'sweep'
    table = out / f'sweep-grid-{tag}.csv'
    kept: list[dict[str, str]] = []
    if not args.manifest and table.exists():
        # resume: keep the table's ok rows of the grid's designs; run the rest
        want = {(p.anchor, p.value): generator.name(p.design) for p in pts}
        with table.open() as f:
            kept = [r for r in csv.DictReader(f)
                    if r['status'] == 'ok' and want.get((r['anchor'], r['value'])) == r['design']]
        pts = [p for p in pts if (p.anchor, p.value) not in {(r['anchor'], r['value']) for r in kept}]
    names = {d: generator.name(d) for d in dict.fromkeys(p.design for p in pts)}
    if args.list:
        for p in pts:
            print(f'{p.anchor:24} {p.value}')
        print(f'{len(pts)} points, {len(names)} designs')
        return 0

    register()
    kernels.CACHE = cli.cache(args, 'kernels')
    refused = kernels.compile_designs(list(names.values()), args.jobs)
    if args.manifest:
        if torch.cuda.is_available():
            refused |= _launch([p for p in pts if names[p.design] not in refused], names, args.model)
        rows = [{'anchor': p.anchor, 'format': config.SCHEMES[p.anchor], 'design': names[p.design],
                 'status': 'refused' if names[p.design] in refused else 'ok',
                 'reason': refused.get(names[p.design], ''),
                 'compile_seconds': kernels.compile_seconds(names[p.design]), **p.about} for p in pts]
        outputs.write_csv(out / 'manifest.csv', rows)
        print(f'{len(rows)} designs, {len(rows) - len(refused)} run, {len(refused)} refused: '
              f'{out / "manifest.csv"}')
        return 0
    metrics: dict[str, dict[str, str]] = {}
    for scheme in sorted({config.SCHEMES[p.anchor] for p in pts}):
        run = [names[d] for d in dict.fromkeys(p.design for p in pts if config.SCHEMES[p.anchor] == scheme)
               if names[d] not in refused]
        if not run:
            continue
        json_out = out / f'e2e-{tag}-{scheme}.json'
        floor = f'{scheme}-exact-rand'
        done = subprocess.run([sys.executable, str(Path(__file__).resolve().parent / 'frozen.py'),
                               '--model', args.model, '--scheme', scheme, '--results', str(args.results),
                               '--register', str(Path(__file__).resolve()),
                               '--segments', str(args.segments), '--seed', str(args.seed),
                               '--matmuls', floor, *run, '--free', floor, *run, '-o', str(json_out)],
                              check=False)
        if done.returncode:
            raise SystemExit(f'frozen.py under {scheme} exited {done.returncode}')
        with json_out.with_suffix('.csv').open() as f:
            metrics |= {row['design']: row for row in csv.DictReader(f)}
    rows = []
    for p in pts:
        n = names[p.design]
        # empty for a scheme with no point run
        floor = metrics.get(f'{config.SCHEMES[p.anchor]}-exact-rand', {})
        rows.append({'anchor': p.anchor, 'scheme': config.SCHEMES[p.anchor], 'value': p.value,
                     'is_anchor': p.is_anchor, 'status': 'refused' if n in refused else 'ok',
                     'reason': refused.get(n, ''), 'design': n, **p.about,
                     'floor_kl': floor.get('kl', ''), 'k0_kl': floor.get('free_kl', ''),
                     **{k: v for k, v in metrics.get(n, {}).items() if k not in ('model', 'design')}})
    outputs.write_csv(table, kept + rows)
    outputs.write_json(table.with_suffix('.provenance.json'), {
        'argv': sys.argv, **outputs.commit(),
        'settings': {k: v if isinstance(v, (int, list)) else str(v) for k, v in vars(args).items()}})
    print(f'{len(rows)} points, {len(names) - len(refused)} designs measured, '
          f'{len(refused)} refused: {table}')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
