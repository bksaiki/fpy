"""
The grid evaluation: each of :data:`GRID_ANCHORS` over `F`, and with its join swapped.

Each design runs through `frozen.py`, a pass per scheme over the same
segments: frozen (amplified) and free (observed), against the scheme's exact
run, with R1 an ulp up or down at random as the floor.  The table,
`<results>/sweep/sweep-grid-<model>.csv` (`plot_grid.py`), grows: a rerun
keeps its `ok` rows whose design is unchanged and runs the rest.
`--manifest` instead compiles the grid, launches each design as it will run
(at each of `--model`'s layer widths), and writes `manifest.csv`: a row per
design, its parameters, name, status and compile time.

    python serve/sweep.py --manifest --model Qwen/Qwen3.5-0.8B   # from examples/mmasim
    python serve/sweep.py --model Qwen/Qwen3.5-0.8B
    python serve/sweep.py --list                                 # the points, and stop
"""

import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

# `serve`, for `core`; `examples/mmasim`, for `generator`
sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]

import generator
import torch
from core import cli, kernels, outputs, quant, swap
from generator.catalog import CATALOG

GRID_ANCHORS = ('nv.hopper.bf16.f32', 'nv.ampere.bf16.f32', 'amd.cdna3.bf16', 'amd.cdna2.bf16',
                'nv.blackwell.mx.e4m3', 'nv.hopper.e4m3.f32', 'nv.ada.e4m3.f32', 'amd.cdna3.fp8')
"""The grid's anchors: each generation's BF16 and FP8 instruction (MMA-Sim's
Blackwell BF16 is Hopper's, Ada's Ampere's; CDNA2 has no FP8); see :func:`grid`."""

SCHEMES = {'nv.hopper.bf16.f32': 'bf16', 'nv.ampere.bf16.f32': 'bf16', 'amd.cdna3.bf16': 'bf16',
           'amd.cdna2.bf16': 'bf16', 'nv.blackwell.mx.e4m3': 'mxfp8', 'nv.hopper.e4m3.f32': 'fp8-row',
           'nv.ada.e4m3.f32': 'fp8-row', 'amd.cdna3.fp8': 'fp8-row:fnuz'}
"""Each anchor's scheme, its designs' too."""

GRID_F = (10, 13, 16, 19, 22, 25, 28)
"""The `F` each anchor is swept over, its own added."""

SWAPPED_GUARD = CATALOG['amd.cdna3.bf16'].F2 - CATALOG['amd.cdna3.bf16'].F
"""A join made separate is CDNA3's: its adder this many bits wider than the sum."""


@dataclass(frozen=True)
class Point:
    anchor: str
    value: str
    """Its label: its `F`, its accumulator, and whether its join is swapped."""
    design: generator.Design

    @property
    def is_anchor(self) -> bool:
        return self.design == CATALOG[self.anchor]


def _reads_F(d: generator.Design) -> bool:
    """Whether *d*'s structure reads `F`: a fused sum or a separate join."""
    return d.sum == 'fused' or d.join == 'separate'


def _width(d: generator.Design, f: int) -> generator.Design:
    """*d* with `F` set to *f*; a separate join's `F2` with it, its guard bits kept."""
    return replace(d, F=f, **({'F2': f + d.F2 - d.F} if d.join == 'separate' else {}))


def _swap_join(d: generator.Design) -> generator.Design:
    """*d* with the other vendor's join: NVIDIA's `c` inside the fused sum,
    or CDNA3's added separately."""
    if d.join == 'inside':
        return replace(d, join='separate', F2=d.F + SWAPPED_GUARD)
    return replace(d, join='inside')


def grid() -> list[Point]:
    """Each anchor over `F`, and with its join swapped (`join=swapped` in the
    label); its accumulator its own.  A design that does not read `F`
    (CDNA2's tree) is its own alone."""
    out = []
    for a in GRID_ANCHORS:
        rho = CATALOG[a].rho
        acc = f'rho_bits={rho.nbits - 1 - rho.es} rho_mode={rho.rm.name}'
        for base, tail in ((CATALOG[a], ''), (_swap_join(CATALOG[a]), ' join=swapped')):
            mine = sorted({*GRID_F, base.F}) if _reads_F(base) else [base.F]
            out += [Point(a, f'F={f} {acc}{tail}', _width(base, f)) for f in mine]
    return out


def register() -> None:
    """Every grid design, as `frozen.py --register` calls it."""
    for d in dict.fromkeys(p.design for p in grid()):
        if generator.name(d) not in kernels.TILES:
            kernels.register_design(d)


def _launch(pts: list[Point], names: dict[generator.Design, str], model: str) -> dict[str, str]:
    """Each of *pts*' designs launched at each of *model*'s layer widths; the
    ones that fail, and why.  Stops at an error that leaves the GPU unusable,
    which every later design would hit."""
    failed: dict[str, str] = {}
    for scheme in sorted({SCHEMES[p.anchor] for p in pts}):
        s = quant.scheme(scheme)
        net, run = swap.load(model, s)
        ks = cli.layer_ks(net, run)
        del net, run
        torch.cuda.empty_cache()
        mine = list(dict.fromkeys(names[p.design] for p in pts if SCHEMES[p.anchor] == scheme))
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


def _about(d: generator.Design) -> dict[str, Any]:
    """*d*'s parameters, readably (`generator.about` but its exact form)."""
    return {k: v for k, v in generator.about(d).items() if k != 'exact'}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('--model', choices=swap.MODELS, default=swap.MODEL,
                    help='the model (default: %(default)s)')
    ap.add_argument('--results', type=Path, default=cli.RESULTS,
                    help="the outputs' root (default: %(default)s)")
    ap.add_argument('-j', '--jobs', type=int, default=8,
                    help='processes compiling the designs (default: %(default)s)')
    ap.add_argument('--segments', type=int, default=10,
                    help='WikiText-2 segments (default: %(default)s)')
    ap.add_argument('--seed', type=int, default=0,
                    help="the segments' and random rounding's seed (default: %(default)s)")
    ap.add_argument('--manifest', action='store_true',
                    help='compile and launch the grid as --model runs it, and write manifest.csv')
    ap.add_argument('--list', action='store_true', help='print the points, and stop')
    args = ap.parse_args(argv)

    pts = grid()
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
        rows = [{'anchor': p.anchor, 'format': SCHEMES[p.anchor], 'design': names[p.design],
                 'status': 'refused' if names[p.design] in refused else 'ok',
                 'reason': refused.get(names[p.design], ''),
                 'compile_seconds': kernels.compile_seconds(names[p.design]), **_about(p.design)} for p in pts]
        outputs.write_csv(out / 'manifest.csv', rows)
        print(f'{len(rows)} designs, {len(rows) - len(refused)} run, {len(refused)} refused: '
              f'{out / "manifest.csv"}')
        return 0
    metrics: dict[str, dict[str, str]] = {}
    for scheme in sorted({SCHEMES[p.anchor] for p in pts}):
        run = [names[d] for d in dict.fromkeys(p.design for p in pts if SCHEMES[p.anchor] == scheme)
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
        floor = metrics.get(f'{SCHEMES[p.anchor]}-exact-rand', {})
        rows.append({'anchor': p.anchor, 'scheme': SCHEMES[p.anchor], 'value': p.value,
                     'is_anchor': p.is_anchor, 'status': 'refused' if n in refused else 'ok',
                     'reason': refused.get(n, ''), 'design': n, **_about(p.design),
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
