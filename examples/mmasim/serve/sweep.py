"""
A sweep of MMA design parameters, one at a time around each anchor.

Each point is a `generator` design: an anchor with one parameter changed.
Points are registered through `local.py --register`, compiled in parallel,
and run one `local.py` pass per scheme; a refused point is a row with its
reason.  Writes `<results>/sweep/sweep.csv` (a row per point: its
parameters, status and metrics against the quantized operands' exact
product) and `sweep.provenance.json`.  `--end-to-end` runs `frozen.py`
instead: each point's distance from the exact run, frozen to it (the
anchors also free, and R1 an ulp up or down at random as the floor), in
`sweep-e2e-<model>[-mtbench].csv`; `--free-only` then adds each point's free
run to that table.  `--grid` runs instead every combination of :data:`GRID`'s
axes around its anchor, frozen and free, into `sweep-grid-<model>.csv`
(`plot_grid.py`): the evaluation's design space.  `--manifest` compiles the
grid and writes `manifest.csv`: a row per design, its parameters, name and
whether it runs -- compiled, then on a GPU launched once (Triton compiles a
kernel at its first launch) -- the local error's list.

    python serve/sweep.py                          # from examples/mmasim
    python serve/sweep.py --anchors nv.hopper.e4m3.f32 --params F --tokens 512
    python serve/sweep.py --end-to-end --model Qwen/Qwen3.5-0.8B
    python serve/sweep.py --list                   # the points, and stop
"""

import argparse
import csv
import itertools
import math
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

# `serve`, for `core`; `examples/mmasim`, for `generator`
sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]

import generator
import torch
from core import cli, kernels, outputs, quant, swap
from generator.catalog import CATALOG

import fpy2 as fp

ANCHORS = {
    'nv.hopper.e4m3.f32': 'fp8-row',
    'nv.blackwell.e4m3.f32': 'fp8-row',
    'nv.hopper.bf16.f32': 'bf16',
    'nv.ampere.bf16.f32': 'bf16',
    'amd.cdna3.bf16': 'bf16',
}
"""Each anchor of the one-at-a-time sweep and the scheme its designs run under."""

SCHEMES = {**ANCHORS, 'nv.ada.e4m3.f32': 'fp8-row', 'amd.cdna2.bf16': 'bf16', 'amd.cdna3.fp8': 'fp8-row:fnuz'}
"""Every anchor's scheme, the grid's too."""


def _rho(bits: int | None = None, mode: str | None = None) -> Callable[[generator.Design], generator.Design]:
    """A change of `rho`: its fraction bits (exponents as FP32's), or its mode."""
    def change(d: generator.Design) -> generator.Design:
        m = d.rho.nbits - 1 - d.rho.es if bits is None else bits
        rm = d.rho.rm if mode is None else fp.RoundingMode[mode]
        return replace(d, rho=fp.IEEEContext(8, 9 + m, rm))
    return change


def _set(name: str, v: Any) -> Callable[[generator.Design], generator.Design]:
    """A change of one field; `L` keeps `k` whole instructions."""
    def change(d: generator.Design) -> generator.Design:
        out = replace(d, **{name: v})
        return replace(out, k=math.lcm(d.k, v)) if name == 'L' and d.scales is None else out
    return change


def _width(v: int) -> Callable[[generator.Design], generator.Design]:
    """`F` set to *v*; a separate join's `F2` with it, its guard bits kept."""
    def change(d: generator.Design) -> generator.Design:
        return replace(d, F=v, **({'F2': v + d.F2 - d.F} if d.join == 'separate' else {}))
    return change


def _guard(g: int) -> Callable[[generator.Design], generator.Design]:
    """A separate join's adder *g* bits wider than the sum: `F2 = F + g`."""
    def change(d: generator.Design) -> generator.Design:
        return replace(d, F2=d.F + g)
    return change


PARAMS: dict[str, tuple[Callable[[generator.Design], Any], list[tuple[Any, Callable]]]] = {
    'L': (lambda d: d.L, [(v, _set('L', v)) for v in (2, 4, 8, 16, 32, 64)]),
    'F': (lambda d: d.F, [(v, _width(v)) for v in (8, 10, 13, 16, 19, 22, 25, 28)]),
    'guard': (lambda d: d.F2 - d.F, [(v, _guard(v)) for v in (0, 2, 4, 7, 11)]),
    'rho_bits': (lambda d: d.rho.nbits - 1 - d.rho.es, [(v, _rho(bits=v)) for v in (7, 10, 13, 16, 19, 23)]),
    'rho_mode': (lambda d: d.rho.rm.name, [(v, _rho(mode=v)) for v in ('RTZ', 'RNE', 'RTN', 'RTP')]),
    'sum': (lambda d: d.sum, [(v, _set('sum', v)) for v in ('fused', 'tree', 'exact', 'fma')]),
    'join': (lambda d: d.join, [(v, _set('join', v)) for v in ('inside', 'separate')]),
    'groups': (lambda d: d.groups, [(v, _set('groups', v)) for v in ('none', 'even-odd')]),
    'ftz': (lambda d: d.ftz, [(v, _set('ftz', v)) for v in (False, True)]),
}
"""Each parameter: how to read it off a design, and its values, each with the
change that sets it."""

_UNUSED = {'guard': lambda d: d.join != 'separate',
           'F': lambda d: d.sum != 'fused' and d.join != 'separate'}
"""A parameter an anchor's structure does not read, so not swept around it."""


@dataclass(frozen=True)
class Point:
    anchor: str
    param: str
    value: Any
    design: generator.Design

    @property
    def is_anchor(self) -> bool:
        return self.design == CATALOG[self.anchor]


def points(anchors: list[str], params: list[str]) -> list[Point]:
    """Every anchor's points, a parameter at a time, its own value among
    them."""
    out = []
    for a in anchors:
        base = CATALOG[a]
        for p in params:
            if p in _UNUSED and _UNUSED[p](base):
                continue
            read, values = PARAMS[p]
            mine = list(values)
            if all(v != read(base) for v, _ in mine):
                mine.append((read(base), lambda d: d))
                if all(isinstance(v, int) and not isinstance(v, bool) for v, _ in mine):
                    mine.sort(key=lambda vc: vc[0])
            out += [Point(a, p, v, change(base)) for v, change in mine]
    return out


GRID = (('nv.hopper.bf16.f32', 'nv.ampere.bf16.f32', 'amd.cdna3.bf16', 'amd.cdna2.bf16',
         'nv.blackwell.e4m3.f32', 'nv.hopper.e4m3.f32', 'nv.ada.e4m3.f32', 'amd.cdna3.fp8'),
        (10, 13, 16, 19, 22, 25, 28))
"""`--grid`'s anchors, each generation's BF16 and FP8 instruction (MMA-Sim's
Blackwell BF16 is Hopper's, Ada's Ampere's; CDNA2 has no FP8), and the `F`
each is swept over, its own added; a design that does not read `F` (CDNA2's
tree) is its own alone.  Each anchor also with the other vendor's join:
structures no hardware (and no MMA-Sim model) has."""

SWAPPED_GUARD = 7
"""A join made separate is CDNA3's: its adder 7 bits wider than the sum."""


def _swap_join(d: generator.Design) -> generator.Design:
    """*d* with the other vendor's join: NVIDIA's `c` inside the fused sum,
    or CDNA3's added separately."""
    if d.join == 'inside':
        return replace(d, join='separate', F2=d.F + SWAPPED_GUARD)
    return replace(d, join='inside')


def grid() -> list[Point]:
    """Each of :data:`GRID`'s anchors over `F`, and with its join swapped
    (`join=swapped` in the label); its accumulator its own."""
    anchors, widths = GRID
    out = []
    for a in anchors:
        rho = CATALOG[a].rho
        acc = f'rho_bits={rho.nbits - 1 - rho.es} rho_mode={rho.rm.name}'
        for base, tail in ((CATALOG[a], ''), (_swap_join(CATALOG[a]), ' join=swapped')):
            mine = [base.F] if _UNUSED['F'](base) else sorted({*widths, base.F})
            out += [Point(a, 'grid', f'F={f} {acc}{tail}', _width(f)(base)) for f in mine]
    return out


def register() -> None:
    """Every point of the full sweep and the grid, as `local.py
    --register` calls it."""
    for d in dict.fromkeys(p.design for p in [*points(list(ANCHORS), list(PARAMS)), *grid()]):
        if generator.name(d) not in kernels.TILES:
            kernels.register_design(d)


def _launch(designs: list[tuple[str, str]]) -> dict[str, str]:
    """Each of *designs* (with its scheme) launched once on a small input; the
    ones that fail, and why."""
    failed = {}
    for name, scheme in designs:
        s = quant.scheme(scheme)
        x, w = (quant.quantize(torch.randn(r, 256, device='cuda'), op).elements for r, op in ((64, s.x), (128, s.w)))
        held = kernels.storage(name)
        try:
            kernels.matmul(x.to(held[0]), kernels.prepare(w, held[1], 1), name)
            torch.cuda.synchronize()
        except Exception as ex:  # noqa: BLE001 -- a failure is a result
            failed[name] = f'{type(ex).__name__} at launch: {str(ex).strip().splitlines()[-1][:200]}'
    return failed


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('--anchors', nargs='*', choices=list(ANCHORS), default=list(ANCHORS),
                    help='anchor designs (default: all)')
    ap.add_argument('--params', nargs='*', choices=list(PARAMS), default=list(PARAMS),
                    help='parameters to vary (default: all)')
    ap.add_argument('--model', default=swap.MODEL, help='the model (default: %(default)s)')
    ap.add_argument('--tokens', type=int, default=2048, help='rows captured (default: %(default)s)')
    ap.add_argument('--results', type=Path, default=cli.RESULTS,
                    help="the outputs' root (default: %(default)s)")
    ap.add_argument('-j', '--jobs', type=int, default=8,
                    help='processes compiling the designs (default: %(default)s)')
    ap.add_argument('--end-to-end', action='store_true',
                    help="run frozen.py instead of local.py (see above)")
    ap.add_argument('--segments', type=int, default=10,
                    help='with --end-to-end, WikiText-2 segments (default: %(default)s)')
    ap.add_argument('-w', '--workload', choices=['wikitext', 'mtbench'], default='wikitext',
                    help='with --end-to-end, the workload (default: %(default)s)')
    ap.add_argument('--grid', action='store_true',
                    help="with --end-to-end, :data:`GRID`'s combinations instead, frozen and free")
    ap.add_argument('--manifest', action='store_true',
                    help="compile the grid and write its manifest, manifest.csv")
    ap.add_argument('--free', action='store_true',
                    help='with --end-to-end, every point free as well as frozen')
    ap.add_argument('--free-only', action='store_true',
                    help="with --end-to-end, run every point free only, into the existing table")
    ap.add_argument('--seed', type=int, default=0,
                    help="with --end-to-end, the segments' and random rounding's seed (default: %(default)s)")
    ap.add_argument('--list', action='store_true', help='print the points, and stop')
    args = ap.parse_args(argv)

    if args.grid and not args.end_to_end:
        ap.error('--grid runs end to end')
    pts = grid() if args.grid or args.manifest else points(args.anchors, args.params)
    tag = args.model.rsplit('/', 1)[-1] + ('' if args.workload == 'wikitext' else f'-{args.workload}')
    kept: list[dict[str, str]] = []
    if args.grid and (done := args.results / 'sweep' / f'sweep-grid-{tag}.csv').exists():
        # the grid's table grows: only its points not yet run (a refused one is retried), and
        # only the grid's rows kept
        with done.open() as f:
            kept = list(csv.DictReader(f))
        kept = [r for r in kept
                if r['status'] == 'ok' and (r['anchor'], r['value']) in {(p.anchor, p.value) for p in pts}]
        pts = [p for p in pts if (p.anchor, p.value) not in {(r['anchor'], r['value']) for r in kept}]
    names = {d: generator.name(d) for d in dict.fromkeys(p.design for p in pts)}
    if args.list:
        for p in pts:
            print(f'{p.anchor:24} {p.param:9} {p.value!s:9}')
        print(f'{len(pts)} points, {len(names)} designs')
        return 0

    register()
    kernels.CACHE = cli.cache(args, 'kernels')
    refused = kernels.compile_designs(list(names.values()), args.jobs)
    out = args.results / 'sweep'
    if args.manifest:
        if torch.cuda.is_available():
            refused |= _launch([(names[p.design], SCHEMES[p.anchor]) for p in pts if names[p.design] not in refused])
        about = {p.design: {k: v for k, v in generator.about(p.design).items() if k != 'exact'} for p in pts}
        rows = [{'anchor': p.anchor, 'format': SCHEMES[p.anchor], 'design': names[p.design],
                 'status': 'refused' if names[p.design] in refused else 'ok',
                 'reason': refused.get(names[p.design], ''), **about[p.design]} for p in pts]
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
        common = ['--model', args.model, '--scheme', scheme, '--results', str(args.results),
                  '--register', str(Path(__file__).resolve())]
        if args.end_to_end:
            script, json_out = 'frozen.py', out / f'e2e-{tag}-{scheme}.json'
            anchors = [n for a in args.anchors if ANCHORS[a] == scheme and (n := names.get(CATALOG[a])) in run]
            frozen = [] if args.free_only else run
            free = run if args.free or args.free_only or args.grid else anchors
            extra = ['--segments', str(args.segments), '-w', args.workload, '--seed', str(args.seed),
                     '--matmuls', f'{scheme}-exact-rand', *frozen,
                     '--free', f'{scheme}-exact-rand', *free]
        else:
            script, json_out = 'local.py', out / f'local-{scheme}.json'
            extra = ['--tokens', str(args.tokens), '-j', str(args.jobs), '-d', *run]
        done = subprocess.run([sys.executable, str(Path(__file__).resolve().parent / script),
                               *common, *extra, '-o', str(json_out)], check=False)
        if done.returncode:
            raise SystemExit(f'{script} under {scheme} exited {done.returncode}')
        with json_out.with_suffix('.csv').open() as f:
            for row in csv.DictReader(f):
                if args.end_to_end or (row['reference'], row['group']) == ('quantized', 'all'):
                    metrics[row['design']] = row
    rows = []
    for p in pts:
        about = generator.about(p.design)
        about.pop('exact')
        n = names[p.design]
        m = metrics.get(n, {})
        rows.append({'anchor': p.anchor, 'scheme': SCHEMES[p.anchor], 'param': p.param,
                     'value': p.value, 'is_anchor': p.is_anchor,
                     'status': 'refused' if n in refused else 'ok', 'reason': refused.get(n, ''),
                     'design': n, **about,
                     **({'floor_kl': metrics[f'{SCHEMES[p.anchor]}-exact-rand']['kl'],
                         'k0_kl': metrics[f'{SCHEMES[p.anchor]}-exact-rand']['free_kl']}
                        if args.end_to_end else {}),
                     **{k: v for k, v in m.items()
                        if k not in ('model', 'reference', 'design', 'group')}})
    table = out / (f'sweep-grid-{tag}.csv' if args.grid else
                   f'sweep-e2e-{tag}.csv' if args.end_to_end else 'sweep.csv')
    if args.end_to_end and not (args.free_only or args.grid) and table.exists() and (
            set(args.anchors) != set(ANCHORS) or set(args.params) != set(PARAMS)):
        # a partial rerun: these anchors' and parameters' rows replace the table's
        ran = {(p.anchor, p.param) for p in pts}
        with table.open() as f:
            rows = [r for r in csv.DictReader(f)
                    if r['param'] in PARAMS and (r['anchor'], r['param']) not in ran] + rows
    if args.free_only:
        # the frozen table's rows for these points, each point's free run added
        key = lambda r: (r["anchor"], r["param"], str(r["value"]))
        new = {key(r): r for r in rows}
        with table.open() as f:
            rows = [{**r, **{k: v for k, v in new[key(r)].items() if k.startswith('free_') or k == 'k0_kl'}}
                    for r in csv.DictReader(f) if key(r) in new]
    outputs.write_csv(table, kept + rows)
    outputs.write_json(table.with_suffix('.provenance.json'), {
        'argv': sys.argv, **outputs.commit(),
        'settings': {k: v if isinstance(v, (int, list)) else str(v) for k, v in vars(args).items()}})
    print(f'{len(rows)} points, {len(names) - len(refused)} designs measured, '
          f'{len(refused)} refused: {table}')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
