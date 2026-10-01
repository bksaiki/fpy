"""
A sweep of hardware design parameters (`docs/todos/mmasim-serving.md`,
Stage 3): around each anchor design, one parameter at a time over its values,
every other held at the anchor's; each point's local metrics on cached
activations; one table.

A point the generator cannot form, or the compiler refuses, is a row with its
reason.  Every formable point is registered (this file is `local.py`'s
`--register`), compiled in parallel, and the compiled ones of each scheme run
in one `local.py` pass.  The table, `<results>/sweep/sweep.csv`: a row per
point -- its anchor, the parameter varied and its value, every parameter, its
status, and the design's metrics against the quantized operands' exact
product -- and beside it what it came from (`sweep.provenance.json`).

    python serve/sweep.py                          # from examples/mmasim
    python serve/sweep.py --anchors nv.hopper.e4m3.f32 --params F --tokens 512
    python serve/sweep.py --list                   # the points, and stop
"""

import argparse
import csv
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
from core import cli, kernels, outputs
from generator.catalog import CATALOG

import fpy2 as fp

ANCHORS = {
    'nv.hopper.e4m3.f32': 'fp8-row',
    'nv.blackwell.e4m3.f32': 'fp8-row',
    'nv.hopper.bf16.f32': 'bf16',
    'nv.ampere.bf16.f32': 'bf16',
    'amd.cdna3.bf16': 'bf16',
}
"""Each anchor and the scheme its designs run under."""


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


PARAMS: dict[str, tuple[Callable[[generator.Design], Any], list[tuple[Any, Callable]]]] = {
    'L': (lambda d: d.L, [(v, _set('L', v)) for v in (2, 4, 8, 16, 32, 64)]),
    'F': (lambda d: d.F, [(v, _set('F', v)) for v in (8, 10, 13, 16, 19, 22, 25, 28, 32, 35)]),
    'F2': (lambda d: d.F2, [(v, _set('F2', v)) for v in (16, 20, 24, 28, 31, 35)]),
    'rho_bits': (lambda d: d.rho.nbits - 1 - d.rho.es, [(v, _rho(bits=v)) for v in (7, 10, 13, 16, 19, 23)]),
    'rho_mode': (lambda d: d.rho.rm.name, [(v, _rho(mode=v)) for v in ('RTZ', 'RNE', 'RTN', 'RTP')]),
    'sum': (lambda d: d.sum, [(v, _set('sum', v)) for v in ('fused', 'tree', 'exact', 'fma')]),
    'join': (lambda d: d.join, [(v, _set('join', v)) for v in ('inside', 'separate')]),
    'groups': (lambda d: d.groups, [(v, _set('groups', v)) for v in ('none', 'even-odd')]),
    'ftz': (lambda d: d.ftz, [(v, _set('ftz', v)) for v in (False, True)]),
}
"""Each parameter: how to read it off a design, and its values, each with the
change that sets it."""

_UNUSED = {'F2': lambda d: d.join != 'separate', 'F': lambda d: d.sum != 'fused'}
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
            mine = [(v, change) for v, change in values]
            if all(v != read(base) for v, _ in mine):
                mine.append((read(base), lambda d: d))
                if all(isinstance(v, int) and not isinstance(v, bool) for v, _ in mine):
                    mine.sort(key=lambda vc: vc[0])
            out += [Point(a, p, v, change(base)) for v, change in mine]
    return out


def register() -> None:
    """Every formable point of the full sweep, as `local.py --register` calls
    it."""
    for d in dict.fromkeys(p.design for p in points(list(ANCHORS), list(PARAMS))):
        if generator.unformable(d) is None and generator.name(d) not in kernels.TILES:
            kernels.register_design(d)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('--anchors', nargs='*', choices=list(ANCHORS), default=list(ANCHORS),
                    help='anchor designs (default: all)')
    ap.add_argument('--params', nargs='*', choices=list(PARAMS), default=list(PARAMS),
                    help='parameters to vary (default: all)')
    ap.add_argument('--model', default='Qwen/Qwen3-0.6B', help='the model (default: %(default)s)')
    ap.add_argument('--tokens', type=int, default=2048, help='rows captured (default: %(default)s)')
    ap.add_argument('--results', type=Path, default=cli.RESULTS,
                    help="the outputs' root (default: %(default)s)")
    ap.add_argument('-j', '--jobs', type=int, default=8,
                    help='processes compiling the designs (default: %(default)s)')
    ap.add_argument('--list', action='store_true', help='print the points, and stop')
    args = ap.parse_args(argv)

    pts = points(args.anchors, args.params)
    status = {}
    for d in dict.fromkeys(p.design for p in pts):
        status[d] = ('unformable', why) if (why := generator.unformable(d)) else ('ok', '')
    if args.list:
        for p in pts:
            print(f'{p.anchor:24} {p.param:9} {p.value!s:9} {status[p.design][0]:10} '
                  f'{status[p.design][1]}')
        print(f'{len(pts)} points, {len(status)} designs')
        return 0

    register()
    kernels.CACHE = args.results / 'cache' / 'kernels'
    names = {d: generator.name(d) for d, (s, _) in status.items() if s == 'ok'}
    refused = kernels.compile_designs(list(names.values()), args.jobs)
    for d, n in names.items():
        if n in refused:
            status[d] = ('refused', refused[n])
    out = args.results / 'sweep'
    metrics: dict[str, dict[str, str]] = {}
    for scheme in sorted({ANCHORS[p.anchor] for p in pts}):
        run = [names[d] for d in dict.fromkeys(p.design for p in pts if ANCHORS[p.anchor] == scheme)
               if status[d][0] == 'ok']
        if not run:
            continue
        json_out = out / f'local-{scheme}.json'
        done = subprocess.run(
            [sys.executable, str(Path(__file__).resolve().parent / 'local.py'),
             '--model', args.model, '--scheme', scheme, '--tokens', str(args.tokens),
             '--register', str(Path(__file__).resolve()), '--results', str(args.results),
             '-j', str(args.jobs), '-d', *run, '-o', str(json_out)], check=False)
        if done.returncode:
            raise SystemExit(f'local.py under {scheme} exited {done.returncode}')
        with json_out.with_suffix('.csv').open() as f:
            for row in csv.DictReader(f):
                if (row['reference'], row['group']) == ('quantized', 'all'):
                    metrics[row['design']] = row
    rows = []
    for p in pts:
        about = generator.about(p.design)
        about.pop('exact')
        s, why = status[p.design]
        m = metrics.get(names.get(p.design, ''), {})
        rows.append({'anchor': p.anchor, 'scheme': ANCHORS[p.anchor], 'param': p.param,
                     'value': p.value, 'is_anchor': p.is_anchor, 'status': s, 'reason': why,
                     'design': names.get(p.design, ''), **about,
                     **{k: v for k, v in m.items()
                        if k not in ('model', 'reference', 'design', 'group')}})
    outputs.write_csv(out / 'sweep.csv', rows)
    outputs.write_json(out / 'sweep.provenance.json', {
        'argv': sys.argv, **outputs.commit(),
        'settings': {k: v if isinstance(v, (int, list)) else str(v) for k, v in vars(args).items()}})
    print(f'{len(rows)} points, {sum(s == "ok" for s, _ in status.values())} designs measured, '
          f'{sum(s == "refused" for s, _ in status.values())} refused, '
          f'{sum(s == "unformable" for s, _ in status.values())} unformable: {out / "sweep.csv"}')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
