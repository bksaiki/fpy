"""
Measures format inference's precision, with and without digit bounds, on the
MMA-Sim designs: each site's static bound against every value it takes.

    python evaluation/analysis_precision/run.py -o results/         # every design
    python evaluation/analysis_precision/run.py -o results/ volta   # matching designs
    python evaluation/analysis_precision/run.py -o results/ -j 8 -n 1000

Writes `<out>/<design>.json` per design; exits 1 if any observed value lies
outside its static bound.
"""

import argparse
import dataclasses
import enum
import json
import sys
import time
import traceback
from functools import partial
from pathlib import Path
from typing import Any

from fpy2.utils import Location

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'examples' / 'mmasim'))

from compile import DESIGNS, _filename, in_processes
from inputs import SOURCES
from observe import Key, SiteResult, compare, compiler_view, key, observe, statics


class Analysis(enum.Enum):
    """The analyses compared: format inference alone, and with digit bounds."""

    FORMAT = 'format'
    DIGIT_BOUND = 'digit_bound'


def _loc(loc: Location | None) -> str | None:
    if loc is None:
        return None
    return f'{Path(loc.source).name}:{loc.start_line}:{loc.start_column}'


def _results(name: str, n: int, seed: int, source: str) -> dict[Analysis, dict[Key, SiteResult]]:
    """Each analysis's result at each site of design *name*."""
    func, arg_types = dict(DESIGNS)[name]()
    inputs = list(SOURCES[source].inputs(arg_types, n, seed))
    out = {}
    for a in Analysis:
        entry, analyses = compiler_view(func, arg_types, a is Analysis.DIGIT_BOUND)
        sites = statics(entry, analyses)
        obs = observe(entry, inputs, sites)
        out[a] = {key(s): compare(b, obs[s]) for s, b in sites.items()}
    return out


def measure(name: str, n: int, seed: int, source: str) -> dict[str, Any]:
    """Design *name*'s record: one entry per site, with each analysis's result,
    or the error that stopped it."""
    start = time.perf_counter()
    try:
        results = _results(name, n, seed, source)
    except Exception:  # noqa: BLE001 -- one design's failure is reported, not fatal
        return {'design': name, 'error': traceback.format_exc()}
    keys = sorted({k for r in results.values() for k in r}, key=repr)
    return {
        'design': name, 'inputs': source, 'n': n, 'seed': seed,
        'seconds': time.perf_counter() - start,
        'sites': [_site(k, results) for k in keys],
    }


def _site(k: Key, results: dict[Analysis, dict[Key, SiteResult]]) -> dict[str, Any]:
    calls, loc, name = k
    return {
        'calls': [_loc(c) for c in calls], 'loc': _loc(loc), 'name': name,
        **{a.value: dataclasses.asdict(r[k]) if k in r else None for a, r in results.items()},
    }


def _summary(record: dict[str, Any]) -> str:
    sites = record['sites']
    tops = [sum(bool(s[a.value] and s[a.value]['top']) for s in sites) for a in Analysis]
    unsound = sum(not s[a.value]['sound'] for s in sites for a in Analysis if s[a.value])
    return (f'{len(sites):3} sites  top {tops[0]:2} -> {tops[1]:2}  '
            f'unsound {unsound}  {record["seconds"]:.1f}s')


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('filter', nargs='*', metavar='SUBSTRING',
                    help='only designs whose name contains one of these')
    ap.add_argument('-o', '--out', metavar='DIR', type=Path, required=True,
                    help='write each design to DIR/<design>.json')
    ap.add_argument('-n', '--inputs-per-design', type=int, default=1000,
                    help='inputs per design (default 1000)')
    ap.add_argument('-s', '--seed', type=int, default=0, help='seed (default 0)')
    ap.add_argument('--inputs', choices=sorted(SOURCES), default='random',
                    help='input source (default random)')
    ap.add_argument('-j', '--jobs', type=int, default=1,
                    help='measure in this many processes (default 1)')
    args = ap.parse_args(argv)

    names = [name for name, _ in DESIGNS
             if not args.filter or any(f in name for f in args.filter)]
    if not names:
        ap.error(f'no design matches {args.filter}')
    args.out.mkdir(parents=True, exist_ok=True)

    width = max(map(len, names))
    unsound = failed = 0
    run = partial(measure, n=args.inputs_per_design, seed=args.seed, source=args.inputs)
    for name, record in zip(names, in_processes(run, names, args.jobs), strict=True):
        path = args.out / (_filename(name).removesuffix('.cpp') + '.json')
        path.write_text(json.dumps(record, indent=1))
        if 'error' in record:
            failed += 1
            print(f'{name:{width}}  error: {record["error"].strip().splitlines()[-1]}')
            continue
        unsound += any(not s[a.value]['sound'] for s in record['sites'] for a in Analysis
                       if s[a.value])
        print(f'{name:{width}}  {_summary(record)}')
    print(f'{len(names) - failed}/{len(names)} measured, {unsound} with an unsound site')
    return 1 if unsound or failed else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
