"""
Measures format inference's precision, with and without digit bounds, on the
MMA-Sim designs: each site's static bound against every value it takes.

    python evaluation/analysis/run.py           # everything, about 7 min
    python evaluation/analysis/run.py volta     # matching designs only

Replaces the results directory (`evaluation/results/analysis/`, or `-o DIR`)
with `<design>.json` per design, an `anf/` pass (sites counted with every
scalar subexpression named; analysis only), and `report.py`'s tables, figure
and `precision-summary.txt`.  Exits 1 if any observed value lies outside its
static bound or any design fails.
"""

import argparse
import dataclasses
import json
import shutil
import sys
import time
import traceback
from functools import partial
from pathlib import Path
from typing import Any

from fpy2.utils import Location

_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(_ROOT / 'evaluation' / 'common' / 'mma'), str(_ROOT / 'examples' / 'mmasim')]
sys.path.append(str(_ROOT / 'evaluation'))  # last: `config`, not shadowing this `run`

import config
import report
from compile import DESIGNS, _filename, in_processes
from inputs import SOURCES
from observe import (
    Analysis,
    Key,
    SiteResult,
    compare,
    compiler_view,
    key,
    observe_split,
    statics,
)

RESULTS = Path(__file__).resolve().parents[1] / 'results' / 'analysis'


def _foreign(out: Path) -> list[str]:
    """What in *out* a run did not write, and so must not be deleted."""
    if not out.exists():
        return []
    return sorted(p.name for p in out.iterdir()
                  if not (p.suffix == '.json' or p.name == 'anf' or p.name.startswith('precision-')))


def _loc(loc: Location | None) -> str | None:
    if loc is None:
        return None
    return f'{Path(loc.source).name}:{loc.start_line}:{loc.start_column}'


def _results(
    name: str, n: int, seed: int, source: str, anf: bool,
) -> dict[Analysis, dict[Key, SiteResult]]:
    """Each analysis's result at each site of design *name*."""
    func, arg_types = dict(DESIGNS)[name]()
    src = SOURCES[source]
    inputs = list(src.inputs(arg_types, n, seed))
    out = {}
    for a in Analysis:
        entry, analyses = compiler_view(func, arg_types, a is Analysis.DIGIT_BOUND, anf)
        sites = statics(entry, analyses)
        every, random = observe_split(entry, inputs, sites, lambda i: src.is_random(i, n))
        out[a] = {key(s): compare(b, every[s], random[s]) for s, b in sites.items()}
    return out


def measure(name: str, n: int, seed: int, source: str, anf: bool = False) -> dict[str, Any]:
    """Design *name*'s record: one entry per site, with each analysis's result,
    or the error that stopped it."""
    start = time.perf_counter()
    try:
        results = _results(name, n, seed, source, anf)
    except Exception:  # noqa: BLE001 -- one design's failure is reported, not fatal
        return {'design': name, 'error': traceback.format_exc()}
    keys = sorted({k for r in results.values() for k in r}, key=repr)
    return {
        'design': name, 'inputs': source, 'n': n, 'seed': seed, 'anf': anf,
        'random': sum(SOURCES[source].is_random(i, n) for i in range(n)),
        'seconds': time.perf_counter() - start,
        'sites': [_site(k, results) for k in keys],
    }


def _site(k: Key, results: dict[Analysis, dict[Key, SiteResult]]) -> dict[str, Any]:
    calls, loc, rhs, name = k
    return {
        'calls': [_loc(c) for c in calls], 'loc': _loc(loc), 'rhs': _loc(rhs), 'name': name,
        **{a.value: dataclasses.asdict(r[k]) if k in r else None for a, r in results.items()},
    }


def _summary(record: dict[str, Any]) -> str:
    sites = record['sites']
    tops = [sum(bool(s[a.value] and s[a.value]['top']) for s in sites) for a in Analysis]
    unsound = sum(not s[a.value]['sound'] for s in sites for a in Analysis if s[a.value])
    return (f'{len(sites):3} sites  top {tops[0]:2} -> {tops[1]:2}  '
            f'unsound {unsound}  {record["seconds"]:.1f}s')


def _measure_all(
    names: list[str], out: Path, jobs: int, **kw: Any,
) -> tuple[int, int]:
    """Measure *names* into *out*, one line each; how many failed, and how
    many have an unsound site."""
    out.mkdir(parents=True)
    width = max(map(len, names))
    unsound = failed = 0
    for name, record in zip(names, in_processes(partial(measure, **kw), names, jobs), strict=True):
        (out / (_filename(name).removesuffix('.cpp') + '.json')).write_text(json.dumps(record, indent=1))
        if 'error' in record:
            failed += 1
            print(f'{name:{width}}  error: {record["error"].strip().splitlines()[-1]}')
            continue
        unsound += any(not s[a.value]['sound'] for s in record['sites'] for a in Analysis
                       if s[a.value])
        print(f'{name:{width}}  {_summary(record)}')
    return failed, unsound


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('filter', nargs='*', metavar='SUBSTRING',
                    help='only designs whose name contains one of these')
    ap.add_argument('-o', '--out', metavar='DIR', type=Path, default=RESULTS,
                    help='the results directory, replaced (default %(default)s)')
    ap.add_argument('-n', '--inputs-per-design', type=int, default=config.ANALYSIS_INPUTS_PER_DESIGN,
                    help='inputs per design (default %(default)s)')
    ap.add_argument('-s', '--seed', type=int, default=config.ANALYSIS_SEED, help='seed (default %(default)s)')
    ap.add_argument('--inputs', choices=sorted(SOURCES), default=config.ANALYSIS_INPUTS,
                    help='input source (default %(default)s)')
    ap.add_argument('-j', '--jobs', type=int, default=8,
                    help='measure in this many processes (default 8)')
    args = ap.parse_args(argv)

    names = [name for name, _ in DESIGNS
             if not args.filter or any(f in name for f in args.filter)]
    if not names:
        ap.error(f'no design matches {args.filter}')
    if foreign := _foreign(args.out):
        ap.error(f'{args.out} holds files a run did not write: {", ".join(foreign)}')
    shutil.rmtree(args.out, ignore_errors=True)

    failed, unsound = _measure_all(
        names, args.out, args.jobs,
        n=args.inputs_per_design, seed=args.seed, source=args.inputs)
    print(f'{len(names) - failed}/{len(names)} measured, {unsound} with an unsound site')
    print('-- anf (analysis only)')
    anf_failed, _ = _measure_all(names, args.out / 'anf', args.jobs,
                                 n=0, seed=args.seed, source=args.inputs, anf=True)
    print('-- report')
    report.main([str(args.out)])
    return 1 if unsound or failed or anf_failed else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
