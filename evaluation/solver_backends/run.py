"""
Compiles the MMA-Sim designs under each digit-bound solver backend, timing the
compilation and the solver queries inside it.

    python evaluation/solver_backends/run.py               # everything
    python evaluation/solver_backends/run.py volta -r 1    # matching designs, once each

Replaces the results directory (`results/` beside this file, or `-o DIR`)
with `<design>.json` per design, and `report.py`'s table and figure.  Each compilation runs in a fresh process, under a
wall-clock limit.  Exits 1 if a backend fails a design z3 compiles.
"""

import argparse
import dataclasses
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'examples' / 'mmasim'))

import report
from compile import DESIGNS, _filename

RESULTS = Path(__file__).resolve().parent / 'results'

BACKENDS = ('z3', 'z3-opt', 'cvc5', 'highs')
"""The backends compared, z3 first: the others are read against it."""


def compile_once(name: str, backend: str) -> dict[str, Any]:
    """Compile design *name* under *backend*, in this process: the wall time,
    the solver statistics summed over every store, and the C++'s hash, or
    the error that stopped it."""
    from compile import compile_design

    from fpy2.analysis.digit_bound import SolverStats, solver_backend

    build = dict(DESIGNS)[name]
    start = time.perf_counter()
    with solver_backend(backend) as solvers:
        try:
            src, _ = compile_design(build)
            outcome = {'cpp_sha256': hashlib.sha256(src.encode()).hexdigest()}
        except Exception:  # noqa: BLE001 -- one design's failure is reported, not fatal
            outcome = {'error': traceback.format_exc()}
    seconds = time.perf_counter() - start
    stats = sum((s.stats for s in solvers), SolverStats())
    return {'seconds': seconds, 'stats': dataclasses.asdict(stats), **outcome}


def _run_child(name: str, backend: str, timeout: float) -> dict[str, Any]:
    """:func:`compile_once` in a fresh process, killed past *timeout*
    seconds."""
    # one thread: the backends are compared as single-threaded solvers
    env = {**os.environ, 'OMP_NUM_THREADS': '1'}
    cmd = [sys.executable, __file__, '--child', name, backend]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, env=env, check=False)
    except subprocess.TimeoutExpired:
        return {'timeout': timeout}
    if proc.returncode != 0:
        return {'error': proc.stderr.strip() or f'exit {proc.returncode}'}
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """One backend's runs on one design: medians over the runs that
    compiled, and their outcome."""
    ok = [r for r in runs if 'cpp_sha256' in r]
    if not ok:
        return {'runs': runs, 'outcome': 'timeout' if 'timeout' in runs[0] else 'error'}
    return {
        'runs': runs, 'outcome': 'ok', 'cpp_sha256': ok[0]['cpp_sha256'],
        'seconds': statistics.median(r['seconds'] for r in ok),
        'solver_time': statistics.median(r['stats']['solver_time'] for r in ok),
        # the counts are the same every run; only the times vary
        'stats': ok[0]['stats'],
    }


def measure(
    names: list[str], backends: list[str], repeats: int, jobs: int, timeout: float,
    out: Path,
) -> tuple[int, list[str]]:
    """Every design in *names* under every backend, *repeats* times, into
    *out*: how many records were written, and the designs some backend
    failed where z3 compiled."""
    tasks = [(n, b, r) for n in names for b in backends for r in range(repeats)]
    results: dict[tuple[str, str], list[dict[str, Any]]] = {}
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {t: pool.submit(_run_child, t[0], t[1], timeout) for t in tasks}
        width = max(map(len, names))
        for name in names:
            record: dict[str, Any] = {'design': name, 'repeats': repeats, 'backends': {}}
            for b in backends:
                runs = [futures[(name, b, r)].result() for r in range(repeats)]
                record['backends'][b] = results[(name, b)] = _summarize(runs)
            (out / (_filename(name).removesuffix('.cpp') + '.json')).write_text(
                json.dumps(record, indent=1))
            print(f'{name:{width}}  {_line(record)}', flush=True)
    broken = [
        n for n in names
        if results[(n, backends[0])]['outcome'] == 'ok'
        and any(results[(n, b)]['outcome'] != 'ok' for b in backends[1:])
    ]
    return len(names), broken


def _line(record: dict[str, Any]) -> str:
    parts = []
    for b, r in record['backends'].items():
        if r['outcome'] != 'ok':
            parts.append(f'{b} {r["outcome"]}')
        else:
            parts.append(f'{b} {r["seconds"]:6.2f}s (solver {r["solver_time"]:6.2f}s)')
    return '  '.join(parts)


def _foreign(out: Path) -> list[str]:
    """What in *out* a run did not write, and so must not be deleted."""
    if not out.exists():
        return []
    return sorted(p.name for p in out.iterdir()
                  if not (p.suffix == '.json' or p.name.startswith('backends-')))


def main(argv: list[str]) -> int:
    if argv[:1] == ['--child']:
        name, backend = argv[1:3]
        print(json.dumps(compile_once(name, backend)))
        return 0

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('filter', nargs='*', metavar='SUBSTRING',
                    help='only designs whose name contains one of these')
    ap.add_argument('-o', '--out', metavar='DIR', type=Path, default=RESULTS,
                    help='the results directory, replaced (default results/ here)')
    ap.add_argument('-b', '--backends', nargs='+', choices=BACKENDS, default=list(BACKENDS),
                    help='backends to compare, the first the reference (default all, z3 first)')
    ap.add_argument('-r', '--repeats', type=int, default=3,
                    help='compilations per design and backend; the median is reported (default 3)')
    ap.add_argument('-j', '--jobs', type=int, default=4,
                    help='compilations at once (default 4); 1 for the least noise')
    ap.add_argument('-t', '--timeout', type=float, default=600,
                    help='seconds one compilation may take (default 600)')
    args = ap.parse_args(argv)

    names = [name for name, _ in DESIGNS
             if not args.filter or any(f in name for f in args.filter)]
    if not names:
        ap.error(f'no design matches {args.filter}')
    if foreign := _foreign(args.out):
        ap.error(f'{args.out} holds files a run did not write: {", ".join(foreign)}')
    shutil.rmtree(args.out, ignore_errors=True)
    args.out.mkdir(parents=True)

    (args.out / 'backends-run.json').write_text(json.dumps({
        'backends': args.backends, 'repeats': args.repeats, 'jobs': args.jobs,
        'timeout': args.timeout, 'designs': names,
    }, indent=1))
    n, broken = measure(names, args.backends, args.repeats, args.jobs, args.timeout, args.out)
    print(f'{n} designs measured; {len(broken)} failed by a backend z3 compiles'
          + (f': {", ".join(broken)}' if broken else ''))
    print('-- report')
    report.main([str(args.out)])
    return 1 if broken else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
