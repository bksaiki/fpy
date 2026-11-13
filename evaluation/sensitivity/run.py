"""
Compiles the MMA-Sim designs under each digit-bound solver backend, timing the
compilation and the solver queries inside it.

    python evaluation/sensitivity/run.py               # everything, about 30 min at -j 4
    python evaluation/sensitivity/run.py volta -r 1    # matching designs, once each
    python evaluation/sensitivity/run.py --smoke       # the first 2 designs, once each

Replaces the results directory (`evaluation/results/sensitivity/`, or `-o
DIR`) with `<design>.json` per design, `sensitivity-run.json`, and
`report.py`'s table and figure.  Each compilation runs in a fresh process,
under a wall-clock limit; each backend call inside it under a limit of its
own (`-c`).  Exits 1 if a backend fails a design z3 compiles.

A design counts as compiled under a backend only where every repeat
compiled.
"""

import argparse
import dataclasses
import hashlib
import json
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(_ROOT / 'evaluation' / 'common' / 'mma'), str(_ROOT / 'examples' / 'mmasim')]
sys.path.append(str(_ROOT / 'evaluation'))  # last: `config`, not shadowing this `run`

import config
import cvc5_solver  # adds backend `cvc5`
from compile import DESIGNS, _filename

from fpy2.analysis.digit_bound import BACKENDS, SolverStats, make_solver, solver_backend

RESULTS = Path(__file__).resolve().parents[1] / 'results' / 'sensitivity'


def _record(name: str) -> str:
    """The file design *name*'s record is written to."""
    return _filename(name).removesuffix('.cpp') + '.json'


def _foreign(out: Path) -> list[str]:
    """What in *out* a run did not write, and so must not be deleted."""
    if not out.exists():
        return []
    own = {_record(name) for name, _ in DESIGNS}
    return sorted(p.name for p in out.iterdir()
                  if not (p.name in own or p.name.startswith('sensitivity-')))


def compile_once(
    name: str, backend: str, call_timeout: float = config.SENSITIVITY_CALL_TIMEOUT,
) -> dict[str, Any]:
    """Compile design *name* under *backend*, in this process, each backend
    call limited to *call_timeout* seconds: the wall time, the solver
    statistics summed over every store, each query (`trace`, as columns), and
    the C++'s hash, or the error that stopped it."""
    from compile import cpp_compiler, prepare

    build = dict(DESIGNS)[name]
    # untimed: a backend's first solver may warm up its library
    make_solver(backend)
    start = time.perf_counter()
    with solver_backend(
            backend, timeout_ms=round(1000 * call_timeout), keep_trace=True) as solvers:
        try:
            # the C++ alone, as `compilation/compile_times.py` times it (no signature)
            src = cpp_compiler().compile_module(prepare(build))
            outcome = {'cpp_sha256': hashlib.sha256(src.encode()).hexdigest()}
        except Exception:  # noqa: BLE001 -- one design's failure is reported, not fatal
            outcome = {'error': traceback.format_exc()}
    seconds = time.perf_counter() - start
    stats = sum((s.stats for s in solvers), SolverStats())
    queries = [q for s in solvers for q in s.trace]
    trace = {
        'constraints': [q.constraints for q in queries],
        'seconds': [round(q.seconds, 7) for q in queries],
    }
    return {'seconds': seconds, 'stats': dataclasses.asdict(stats), 'trace': trace, **outcome}


def _run_child(
    name: str, backend: str, timeout: float, call_timeout: float,
) -> dict[str, Any]:
    """:func:`compile_once` in a fresh process, killed past *timeout*
    seconds."""
    cmd = [sys.executable, __file__, '--child', name, backend, str(call_timeout)]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return {'timeout': timeout}
    if proc.returncode != 0:
        return {'error': proc.stderr.strip() or f'exit {proc.returncode}'}
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        return {'error': f'no result on stdout: {proc.stdout[-500:]!r}'}


def _summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """One backend's runs on one design: their outcome, and the median run by
    compile time.

    The outcome is `ok` only where every run compiled: a median over the
    runs that happened to finish would hide the others.  Otherwise it names
    each way the rest failed (`error`, `timeout`, or `error+timeout`), and
    `compiled` how many did not fail.

    The time, the counts and the queries are the median run's.  The counts
    are the same every run unless a backend call hit its limit in some runs
    and not others, which nothing checks: `runs` keeps each run's to compare.
    Only that run's queries are kept: the others' would triple the results."""
    ok = [r for r in runs if 'cpp_sha256' in r]
    kept = [{k: v for k, v in r.items() if k != 'trace'} for r in runs]
    if len(ok) < len(runs):
        failed = sorted({'timeout' if 'timeout' in r else 'error'
                         for r in runs if 'cpp_sha256' not in r})
        return {'runs': kept, 'outcome': '+'.join(failed), 'compiled': len(ok)}
    # the lower median on an even count: a run that happened
    mid = sorted(ok, key=lambda r: r['seconds'])[(len(ok) - 1) // 2]
    return {
        'runs': kept, 'outcome': 'ok', 'cpp_sha256': mid['cpp_sha256'],
        'seconds': mid['seconds'], 'solver_time': mid['stats']['solver_time'],
        'stats': mid['stats'], 'trace': mid['trace'],
    }


def measure(
    names: list[str], backends: list[str], repeats: int, jobs: int, timeout: float,
    call_timeout: float, out: Path,
) -> tuple[int, list[str]]:
    """Every design in *names* under every backend, *repeats* times, into
    *out*: how many records were written, and the designs some backend
    failed where z3 compiled."""
    tasks = [(n, b, r) for n in names for b in backends for r in range(repeats)]
    results: dict[tuple[str, str], dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {t: pool.submit(_run_child, t[0], t[1], timeout, call_timeout)
                   for t in tasks}
        width = max(map(len, names))
        for name in names:
            record: dict[str, Any] = {'design': name, 'repeats': repeats, 'backends': {}}
            for b in backends:
                runs = [futures[(name, b, r)].result() for r in range(repeats)]
                record['backends'][b] = results[(name, b)] = _summarize(runs)
            (out / _record(name)).write_text(
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


def main(argv: list[str]) -> int:
    if argv[:1] == ['--child']:
        name, backend, call_timeout = argv[1:4]
        print(json.dumps(compile_once(name, backend, float(call_timeout))))
        return 0

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('filter', nargs='*', metavar='SUBSTRING',
                    help='only designs whose name contains one of these')
    ap.add_argument('-o', '--out', metavar='DIR', type=Path, default=RESULTS,
                    help='the results directory, replaced (default %(default)s)')
    ap.add_argument('-b', '--backends', nargs='+', choices=list(BACKENDS),
                    default=list(config.SENSITIVITY_BACKENDS),
                    help='backends to compare, the first the reference (default %(default)s)')
    ap.add_argument('-r', '--repeats', type=int, default=config.SENSITIVITY_REPEATS,
                    help='compilations per design and backend; the median is reported '
                         '(default %(default)s)')
    ap.add_argument('-j', '--jobs', type=int, default=1,
                    help='compilations at once (default %(default)s, the least noise)')
    ap.add_argument('-t', '--timeout', type=float, default=config.SENSITIVITY_TIMEOUT,
                    help='seconds one compilation may take (default %(default)s)')
    ap.add_argument('-c', '--call-timeout', type=float, default=config.SENSITIVITY_CALL_TIMEOUT,
                    help='seconds one backend call may take (default %(default)s; '
                         'the solvers alone default to 10)')
    ap.add_argument('--smoke', action='store_true', help='the first 2 designs, once each')
    args = ap.parse_args(argv)
    if args.smoke:
        args.repeats = 1

    names = [name for name, _ in DESIGNS
             if not args.filter or any(f in name for f in args.filter)]
    if args.smoke:
        names = names[:2]
    if not names:
        ap.error(f'no design matches {args.filter}')
    if foreign := _foreign(args.out):
        ap.error(f'{args.out} holds files a run did not write: {", ".join(foreign)}')
    shutil.rmtree(args.out, ignore_errors=True)
    args.out.mkdir(parents=True)

    (args.out / 'sensitivity-run.json').write_text(json.dumps({
        'backends': args.backends, 'repeats': args.repeats, 'jobs': args.jobs,
        'timeout': args.timeout, 'call_timeout': args.call_timeout, 'designs': names,
    }, indent=1))
    n, broken = measure(names, args.backends, args.repeats, args.jobs, args.timeout,
                        args.call_timeout, args.out)
    print(f'{n} designs measured; {len(broken)} failed by a backend z3 compiles'
          + (f': {", ".join(broken)}' if broken else ''))
    print('-- report')
    # here, not at the top: every `--child` would load matplotlib for nothing
    import report
    report.main([str(args.out)])
    return 1 if broken else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
