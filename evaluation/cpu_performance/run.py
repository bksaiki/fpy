"""
Times FPy's C++ for every MMA-Sim design against MMA-Sim itself, on the CPU:
batches of dot products, each chained one K-wide call at a time from Python,
by `torch.ops.fpy2_models.<op>` and by MMA-Sim's `dpa`.  Every configuration
is first checked to agree with MMA-Sim bit for bit.

    python evaluation/cpu_performance/run.py                # every design, about 1.5 h (M3 Pro)
    python evaluation/cpu_performance/run.py --smoke        # one batch and length, briefly
    python evaluation/cpu_performance/run.py --designs hopper blackwell.mx -j 1

The designs compile first, as the `common/fpy2_models` package (its
`setup.py`, about a minute).  Both implementations run on `-j` threads
(default: PyTorch's, its performance cores).

Writes into the results directory (`evaluation/results/cpu_performance/`, or
`-o DIR`) `validation.json`, `measurements.csv` (each block's time),
`summary.csv` (each configuration's median), `provenance.json` and
`report.py`'s outputs.  Exits 1 if any configuration disagrees with MMA-Sim.
"""

import argparse
import importlib.metadata
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from itertools import product
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
PACKAGE = _ROOT / 'evaluation' / 'common' / 'fpy2_models'
sys.path[:0] = [str(PACKAGE), str(_ROOT / 'evaluation' / 'common' / 'mma'),
                str(_ROOT / 'examples' / 'mmasim')]
sys.path.append(str(_ROOT / 'evaluation'))  # last: `config`, not shadowing this `run`

import config
import torch
from core import outputs
from torch.utils import benchmark

RESULTS = _ROOT / 'evaluation' / 'results' / 'cpu_performance'


def build() -> dict[str, Any]:
    """Every design compiled and loaded as `torch.ops.fpy2_models.<op>`, by
    rebuilding the `fpy2_models` package, so that no stale build is timed;
    `generate_models.py`'s manifest of them."""
    subprocess.run([sys.executable, 'setup.py', '--quiet', 'build_ext', '--inplace'],
                   cwd=PACKAGE, check=True)
    import fpy2_models  # registers the operators
    return json.loads((PACKAGE / 'build' / 'generated_models' / 'manifest.json').read_text())


def time_blocks(fn: Callable[[], Any], threads: int, min_run_time: float) -> list[float]:
    """Seconds per call of *fn*, one per block `blocked_autorange` times: calls
    per block enough to make the timer's overhead negligible, blocks until
    *min_run_time* has passed."""
    m = benchmark.Timer(stmt='fn()', globals={'fn': fn}, num_threads=threads).blocked_autorange(
        min_run_time=min_run_time)
    return [t / m.number_per_run for t in m.raw_times]


KEY = ('design', 'implementation', 'batch_size', 'dot_length', 'threads')


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Each configuration's median time per call, and its spread, over every
    block of every replicate."""
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in KEY)].append(r['seconds_per_call'])
    out = []
    for key, times in sorted(groups.items()):
        cfg = dict(zip(KEY, key))
        median = statistics.median(times)
        q1, _, q3 = (statistics.quantiles(times, n=4, method='inclusive') if len(times) > 1
                     else (median,) * 3)
        out.append(cfg | {'samples': len(times), 'median_seconds_per_call': median,
                          'iqr_seconds_per_call': q3 - q1, 'relative_iqr': (q3 - q1) / median,
                          'ns_per_dot': median * 1e9 / cfg['batch_size'],
                          'dots_per_second': cfg['batch_size'] / median})
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('-o', '--out', type=Path, default=RESULTS, help='results directory (default: %(default)s)')
    ap.add_argument('-j', '--threads', type=int, help="threads for both implementations (default: PyTorch's)")
    ap.add_argument('--designs', nargs='+', default=[], metavar='NAME',
                    help='only the designs whose names contain one of these')
    ap.add_argument('--batches', type=int, nargs='+', default=config.CPU_PERF_BATCHES)
    ap.add_argument('--lengths', type=int, nargs='+', default=config.CPU_PERF_LENGTHS)
    ap.add_argument('--compile', action='store_true', default=config.CPU_PERF_COMPILE,
                    help="also time MMA-Sim under default torch.compile")
    ap.add_argument('--smoke', action='store_true', help='one small batch and length, briefly')
    args = ap.parse_args(argv)
    if args.smoke:
        args.batches, args.lengths = [min(args.batches)], [min(args.lengths)]
    replicates = 1 if args.smoke else config.CPU_PERF_REPLICATES
    min_run_time = 0.05 if args.smoke else config.CPU_PERF_MIN_RUN_TIME
    if args.threads is not None:
        # PyTorch, which runs FPy's models, and MMA-Sim's OpenMP FMA helper
        os.environ['OMP_NUM_THREADS'] = str(args.threads)
        torch.set_num_threads(args.threads)
    threads = torch.get_num_threads()

    out = args.out.resolve()
    manifest = build()
    import cases as cs  # MMA-Sim, compiling its FMA helper, after the designs' build
    import report
    cases = cs.select(cs.cases(manifest), args.designs)

    started = time.monotonic()
    validations = [{'design': n, 'status': 'unsupported', 'reason': r}
                   for n, r in manifest['unsupported'].items()]
    rows = []
    rng = random.Random(config.CPU_PERF_SEED)
    for (i, c), length, batch in product(enumerate(cases), args.lengths, args.batches):
        label = f'{c.name} batch={batch} length={length}'
        seed = config.CPU_PERF_SEED + i * 1_000_003 + batch
        torch._dynamo.reset()   # compiled from scratch each time, as in a fresh process
        p = cs.prepare(c, batch, seed, length, args.compile, config.CPU_PERF_DISTRIBUTION)
        v = {'design': c.name, 'batch_size': batch, 'dot_length': length, 'seed': seed,
             **cs.validate(p)}
        impls = {'fpy': p.fpy, 'mmasim': p.mmasim}
        if args.compile:
            v['mmasim_compiled'] = cs.validate_compiled(p)
            if v['mmasim_compiled']['valid']:
                impls['mmasim-compiled'] = p.mmasim_compiled
            else:
                print(f'COMPILE {label}: {v["mmasim_compiled"]["reason"]}', flush=True)
        validations.append(v)
        if not v['valid']:
            print(f'INVALID {label}: {v["reason"]}', flush=True)
            continue
        # warmed up, then each timed once per replicate, in a shuffled order so
        # that none is always first
        for fn in impls.values():
            fn()
        order = list(impls.items())
        for rep in range(replicates):
            rng.shuffle(order)
            for impl, fn in order:
                rows += [{'design': c.name, 'implementation': impl, 'batch_size': batch,
                          'dot_length': length, 'threads': threads, 'replicate': rep,
                          'seconds_per_call': t} for t in time_blocks(fn, threads, min_run_time)]
        print(f'TIMED   {label}', flush=True)

    invalid = sum(v.get('valid') is False for v in validations)
    outputs.write_json(out / 'validation.json', validations)
    outputs.write_csv(out / 'measurements.csv', rows)
    cxx = os.environ.get('CXX', 'c++')
    outputs.write_json(out / 'provenance.json', {
        'argv': sys.argv, **outputs.commit(), 'machine': platform.platform(),
        'processor': platform.processor() or platform.machine(), 'cpus': os.cpu_count(),
        'python': platform.python_version(), 'torch': torch.__version__,
        'torchao': importlib.metadata.version('torchao'), 'mmasim': importlib.metadata.version('mmasim'),
        'cxx': subprocess.run([cxx, '--version'], capture_output=True, text=True).stdout.splitlines()[0],
        'threads': threads, 'replicates': replicates, 'min_run_time': min_run_time,
        'distribution': cs.DISTRIBUTIONS[config.CPU_PERF_DISTRIBUTION], 'seed': config.CPU_PERF_SEED,
        'designs': args.designs, 'batches': args.batches, 'lengths': args.lengths,
        'compile': args.compile, 'smoke': args.smoke,
        'elapsed_seconds': time.monotonic() - started, 'invalid_configurations': invalid})
    outputs.write_csv(out / 'summary.csv', summarize(rows))   # last: the run is complete
    report.main([str(out)])
    return 1 if invalid else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
