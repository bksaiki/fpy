"""
Times FPy's C++ for every MMA-Sim design against MMA-Sim itself, on the CPU:
batches of dot products, each chained one K-wide call at a time from Python,
by `torch.ops.fpy2_models.<op>` and by MMA-Sim's `dpa` tuned (`no_sync`'s
arithmetic under `torch.compile`).  Each operator is first checked to give the
same bits on a repeat call, on one thread or many, and batched or one row at a
time; every configuration, both FPy and tuned MMA-Sim, to agree with MMA-Sim
as released bit for bit.

    python evaluation/cpu_performance/run.py                # every design, about 1.5 h (M3 Pro)
    python evaluation/cpu_performance/run.py --smoke        # one batch and length, briefly
    python evaluation/cpu_performance/run.py --designs hopper blackwell.mx -j 1

The designs compile first, as the `common/fpy2_models` package (its
`setup.py`, about a minute).  Both implementations run on `-j` threads
(default: PyTorch's, its performance cores).

Writes into the results directory (`evaluation/results/cpu_performance/`, or
`-o DIR`) `validation.json`, `measurements.csv` (each block's time),
`summary.csv` (each configuration's median), `provenance.json` and
`report.py`'s outputs.  A partial run -- `--smoke`, `--designs`, or batches or
lengths other than `config.py`'s -- writes into
`evaluation/results/cpu_performance-partial/` instead, so as not to replace a
full run's.  `summary.csv` comes last, only once every configuration agrees
with MMA-Sim: `evaluation/run.py` takes it to mean the run is finished.
Exits 1, without it, if an operator fails its checks or any configuration
disagrees.
"""

import argparse
import importlib.metadata
import json
import os
import platform
import random
import shlex
import statistics
import subprocess
import sys
import time
import zlib
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
PARTIAL = RESULTS.with_name('cpu_performance-partial')


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


def compiler_version(cxx: str) -> str:
    """The first line *cxx* `--version` prints, or why there is none: never
    raises, as it runs after the timing."""
    try:
        done = subprocess.run([*shlex.split(cxx), '--version'], capture_output=True, text=True,
                              timeout=60, check=False)
    except (OSError, ValueError, subprocess.SubprocessError) as e:
        return f'{cxx}: {type(e).__name__}: {e}'
    lines = (done.stdout or done.stderr).splitlines()
    return lines[0] if lines else f'{cxx}: no version printed'


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
    ap.add_argument('-o', '--out', type=Path,
                    help=f'results directory (default: {RESULTS}, or {PARTIAL} for a partial run)')
    ap.add_argument('-j', '--threads', type=int, help="threads for both implementations (default: PyTorch's)")
    ap.add_argument('--designs', nargs='+', default=[], metavar='NAME',
                    help='only the designs whose names contain one of these')
    ap.add_argument('--batches', type=int, nargs='+', default=config.CPU_PERF_BATCHES)
    ap.add_argument('--lengths', type=int, nargs='+', default=config.CPU_PERF_LENGTHS)
    ap.add_argument('--smoke', action='store_true', help='one small batch and length, briefly')
    args = ap.parse_args(argv)
    if args.smoke:
        args.batches, args.lengths = [min(args.batches)], [min(args.lengths)]
    replicates = 1 if args.smoke else config.CPU_PERF_REPLICATES
    min_run_time = 0.05 if args.smoke else config.CPU_PERF_MIN_RUN_TIME
    if args.threads is not None:
        torch.set_num_threads(args.threads)   # PyTorch's, which runs FPy's models
    threads = torch.get_num_threads()
    # as many for MMA-Sim's OpenMP FMA helper.  On macOS it shares PyTorch's
    # OpenMP runtime, and so its threads; on Linux, GCC links it to its own
    # libgomp, which reads this when the helper loads (`import cases`, below)
    # and would otherwise take every core
    os.environ['OMP_NUM_THREADS'] = str(threads)

    partial = (args.smoke or args.designs or tuple(args.batches) != config.CPU_PERF_BATCHES
               or tuple(args.lengths) != config.CPU_PERF_LENGTHS)
    out = (args.out or (PARTIAL if partial else RESULTS)).resolve()
    # an earlier run's, which would mark this one finished should it fail
    (out / 'summary.csv').unlink(missing_ok=True)
    manifest = build()
    import cases as cs  # MMA-Sim, compiling its FMA helper, after the designs' build
    import report
    cases = cs.select(cs.cases(manifest), args.designs)
    if broken := {c.name: why for c in cases if (why := cs.check(c)) is not None}:
        for name, why in broken.items():
            print(f'BROKEN  {name}: {why}', flush=True)
        return 1

    started = time.monotonic()
    validations = [{'design': n, 'status': 'unsupported', 'reason': r}
                   for n, r in manifest['unsupported'].items()]
    rows = []
    rng = random.Random(config.CPU_PERF_SEED)
    for c, length, batch in product(cases, args.lengths, args.batches):
        label = f'{c.name} batch={batch} length={length}'
        # by the design's name, not its place among those run, so that a
        # `--designs` run draws a full run's inputs
        seed = config.CPU_PERF_SEED + zlib.crc32(c.name.encode()) * 1_000_003 + batch
        torch._dynamo.reset()   # compiled from scratch each time, as in a fresh process
        p = cs.prepare(c, batch, seed, length, config.CPU_PERF_DISTRIBUTION)
        want = p.mmasim()   # once: MMA-Sim as released is the slowest call of all
        checks = {'fpy': cs.validate(p, want), 'mmasim-tuned': cs.validate_tuned(p, want)}
        v = {'design': c.name, 'batch_size': batch, 'dot_length': length, 'seed': seed,
             'valid': all(r['valid'] for r in checks.values()), **checks}
        validations.append(v)
        if not v['valid']:
            why = '; '.join(f'{k}: {r["reason"]}' for k, r in checks.items() if not r['valid'])
            print(f'INVALID {label}: {why}', flush=True)
            continue
        # warmed up by validation, which ran each once (compiling MMA-Sim);
        # each timed once per replicate, in a shuffled order so that none is
        # always first
        order = [('fpy', p.fpy), ('mmasim-tuned', p.tuned)]
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
    outputs.write_json(out / 'provenance.json', {
        'argv': sys.argv, **outputs.commit(), 'machine': platform.platform(),
        'processor': platform.processor() or platform.machine(), 'cpus': os.cpu_count(),
        'python': platform.python_version(), 'torch': torch.__version__,
        'torchao': importlib.metadata.version('torchao'), 'mmasim': importlib.metadata.version('mmasim'),
        'cxx': compiler_version(os.environ.get('CXX', 'c++')),
        'mmasim_cxx': compiler_version(cs.MMASIM_CXX),
        'threads': threads, 'replicates': replicates, 'min_run_time': min_run_time,
        'distribution': cs.DISTRIBUTIONS[config.CPU_PERF_DISTRIBUTION], 'seed': config.CPU_PERF_SEED,
        'designs': args.designs, 'batches': args.batches, 'lengths': args.lengths,
        'smoke': args.smoke,
        'elapsed_seconds': time.monotonic() - started, 'invalid_configurations': invalid})
    if invalid:
        print(f'{invalid} configurations disagree with MMA-Sim: no summary.csv', flush=True)
        return 1
    outputs.write_csv(out / 'summary.csv', summarize(rows))   # last: the run is complete
    report.main([str(out)])
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
