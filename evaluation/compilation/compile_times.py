"""
Time how long FPy takes to generate each MMA-Sim design's model, with each
backend: its C++ (as `compile.compile_design` generates it, but not the entry's
signature, which `fpy2_models` needs for its wrapper and which redoes the
specialization and analyses) and its Triton kernel
(`compile_triton.compile_matmul`). Generation only: neither the C++ compiler
nor Triton's own compilation runs.

The table, `<results>/compilation/compile-times.csv` (`plots.py`), has a row
per design and backend: whether FPy compiled it, why not, and the seconds it
took. Its provenance has the run's wall time.

    python evaluation/compilation/compile_times.py -j 16
"""

import argparse
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# `common/mma`, for `compile`, `compile_triton` and `core`; `examples/mmasim`, for `models`
sys.path[:0] = [str(ROOT / 'evaluation' / 'common' / 'mma'), str(ROOT / 'examples' / 'mmasim')]

from compile import DESIGNS, prepare
from compile_triton import compile_matmul
from core import outputs

import fpy2 as fp

RESULTS = ROOT / 'evaluation' / 'results'
BACKENDS = ('cpp', 'triton')
# compiled once in each process before timing, so no design pays for the
# first import or cache fill
WARM = 'fp32 (fma)'


def generate_cpp(build) -> str:
    """*build*'s C++, as `compile.compile_design` generates it."""
    compiler = fp.CppCompiler(unfold=fp.CppCompiler.UnfoldMode.ROUNDINGS, enable_fenv=False)
    return compiler.compile_module(prepare(build))


def generate(job: tuple[str, str]) -> dict:
    """One design's model from one backend, and the seconds it took; or why
    FPy refused it."""
    name, backend = job
    build = dict(DESIGNS)[name]
    t0 = time.perf_counter()
    try:
        generate_cpp(build) if backend == 'cpp' else compile_matmul(build, None)
        status, reason = 'ok', ''
    except Exception as e:  # noqa: BLE001 -- a refusal is a result
        status, reason = 'refused', f'{type(e).__name__}: {e}'
    return {'design': name, 'backend': backend, 'status': status, 'reason': reason,
            'seconds': time.perf_counter() - t0}


def _warm() -> None:
    for backend in BACKENDS:
        generate((WARM, backend))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=str(__doc__).splitlines()[1])
    ap.add_argument('--results', type=Path, default=RESULTS,
                    help="the outputs' root (default: %(default)s)")
    ap.add_argument('-j', '--jobs', type=int, default=1,
                    help='processes generating models (default: %(default)s)')
    args = ap.parse_args(argv)

    jobs = [(name, backend) for backend in BACKENDS for name, _ in DESIGNS]
    t0 = time.perf_counter()
    if args.jobs <= 1:
        _warm()
        rows = [generate(j) for j in jobs]
    else:
        with ProcessPoolExecutor(args.jobs, initializer=_warm) as pool:
            rows = list(pool.map(generate, jobs))
    wall = time.perf_counter() - t0
    for r in rows:
        print(f'{r["backend"]:6} {r["design"]:40} {r["status"]:7} {r["seconds"]:6.1f}s', flush=True)

    table = args.results / 'compilation' / 'compile-times.csv'
    outputs.write_csv(table, rows)
    outputs.write_json(table.with_suffix('.provenance.json'), {
        'argv': sys.argv, **outputs.commit(), 'machine': platform.platform(), 'cpus': os.cpu_count(),
        'settings': {k: v if isinstance(v, int) else str(v) for k, v in vars(args).items()},
        'seconds': {'total': wall}})
    print(f'{len(DESIGNS)} designs, {sum(r["status"] == "refused" for r in rows)} refused: {table}')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
