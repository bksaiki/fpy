"""
Runs the whole evaluation.

Each evaluation runs into `<results>/<name>/`, then its plots and tables are
drawn.  Defaults are `config.py`'s.

An evaluation whose final output exists is finished and not rerun; its plots
are redrawn.  After changing `config.py`, rerun what it affects with
`--no-cache`.  Before anything runs, checks that the machine has what each
evaluation to run needs (README.md, Setup).

    python evaluation/run.py evaluation/results                  # every evaluation
    python evaluation/run.py DIR --disable gsm8k --disable llm   # all but these
    python evaluation/run.py DIR --plot        # only redraw, from existing results
    python evaluation/run.py DIR --no-cache    # discard results and compile caches first
    python evaluation/run.py DIR --threads 16  # processes or threads per evaluation (default 1)
    python evaluation/run.py DIR --smoke       # each evaluation at a token size, in minutes
"""

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent

sys.path[:0] = [str(HERE / 'common' / 'mma'), str(HERE.parent / 'examples' / 'mmasim')]
import config

PY = sys.executable
TAG = config.MODEL.rsplit('/', 1)[-1]

Commands = Callable[[Path], list[list[str]]]
"""An evaluation's commands, given its results directory."""

RunCommands = Callable[[Path, list[str], list[str]], list[list[str]]]
"""Its run's commands, given also `-j N` (`--threads`), and `--smoke` or
nothing."""


@dataclass(frozen=True)
class Eval:
    run: RunCommands
    plot: Commands
    done: str
    """Its final output, under its results directory: present when finished."""
    needs: tuple[str, ...]
    """Modules its run imports, beyond `fpy2`."""
    tools: tuple[str, ...] = ()
    """Programs its run calls."""
    gpu: bool = False


EVALS = {
    'analysis': Eval(
        run=lambda d, j, s: [[PY, str(HERE / 'analysis' / 'run.py'), '-o', str(d), *j, *s]],
        plot=lambda d: [[PY, str(HERE / 'analysis' / 'report.py'), str(d)]],
        done='precision-paper.json', needs=('matplotlib',)),
    'sensitivity': Eval(
        # one compilation at a time, so each time is its own
        run=lambda d, j, s: [[PY, str(HERE / 'sensitivity' / 'run.py'), '-o', str(d), '-j', '1', *s]],
        plot=lambda d: [[PY, str(HERE / 'sensitivity' / 'report.py'), str(d)]],
        done='sensitivity-paper.json', needs=('matplotlib',)),
    # `compile_times.py` and `plots.py` add `compilation/` to the root they are given;
    # one design at a time, so each time is its own
    'compilation': Eval(
        run=lambda d, j, s: [[PY, str(HERE / 'compilation' / 'compile_times.py'), '--results', str(d.parent),
                              '-j', '1', *s]],
        plot=lambda d: [[PY, str(HERE / 'compilation' / 'plots.py'), '--results', str(d.parent)]],
        done='compile-times.csv', needs=('matplotlib',)),
    'ablation': Eval(
        run=lambda d, j, s: [[PY, str(HERE / 'ablation' / 'run.py'), '-o', str(d), *j]],   # no smoke size
        plot=lambda d: [[PY, str(HERE / 'ablation' / 'report.py'), str(d)]],
        done='ablation-paper.json', needs=()),
    # `explore.py` and `plots.py` add `blas/` to the root they are given
    'blas': Eval(
        run=lambda d, j, s: [[PY, str(HERE / 'dse' / 'blas' / 'explore.py'), '--results', str(d.parent),
                              *j, *s]],
        plot=lambda d: [[PY, str(HERE / 'dse' / 'blas' / 'plots.py'), '--results', str(d.parent)]],
        done='blas-grid.csv', needs=('torch', 'matplotlib'),
        tools=(os.environ.get('CXX', 'c++'),)),  # as `torch.utils.cpp_extension` picks it
    'cpu_performance': Eval(
        run=lambda d, j, s: [[PY, str(HERE / 'cpu_performance' / 'run.py'), '-o', str(d), *j, *s]],
        plot=lambda d: [[PY, str(HERE / 'cpu_performance' / 'report.py'), str(d)]],
        done='summary.csv', needs=('torch', 'torchao', 'mmasim', 'matplotlib'),
        tools=(os.environ.get('CXX', 'c++'),)),
    'gsm8k': Eval(
        run=lambda d, j, s: [[PY, str(HERE / 'gsm8k' / 'run.py'), '-o', str(d), *j, *s]],
        plot=lambda d: [[PY, str(HERE / 'gsm8k' / 'report.py'), str(d)],
                        [PY, str(HERE / 'gsm8k' / 'plot.py'), str(d)]],
        done='gsm8k-speed.json', needs=('torch', 'triton', 'transformers', 'datasets', 'mmasim', 'matplotlib'),
        gpu=True),
    'llm': Eval(
        # the manifest compiles one design at a time, so each compile time is its own
        run=lambda d, j, s: [[PY, str(HERE / 'dse' / 'llm' / 'sweep.py'), '--results', str(d), '--manifest',
                              '-j', '1', *s],
                             [PY, str(HERE / 'dse' / 'llm' / 'sweep.py'), '--results', str(d), *j, *s]],
        plot=lambda d: [[PY, str(HERE / 'dse' / 'llm' / 'plot_grid.py'), '--results', str(d),
                         '--table', f'sweep-grid-{TAG}.csv']],
        done=f'sweep/sweep-grid-{TAG}.provenance.json',
        needs=('torch', 'triton', 'transformers', 'datasets', 'matplotlib'), gpu=True),
}
"""Every evaluation, in the order run: CPU first."""


def _missing(evals: list[Eval]) -> list[str]:
    """What *evals* need that this machine lacks."""
    out = [f'the Python module `{m}`' for m in sorted({m for e in evals for m in e.needs})
           if importlib.util.find_spec(m) is None]
    out += [f'`{t}` on PATH' for t in sorted({t for e in evals for t in e.tools}) if shutil.which(t) is None]
    if any(e.gpu for e in evals) and importlib.util.find_spec('torch') is not None:
        import torch

        if not torch.cuda.is_available():
            out.append('a CUDA GPU')
    return out


def _run(cmds: list[list[str]]) -> bool:
    """Runs *cmds* in order, stopping at the first that fails; whether all succeeded."""
    for cmd in cmds:
        print('$', ' '.join(cmd), flush=True)
        if subprocess.run(cmd, check=False).returncode:
            return False
    return True


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('results', type=Path, help="the results' root")
    ap.add_argument('--disable', action='append', choices=list(EVALS), default=[], metavar='EVAL',
                    help=f'skip an evaluation; repeatable (of {", ".join(EVALS)})')
    ap.add_argument('--threads', type=int, default=1, metavar='N',
                    help='processes or threads per evaluation (default %(default)s)')
    ap.add_argument('--plot', action='store_true', help='only redraw plots and tables')
    ap.add_argument('--no-cache', action='store_true',
                    help="discard each evaluation's results and compile caches, and rerun it")
    ap.add_argument('--smoke', action='store_true',
                    help='run each evaluation at a token size (into a results root of its own)')
    args = ap.parse_args(argv)
    if args.plot and args.no_cache:
        ap.error('--plot redraws from results that --no-cache would discard')

    names = [n for n in EVALS if n not in args.disable]
    todo = [EVALS[n] for n in names if not args.plot
            and (args.no_cache or not (args.results / n / EVALS[n].done).exists())]
    if missing := _missing(todo):
        sys.exit('missing: ' + '; '.join(missing) + f'\nsee {HERE / "README.md"}, Setup, or --disable '
                 'the evaluations that need them')

    failed = []
    for name in names:
        ev, out = EVALS[name], args.results.resolve() / name
        print(f'== {name}: {out}', flush=True)
        if args.no_cache:
            shutil.rmtree(out, ignore_errors=True)
        if not args.plot and not (out / ev.done).exists():
            if not _run(ev.run(out, ['-j', str(args.threads)],
                               ['--smoke'] if args.smoke else [])):
                failed.append(name)
        if (out / ev.done).exists():   # plotted even if it failed (e.g. a bit check)
            if not _run(ev.plot(out)) and name not in failed:
                failed.append(name)
        elif args.plot:
            print(f'{name}: no results to plot', flush=True)
    # the CPU and GPU speedups side by side, once both are measured
    perf = ('cpu_performance', 'gsm8k')
    if all((args.results / n / EVALS[n].done).exists() for n in perf):
        print(f'== performance: {args.results.resolve() / "performance"}', flush=True)
        if not _run([[PY, str(HERE / 'performance' / 'plots.py'), '--results', str(args.results.resolve())]]):
            failed.append('performance')
    if failed:
        print(f'failed: {", ".join(failed)}', flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
