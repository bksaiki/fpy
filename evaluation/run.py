"""
Runs the whole evaluation.

Each evaluation runs into `<results>/<name>/`, then its plots and tables are
drawn.  Defaults are `config.py`'s.

An evaluation whose final output exists is finished and not rerun; its plots
are redrawn.  After changing `config.py`, rerun what it affects with
`--no-cache`.

    python evaluation/run.py evaluation/results                  # every evaluation
    python evaluation/run.py DIR --disable gsm8k --disable llm   # all but these
    python evaluation/run.py DIR --plot        # only redraw, from existing results
    python evaluation/run.py DIR --no-cache    # discard results and compile caches first
    python evaluation/run.py DIR --threads 8   # processes or threads per evaluation
"""

import argparse
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

RunCommands = Callable[[Path, list[str]], list[list[str]]]
"""Its run's commands, given also `-j N` (`--threads`) or nothing; an
evaluation without such an option ignores it."""


@dataclass(frozen=True)
class Eval:
    run: RunCommands
    plot: Commands
    done: str
    """Its final output, under its results directory: present when finished."""


EVALS = {
    'analysis': Eval(
        run=lambda d, j: [[PY, str(HERE / 'analysis' / 'run.py'), '-o', str(d), *j]],
        plot=lambda d: [[PY, str(HERE / 'analysis' / 'report.py'), str(d)]],
        done='precision-paper.json'),
    # `explore.py` and `plots.py` add `blas/` to the root they are given
    'blas': Eval(
        run=lambda d, j: [[PY, str(HERE / 'dse' / 'blas' / 'explore.py'), '--results', str(d.parent), *j]],
        plot=lambda d: [[PY, str(HERE / 'dse' / 'blas' / 'plots.py'), '--results', str(d.parent)]],
        done='blas-grid.csv'),
    'gsm8k': Eval(
        run=lambda d, j: [[PY, str(HERE / 'gsm8k' / 'run.py'), '-o', str(d)]],
        plot=lambda d: [[PY, str(HERE / 'gsm8k' / 'report.py'), str(d)]],
        done='gsm8k-speed.json'),
    'llm': Eval(
        # the manifest compiles one design at a time, so each compile time is its own
        run=lambda d, j: [[PY, str(HERE / 'dse' / 'llm' / 'sweep.py'), '--results', str(d), '--manifest', '-j', '1'],
                          [PY, str(HERE / 'dse' / 'llm' / 'sweep.py'), '--results', str(d), *j]],
        plot=lambda d: [[PY, str(HERE / 'dse' / 'llm' / 'plot_grid.py'), '--results', str(d),
                         '--table', f'sweep-grid-{TAG}.csv']],
        done=f'sweep/sweep-grid-{TAG}.provenance.json'),
}
"""Every evaluation, in the order run: CPU first."""


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
    ap.add_argument('--threads', type=int, metavar='N',
                    help="processes or threads per evaluation (default: each evaluation's own)")
    ap.add_argument('--plot', action='store_true', help='only redraw plots and tables')
    ap.add_argument('--no-cache', action='store_true',
                    help="discard each evaluation's results and compile caches, and rerun it")
    args = ap.parse_args(argv)
    if args.plot and args.no_cache:
        ap.error('--plot redraws from results that --no-cache would discard')

    failed = []
    for name in [n for n in EVALS if n not in args.disable]:
        ev, out = EVALS[name], args.results.resolve() / name
        print(f'== {name}: {out}', flush=True)
        if args.no_cache:
            shutil.rmtree(out, ignore_errors=True)
        if not args.plot and not (out / ev.done).exists():
            if not _run(ev.run(out, [] if args.threads is None else ['-j', str(args.threads)])):
                failed.append(name)
                continue
        if (out / ev.done).exists():
            if not _run(ev.plot(out)):
                failed.append(name)
        elif args.plot:
            print(f'{name}: no results to plot', flush=True)
    if failed:
        print(f'failed: {", ".join(failed)}', flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
