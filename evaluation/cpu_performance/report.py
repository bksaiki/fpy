"""
The CPU performance evaluation's figure, table and summary, from `run.py`'s
`summary.csv`; `run.py` calls it last.  Speedup is FPy's over tuned MMA-Sim
(`no_sync`'s arithmetic under `torch.compile`), at one batch size and dot
length (the largest of each by default): it barely varies with the length.

    python evaluation/cpu_performance/report.py evaluation/results/cpu_performance

Writes into the results directory:

- `perf-cpu-speedup.{pdf,png}`: each design's speedup, ascending, colored by
  its hardware family;
- `plots/perf-cpu-speedup-b<batch>-n<length>.{pdf,png}`: the same, at every
  configuration measured;
- `perf-cpu-speedup.tex`: each family's geometric mean, least and greatest
  speedup, and all designs';
- `perf-cpu-tuning.tex`: the same for tuned MMA-Sim's speedup over MMA-Sim as
  released, where `run.py` timed it;
- `perf-cpu-macros.tex`: the paper's `EvalCPUPerf...` macros, for its
  `eval.tex`: all designs' speedups, tuning's median speedup, and the
  configuration they are at;

and prints the per-design table.
"""

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[1]))

from common.plot import FAMILIES, family, geomean, speedup_figure

PREFIX = 'perf-cpu-'
"""This report's outputs' prefix."""

BASELINE = 'mmasim-tuned'
"""The MMA-Sim implementation FPy is compared with."""

RELEASED = 'mmasim-released'
"""MMA-Sim as released, which tuning is compared with."""

UNSTABLE = 0.10
"""A relative IQR above which a measurement is flagged."""


def load(results: Path) -> list[dict[str, Any]]:
    """`summary.csv`'s rows, numbers parsed."""
    with (results / 'summary.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ('batch_size', 'dot_length', 'threads'):
            r[k] = int(r[k])
        for k in ('relative_iqr', 'dots_per_second', 'ns_per_dot'):
            r[k] = float(r[k])
    return rows


def speedups(rows: list[dict[str, Any]], impl: str = 'fpy',
             base: str = BASELINE) -> dict[tuple[str, int, int], float]:
    """*impl*'s speedup over *base*, by (design, batch size, length)."""
    rate = {(r['implementation'], r['design'], r['batch_size'], r['dot_length']): r['dots_per_second']
            for r in rows}
    return {tuple(cfg): x / rate[base, *cfg] for (i, *cfg), x in rate.items()
            if i == impl and (base, *cfg) in rate}


def headline(rows: list[dict[str, Any]], batch: int | None = None, length: int | None = None,
             impl: str = 'fpy', base: str = BASELINE) -> dict[str, float]:
    """Each design's speedup of *impl* over *base* at *batch* and *length*
    (the largest of each by default)."""
    batch = batch or max(r['batch_size'] for r in rows)
    length = length or max(r['dot_length'] for r in rows)
    return {d: x for (d, b, n), x in speedups(rows, impl, base).items() if (b, n) == (batch, length)}


def table(values: dict[str, float]) -> list[dict[str, Any]]:
    """Each family's designs, geometric mean, least and greatest speedup in
    *values* (by design), in `FAMILIES`' order, then all of them."""
    by: dict[str, list[float]] = defaultdict(list)
    for d, x in values.items():
        by[family(d)].append(x)
    groups = [(f, by[f]) for f in FAMILIES if by[f]] + [('All', list(values.values()))]
    return [{'family': f, 'designs': len(xs), 'geomean': geomean(xs), 'min': min(xs), 'max': max(xs)}
            for f, xs in groups if xs]


def latex(rows: list[dict[str, Any]]) -> str:
    lines = [r'\begin{tabular}{lrrrr}', r'\toprule',
             r'Family & Designs & Geomean & Min & Max \\', r'\midrule']
    for r in rows:
        if r['family'] == 'All':
            lines.append(r'\midrule')
        lines.append(f'{r["family"]} & {r["designs"]} & {r["geomean"]:.2f}$\\times$ & '
                     f'{r["min"]:.2f}$\\times$ & {r["max"]:.2f}$\\times$ \\\\')
    return '\n'.join([*lines, r'\bottomrule', r'\end{tabular}', ''])


def macros(values: dict[str, float], batch: int, length: int, threads: int,
           tuning: dict[str, float] | None = None) -> str:
    """The paper's macros for *values*' speedups, as `eval.tex` writes them,
    rounded as `latex` rounds them, and *tuning*'s median if given."""
    def cmd(name: str, value: str) -> str:
        return f'\\newcommand{{\\EvalCPUPerf{name}}}{{{value}\\xspace}}'

    xs = list(values.values())
    return '\n'.join([
        '%% CPU performance (fpy evaluation/cpu_performance/report.py)',
        f'% FPy over tuned MMA-Sim: a batch of {batch} dot products of length {length},',
        f'% on {threads} threads; least, greatest and geometric mean over {len(xs)} designs.',
        *(cmd(f'Speedup{k}', f'{x:.2f}$\\times$')
          for k, x in (('Min', min(xs)), ('Max', max(xs)), ('Mean', geomean(xs)))),
        *([cmd('OptRatio', f'{statistics.median(tuning.values()):.2f}$\\times$')] if tuning else []),
        cmd('Designs', str(len(xs))),
        cmd('Batch', str(batch)),
        cmd('Length', str(length)),
        cmd('Threads', str(threads)),
        ''])


def summary(rows: list[dict[str, Any]], batch: int, length: int) -> str:
    """The headline configuration's dot products per second and speedups, per
    design; `!` marks a measurement whose relative IQR exceeds `UNSTABLE`."""
    at = {(r['design'], r['implementation']): r for r in rows
          if r['batch_size'] == batch and r['dot_length'] == length}
    designs = list(dict.fromkeys(d for d, _ in at))
    lines = [f'batch={batch} length={length} threads={rows[0]["threads"]}: FPy speedup over tuned MMA-Sim',
             f'{"design":32} {"FPy dots/s":>11} {"MMA-Sim dots/s":>15} {"speedup":>9}']
    for d in designs:
        fpy, mm = at.get((d, 'fpy')), at.get((d, BASELINE))
        if not (fpy and mm):
            lines.append(f'{d:32} invalid or untimed')
            continue
        flag = max(fpy['relative_iqr'], mm['relative_iqr']) > UNSTABLE
        lines.append(f'{d:32} {fpy["dots_per_second"]:11.3g} {mm["dots_per_second"]:15.3g}'
                     f' {fpy["dots_per_second"] / mm["dots_per_second"]:8.2f}x' + (' !' if flag else ''))
    return '\n'.join(lines) + '\n'


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('results', type=Path, help="run.py's results directory")
    ap.add_argument('--batch', type=int, help='batch size reported (default: the largest)')
    ap.add_argument('--length', type=int, help='dot length reported (default: the longest)')
    args = ap.parse_args(argv)
    rows = load(args.results)
    if not rows:
        sys.exit(f'{args.results / "summary.csv"} has no measurements')
    batch = args.batch or max(r['batch_size'] for r in rows)
    length = args.length or max(r['dot_length'] for r in rows)
    ups = headline(rows, batch, length)
    if not ups:
        sys.exit(f'no design was timed at batch={batch} length={length}')
    speedup_figure([ups], args.results / f'{PREFIX}speedup')
    for b, n in sorted({(r['batch_size'], r['dot_length']) for r in rows}):
        speedup_figure([headline(rows, b, n)], args.results / 'plots' / f'{PREFIX}speedup-b{b}-n{n}')
    (args.results / f'{PREFIX}speedup.tex').write_text(latex(table(ups)))
    if tuning := headline(rows, batch, length, BASELINE, RELEASED):
        (args.results / f'{PREFIX}tuning.tex').write_text(latex(table(tuning)))
    threads = next(r['threads'] for r in rows if (r['batch_size'], r['dot_length']) == (batch, length))
    (args.results / f'{PREFIX}macros.tex').write_text(macros(ups, batch, length, threads, tuning))
    print(summary(rows, batch, length), end='')
    print(f'geomean over {len(ups)} designs: {geomean(list(ups.values())):.2f}x')
    if tuning:
        print(f'tuned over released MMA-Sim, {len(tuning)} designs: median '
              f'{statistics.median(tuning.values()):.2f}x, geomean {geomean(list(tuning.values())):.2f}x')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
