"""
The CPU performance evaluation's figure, table and summary, from `run.py`'s
`summary.csv`; `run.py` calls it last.  Speedup is FPy's over eager MMA-Sim,
at one batch size and dot length (the largest of each by default): it barely
varies with the length.

    python evaluation/cpu_performance/report.py evaluation/results/cpu_performance

Writes into the results directory:

- `perf-cpu-speedup.{pdf,png}`: each design's speedup, ascending, colored by
  its hardware family;
- `perf-cpu-speedup.tex`: each family's geometric mean, least and greatest
  speedup, and all designs';
- `perf-cpu-macros.tex`: the paper's `EvalCPUPerf...` macros, for its
  `eval.tex`: all designs' speedups, and the configuration they are at;

and prints the per-design table.
"""

import argparse
import csv
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PREFIX = 'perf-cpu-'
"""This report's outputs' prefix."""

BASELINES = {'mmasim': 'eager', 'mmasim-compiled': 'compiled'}
"""The MMA-Sim implementations FPy is compared with, and their columns' names."""

FAMILIES = {
    'Volta / Turing': '#2a78d6', 'Ampere': '#eb6834', 'Ada': '#1baf7a', 'Hopper': '#eda100',
    'Blackwell': '#e87ba4', 'AMD CDNA3': '#b5b3ab', 'AMD CDNA2': '#7d7b75', 'FMA chain': '#3d3c3a',
}
"""Hardware families, in legend order, with their colors: each NVIDIA
generation its own hue, oldest first (Volta and Turing sharing one, as do
datacenter and RTX Blackwell); AMD's and the FMA chains, not compared across
generations, grays."""

UNSTABLE = 0.10
"""A relative IQR above which a measurement is flagged."""


def family(design: str) -> str:
    """*design*'s hardware family, of `FAMILIES`."""
    tokens = design.split('.')
    if tokens[-1] == 'fma':
        return 'FMA chain'
    arch = tokens[1]
    if arch in ('cdna2', 'cdna3'):
        return f'AMD {arch.upper()}'
    return {'volta': 'Volta / Turing', 'turing': 'Volta / Turing',
            'rtx_blackwell': 'Blackwell'}.get(arch, arch.capitalize())


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


def speedups(rows: list[dict[str, Any]]) -> dict[str, dict[tuple[str, int, int], float]]:
    """Per baseline, FPy's speedup over it, by (design, batch size, length)."""
    rate = {(r['implementation'], r['design'], r['batch_size'], r['dot_length']): r['dots_per_second']
            for r in rows}
    out: dict[str, dict] = defaultdict(dict)
    for (impl, *cfg), fpy in rate.items():
        if impl == 'fpy':
            for b in BASELINES:
                if (b, *cfg) in rate:
                    out[b][tuple(cfg)] = fpy / rate[b, *cfg]
    return out


def geomean(xs: list[float]) -> float:
    return math.exp(sum(map(math.log, xs)) / len(xs))


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


def macros(values: dict[str, float], batch: int, length: int, threads: int) -> str:
    """The paper's macros for *values*' speedups, as `eval.tex` writes them,
    rounded as `latex` rounds them."""
    def cmd(name: str, value: str) -> str:
        return f'\\newcommand{{\\EvalCPUPerf{name}}}{{{value}\\xspace}}'

    xs = list(values.values())
    return '\n'.join([
        '%% CPU performance (fpy evaluation/cpu_performance/report.py)',
        f'% FPy over eager MMA-Sim: a batch of {batch} dot products of length {length},',
        f'% on {threads} threads; least, greatest and geometric mean over {len(xs)} designs.',
        *(cmd(f'Speedup{k}', f'{x:.2f}$\\times$')
          for k, x in (('Min', min(xs)), ('Max', max(xs)), ('Mean', geomean(xs)))),
        cmd('Designs', str(len(xs))),
        cmd('Batch', str(batch)),
        cmd('Length', str(length)),
        cmd('Threads', str(threads)),
        ''])


def figure(values: dict[str, float], out: Path) -> None:
    """`perf-cpu-speedup`: *values*' designs ascending, unnamed; the legend
    names their families."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    order = sorted(values, key=values.get)
    heights = [values[d] for d in order]
    fig, ax = plt.subplots(figsize=(7.0, 2.3))
    ax.bar(range(len(order)), heights, 0.9, color=[FAMILIES[family(d)] for d in order])
    mean = geomean(heights)
    ax.axhline(mean, color='black', linewidth=1.0, linestyle=':')
    ax.annotate(f'geomean {mean:.2f}×', (0, mean), xycoords=ax.get_yaxis_transform(),
                xytext=(3, 2), textcoords='offset points', fontsize=7)
    ax.axhline(1.0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_xticks([])
    ax.set_xlabel('Designs', fontsize=8)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}×'))
    ax.tick_params(axis='y', labelsize=7)
    ax.set_ylabel('Speedup', fontsize=8)
    ax.grid(True, axis='y', alpha=0.25)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    shown = [f for f in FAMILIES if f in {family(d) for d in order}]
    handles = [plt.Rectangle((0, 0), 1, 1, color=FAMILIES[f]) for f in shown]
    # a grid of two rows, filled column by column
    ax.legend(handles, shown, loc='upper left', ncol=math.ceil(len(shown) / 2), columnspacing=1.0,
              frameon=False, fontsize=7, handlelength=1.4, handleheight=1.0)
    fig.tight_layout(pad=0.2)
    for ext in ('pdf', 'png'):
        fig.savefig(out / f'{PREFIX}speedup.{ext}', dpi=300)
    plt.close(fig)


def summary(rows: list[dict[str, Any]], batch: int, length: int) -> str:
    """The headline configuration's dot products per second and speedups, per
    design; `!` marks a measurement whose relative IQR exceeds `UNSTABLE`."""
    at = {(r['design'], r['implementation']): r for r in rows
          if r['batch_size'] == batch and r['dot_length'] == length}
    bases = [b for b in BASELINES if any(i == b for _, i in at)]
    designs = list(dict.fromkeys(d for d, _ in at))
    lines = [f'batch={batch} length={length} threads={rows[0]["threads"]}: '
             'FPy speedup over each MMA-Sim',
             f'{"design":32} {"FPy dots/s":>11}' + ''.join(f' {BASELINES[b] + " dots/s":>16}' for b in bases)
             + ''.join(f' {"vs " + BASELINES[b]:>12}' for b in bases)]
    for d in designs:
        if not (fpy := at.get((d, 'fpy'))):
            lines.append(f'{d:32} invalid or untimed')
            continue
        others = [at.get((d, b)) for b in bases]
        flag = max(r['relative_iqr'] for r in (fpy, *others) if r) > UNSTABLE
        lines.append(f'{d:32} {fpy["dots_per_second"]:11.3g}'
                     + ''.join(f' {r["dots_per_second"]:16.3g}' if r else f' {"-":>16}' for r in others)
                     + ''.join(f' {fpy["dots_per_second"] / r["dots_per_second"]:11.2f}x' if r
                               else f' {"-":>12}' for r in others)
                     + (' !' if flag else ''))
    return '\n'.join(lines) + '\n'


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('results', type=Path, help="run.py's results directory")
    ap.add_argument('--batch', type=int, help='batch size reported (default: the largest)')
    ap.add_argument('--length', type=int, help='dot length reported (default: the longest)')
    args = ap.parse_args(argv)
    os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[1] / '.matplotlib'))

    rows = load(args.results)
    if not rows:
        sys.exit(f'{args.results / "summary.csv"} has no measurements')
    batch = args.batch or max(r['batch_size'] for r in rows)
    length = args.length or max(r['dot_length'] for r in rows)
    ups = speedups(rows)
    eager = {d: x for (d, b, n), x in ups['mmasim'].items() if (b, n) == (batch, length)}
    if not eager:
        sys.exit(f'no design was timed at batch={batch} length={length}')
    figure(eager, args.results)
    (args.results / f'{PREFIX}speedup.tex').write_text(latex(table(eager)))
    threads = next(r['threads'] for r in rows if (r['batch_size'], r['dot_length']) == (batch, length))
    (args.results / f'{PREFIX}macros.tex').write_text(macros(eager, batch, length, threads))
    print(summary(rows, batch, length), end='')
    print(f'geomean over {len(eager)} designs: {geomean(list(eager.values())):.2f}x')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
