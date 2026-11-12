"""
The sensitivity evaluation's table and figure, from `run.py`'s results;
`run.py` calls it last.

    python evaluation/sensitivity/report.py evaluation/results/sensitivity

Writes `sensitivity-table.tex`, `sensitivity-macros.tex`,
`sensitivity-scatter.{pdf,png}`, `sensitivity-paper.json` and
`sensitivity-summary.txt` beside the data,
overwriting only those; the README says what each holds.
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

sys.path.append(str(Path(__file__).resolve().parents[1]))

from common.plot import ACCENT, LARGE, ROW, save, tidy

NAMES = {'z3': 'Z3 (Binary search)', 'z3-opt': 'Z3 (Optimize)', 'cvc5': 'cvc5'}
"""Each backend's label."""

METHODS = {'z3': 'Binary search', 'z3-opt': 'Optimize', 'cvc5': 'Binary search'}
"""How each backend finds a maximum."""

ROW_NAMES = {**NAMES, 'z3': 'Z3', 'z3-opt': 'Z3'}
"""Each backend's label in the table, whose method column tells the two z3
rows apart."""

MACROS = {'z3': 'zthreebin', 'z3-opt': 'zthreeopt', 'cvc5': 'cvc'}
"""Each backend's prefix in the paper's macros: letters only, as a LaTeX
command name must be."""



def load(out: Path) -> list[dict[str, Any]]:
    """Every design record in *out*, in the order the run measured them."""
    run = json.loads((out / 'sensitivity-run.json').read_text())
    by_name = {}
    for p in out.glob('*.json'):
        if p.name.startswith('sensitivity-'):
            continue
        rec = json.loads(p.read_text())
        by_name[rec['design']] = rec
    return [by_name[n] for n in run['designs'] if n in by_name]


def _ok(rec: dict[str, Any], b: str) -> dict[str, Any] | None:
    r = rec['backends'].get(b)
    return r if r is not None and r['outcome'] == 'ok' else None


def backends_of(records: list[dict[str, Any]]) -> list[str]:
    return list(records[0]['backends']) if records else []


def queries(records: list[dict[str, Any]], b: str) -> tuple[list[int], list[float]]:
    """Every query backend *b* answered in *records*' median runs: the
    constraints it saw, and its time.  Only those with both above zero,
    which log axes can show: a constant objective sees no constraint."""
    xs: list[int] = []
    ys: list[float] = []
    for rec in records:
        if (r := _ok(rec, b)) is None:
            continue
        for c, s in zip(r['trace']['constraints'], r['trace']['seconds']):
            if c > 0 and s > 0:
                xs.append(c)
                ys.append(s)
    return xs, ys


JITTER = 0.04
"""The scatter's greatest relative jitter of a constraint count."""


def figure(records: list[dict[str, Any]], path: Path) -> None:
    """Each query's time against the constraints it saw, a panel per
    backend in one row on shared log-log axes, as *path* with `.pdf` and
    with `.png`.  Constraint counts recur across designs, so each point is
    jittered by up to `JITTER` along x, else they stack into columns.

    A panel each, not one cloud: tens of thousands of points a backend
    overlap until only the last drawn shows."""
    bs = backends_of(records)
    cols = max(len(bs), 1)
    fig, axes = plt.subplots(1, cols, figsize=ROW, sharex=True, sharey=True, squeeze=False,
                             layout='constrained')
    shown = 0
    rng = random.Random(0)   # the same jitter each time
    for ax, b in zip(axes.flat, bs):
        xs, secs = queries(records, b)
        shown += len(xs)
        xs = [x * (1 + JITTER) ** rng.uniform(-1, 1) for x in xs]
        ys = [1000 * s for s in secs]
        # rasterized: a vector PDF of this many points is slow to open
        # small and opaque: overlaps do not darken into blotches
        ax.scatter(xs, ys, s=0.6, color=ACCENT, linewidths=0, rasterized=True)
        ax.set_title(NAMES.get(b, b), loc='left', **LARGE)
        tidy(ax, 'both')
    ax0 = axes.flat[0]
    if shown:   # a log axis over no point is an error
        ax0.set_xscale('log')
        ax0.set_yscale('log')
        # 10, 100, 1,000 rather than powers of ten
        plain = FuncFormatter(lambda v, _: f'{v:,.10g}')
        ax0.xaxis.set_major_formatter(plain)
        ax0.yaxis.set_major_formatter(plain)
    fig.supxlabel('Constraints in the queried component', **LARGE)
    fig.supylabel('Query time (ms)', **LARGE)
    save(fig, path, dpi=600)   # its points are rasterized, in the PDF too


def numbers(records: list[dict[str, Any]]) -> dict[str, Any]:
    """The totals the table, the summary and the paper cite."""
    bs = backends_of(records)
    ref = bs[0]
    # totals only over designs every backend compiled, so they compare
    common = [rec for rec in records if all(_ok(rec, b) for b in bs)]
    totals = {}
    for b in bs:
        rs = [_ok(rec, b) for rec in common]
        compile_s = sum(r['seconds'] for r in rs)
        solver_s = sum(r['solver_time'] for r in rs)
        queries = sum(r['stats']['queries'] for r in rs)
        totals[b] = {
            'method': METHODS.get(b, b),
            'compile_seconds': compile_s,
            'solver_seconds': solver_s,
            'solver_share': solver_s / compile_s if compile_s else 0.0,
            'queries': queries,
            'seconds_per_query': solver_s / queries if queries else 0.0,
            'degraded': sum(r['stats']['degraded'] for r in rs),
        }
    differs = {
        b: [rec['design'] for rec in common
            if _ok(rec, b)['cpp_sha256'] != _ok(rec, ref)['cpp_sha256']]
        for b in bs[1:]
    }
    failed = {
        b: {rec['design']: _failure(rec['backends'][b]) for rec in records if not _ok(rec, b)}
        for b in bs
    }
    return {
        'designs': len(records), 'compared': len(common), 'reference': ref,
        'totals': totals, 'cpp_differs': differs, 'failed': failed,
    }


def _failure(r: dict[str, Any]) -> str:
    """How a backend failed a design: its outcome, and how many of the
    repeats compiled where some did."""
    if r.get('compiled'):
        return f'{r["outcome"]}, {r["compiled"]} of {len(r["runs"])} compiled'
    return r['outcome']


def totals_latex(nums: dict[str, Any]) -> str:
    """One row per backend, totalled over the designs every backend compiled.

    No constraint or query counts: they are the same in every row while the
    backends agree, since the store poses the same problem, and asks the
    same questions of it, whichever answers.
    """
    lines = [
        '\\begin{tabular}{llrrrr}',
        '\\toprule',
        'Backend & Solver method & Compile (s) & Solver (s) & Solver share & Time/query (ms) \\\\',
        '\\midrule',
    ]
    for b, t in nums['totals'].items():
        per_query = 1000 * t['seconds_per_query']
        lines.append(
            f'{ROW_NAMES.get(b, b)} & {t["method"]} & {t["compile_seconds"]:.0f} & {t["solver_seconds"]:.0f} & '
            f'{100 * t["solver_share"]:.0f}\\% & {per_query:.1f} \\\\')
    lines += ['\\bottomrule', '\\end{tabular}', '']
    return '\n'.join(lines)


def macros(nums: dict[str, Any]) -> str:
    """The numbers the paper's text cites, as macros: each backend's solver
    time and its share of compile time, and the range of the others' solver
    time over the reference's (`\\solverslowdown`)."""
    totals = nums['totals']
    lines = []
    for b, t in totals.items():
        if (name := MACROS.get(b)) is not None:
            lines.append(f'\\newcommand{{\\{name}time}}{{{t["solver_seconds"]:.0f}\\,s}}')
            lines.append(f'\\newcommand{{\\{name}share}}{{{100 * t["solver_share"]:.0f}\\%}}')
    ref = totals[nums['reference']]['solver_seconds']
    ratios = sorted(t['solver_seconds'] / ref for b, t in totals.items()
                    if b != nums['reference'] and ref)
    if ratios:
        span = f'{ratios[0]:.1f}' if len(ratios) == 1 else f'{ratios[0]:.1f}--{ratios[-1]:.1f}'
        lines.append(f'\\newcommand{{\\solverslowdown}}{{{span}$\\times$}}')
    return '\n'.join(lines) + '\n'


def summary(nums: dict[str, Any]) -> str:
    ref = nums['reference']
    lines = [(f'{nums["designs"]} designs; {nums["compared"]} compiled by every backend, '
              f'compared below (reference {NAMES.get(ref, ref)})')]
    for b, t in nums['totals'].items():
        lines.append(
            f'{NAMES.get(b, b):>18}: compile {t["compile_seconds"]:8.1f}s  '
            f'solver {t["solver_seconds"]:8.1f}s ({100 * t["solver_share"]:.0f}% of compile)  '
            f'{t["queries"]} queries, {t["degraded"]} degraded')
    for b, ds in nums['cpp_differs'].items():
        lines.append(f'C++ differs from {NAMES.get(ref, ref)} under {NAMES.get(b, b)}: '
                     + (', '.join(ds) if ds else 'none'))
    for b, fs in nums['failed'].items():
        if fs:
            lines.append(f'not compiled under {NAMES.get(b, b)}: '
                         + ', '.join(f'{d} ({o})' for d, o in fs.items()))
    return '\n'.join(lines)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('results', type=Path, help="run.py's output directory")
    args = ap.parse_args(argv)
    out = args.results

    records = load(out)
    if not records:
        ap.error(f'no results in {out}')
    nums = numbers(records)
    (out / 'sensitivity-table.tex').write_text(totals_latex(nums))
    (out / 'sensitivity-macros.tex').write_text(macros(nums))
    figure(records, out / 'sensitivity-scatter')
    (out / 'sensitivity-paper.json').write_text(json.dumps(nums, indent=1) + '\n')
    text = summary(nums)
    (out / 'sensitivity-summary.txt').write_text(text + '\n')
    print(text)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
