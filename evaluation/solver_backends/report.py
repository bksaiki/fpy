"""
The solver-backend evaluation's table and figures, from `run.py`'s results;
`run.py` calls it last.

    python evaluation/solver_backends/report.py evaluation/solver_backends/results

Writes into the results directory:

- `backends-totals.tex`: one row per backend, over the designs every backend
  compiled: compile and solver time, the solver's share, backend calls,
  and calls and solver time per query;
- `backends-scatter.{pdf,png}`: each design's solver time against its number of
  constraints, one point per design and backend, leaving out designs no
  backend spends :data:`MIN_SOLVER_SECONDS` on;
- `backends-scatter-all.{pdf,png}`: the same with every design.

Both are log-log; `backends-scatter-linear.{pdf,png}` and
`backends-scatter-all-linear.{pdf,png}` are the same on linear axes.

The summary, with the designs whose C++ differs between backends, is printed.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

NAMES = {'z3': 'z3', 'z3-opt': 'z3 (Optimize)', 'cvc5': 'cvc5', 'highs': 'HiGHS'}
"""Each backend's label."""

COLORS = {'z3': '#1f1f1e', 'z3-opt': '#8a5cc7', 'cvc5': '#2a78d6', 'highs': '#d6772a'}
MARKERS = {'z3': 'o', 'z3-opt': 's', 'cvc5': '^', 'highs': 'D'}
INK, MUTED, GRID = '#1f1f1e', '#6b6a64', '#e4e3dc'

MIN_SOLVER_SECONDS = 0.01
"""A design every backend answers in less solver time than this is left out
of the figure: its time is the timer's noise, and its point (the FMA chains,
at a few hundred constraints and a millisecond) only stretches both axes."""


def load(out: Path) -> list[dict[str, Any]]:
    """Every design record in *out*, in the order the run measured them."""
    run = json.loads((out / 'backends-run.json').read_text())
    by_name = {}
    for p in out.glob('*.json'):
        if p.name.startswith('backends-'):
            continue
        rec = json.loads(p.read_text())
        by_name[rec['design']] = rec
    return [by_name[n] for n in run['designs'] if n in by_name]


def _ok(rec: dict[str, Any], b: str) -> dict[str, Any] | None:
    r = rec['backends'].get(b)
    return r if r is not None and r['outcome'] == 'ok' else None


def backends_of(records: list[dict[str, Any]]) -> list[str]:
    return list(records[0]['backends']) if records else []


def plotted(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The designs the figure shows: those some backend spends at least
    :data:`MIN_SOLVER_SECONDS` in the solver on."""
    return [rec for rec in records
            if any((r := _ok(rec, b)) is not None and r['solver_time'] >= MIN_SOLVER_SECONDS
                   for b in rec['backends'])]


def figure(records: list[dict[str, Any]], path: Path, log: bool = True) -> None:
    """Each of *records*' solver time against its number of constraints, a
    point per backend, on log axes where *log* and linear ones otherwise, as
    *path* with `.pdf` and with `.png`."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for b in backends_of(records):
        pts = [(r['stats']['constraints'], r['solver_time'])
               for rec in records if (r := _ok(rec, b)) is not None and r['solver_time'] > 0]
        if pts:
            xs, ys = zip(*pts)
            ax.scatter(xs, ys, s=22, color=COLORS.get(b, INK), marker=MARKERS.get(b, 'o'),
                       label=NAMES.get(b, b), alpha=0.8, linewidths=0)
    if log:
        ax.set_xscale('log')
        ax.set_yscale('log')
    else:
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
    ax.set_xlabel('Constraints', color=INK)
    ax.set_ylabel('Solver time (s)', color=INK)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    if ax.collections:
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path.with_suffix('.pdf'))
    fig.savefig(path.with_suffix('.png'), dpi=200)
    plt.close(fig)


def numbers(records: list[dict[str, Any]]) -> dict[str, Any]:
    """The totals the table and summary cite."""
    bs = backends_of(records)
    ref = bs[0]
    # totals only over designs every backend compiled, so they compare
    common = [rec for rec in records if all(_ok(rec, b) for b in bs)]
    totals = {}
    for b in bs:
        rs = [_ok(rec, b) for rec in common]
        compile_s = sum(r['seconds'] for r in rs)
        solver_s = sum(r['solver_time'] for r in rs)
        totals[b] = {
            'compile_seconds': compile_s,
            'solver_seconds': solver_s,
            'solver_share': solver_s / compile_s if compile_s else 0.0,
            'queries': sum(r['stats']['queries'] for r in rs),
            'calls': sum(r['stats']['calls'] for r in rs),
            'degraded': sum(r['stats']['degraded'] for r in rs),
            'constraints': sum(r['stats']['constraints'] for r in rs),
        }
    for b in bs[1:]:
        s = totals[b]['solver_seconds']
        totals[b]['speedup_vs_ref'] = totals[ref]['solver_seconds'] / s if s else None
    differs = {
        b: [rec['design'] for rec in common
            if _ok(rec, b)['cpp_sha256'] != _ok(rec, ref)['cpp_sha256']]
        for b in bs[1:]
    }
    failed = {
        b: {rec['design']: rec['backends'][b]['outcome'] for rec in records if not _ok(rec, b)}
        for b in bs
    }
    return {
        'designs': len(records), 'compared': len(common), 'reference': ref,
        'totals': totals, 'cpp_differs': differs, 'failed': failed,
    }


def _count(n: int) -> str:
    """*n* rounded to thousands past ten thousand: `227332` -> `227k`."""
    return f'{n / 1000:.0f}k' if n >= 10_000 else str(n)


def totals_latex(nums: dict[str, Any]) -> str:
    """One row per backend, totalled over the designs every backend compiled.

    No constraint or query counts: they are the same in every row while the
    backends agree, since the store poses the same problem, and asks the
    same questions of it, whichever answers.  The summary prints them once.
    """
    lines = [
        '\\begin{tabular}{lrrrrrr}',
        '\\toprule',
        ('Backend & Compile & Solver & Solver share & '
         'Backend calls & Calls/query & Time/query \\\\'),
        '\\midrule',
    ]
    for b, t in nums['totals'].items():
        q = t['queries']
        per_call = t['calls'] / q if q else 0.0
        per_query = 1000 * t['solver_seconds'] / q if q else 0.0
        lines.append(
            f'{NAMES.get(b, b)} & {t["compile_seconds"]:.0f}\\,s & {t["solver_seconds"]:.0f}\\,s & '
            f'{100 * t["solver_share"]:.0f}\\% & '
            f'{_count(t["calls"])} & {per_call:.2f} & {per_query:.1f}\\,ms \\\\')
    lines += ['\\bottomrule', '\\end{tabular}', '']
    return '\n'.join(lines)


def summary(nums: dict[str, Any]) -> str:
    ref = nums['reference']
    lines = [(f'{nums["designs"]} designs; {nums["compared"]} compiled by every backend, '
              f'compared below (reference {NAMES.get(ref, ref)})')]
    t = nums['totals'][ref]
    lines.append(f'{t["constraints"]} constraints, {t["queries"]} queries '
                 '(the same under every backend that agrees)')
    for b, t in nums['totals'].items():
        line = (f'{NAMES.get(b, b):>13}: compile {t["compile_seconds"]:8.1f}s  '
                f'solver {t["solver_seconds"]:8.1f}s ({100 * t["solver_share"]:.0f}% of compile)  '
                f'{t["queries"]} queries, {t["calls"]} backend calls, {t["degraded"]} degraded')
        if t.get('speedup_vs_ref'):
            line += f'  solver x{t["speedup_vs_ref"]:.2f} vs {NAMES.get(ref, ref)}'
        lines.append(line)
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
    (out / 'backends-totals.tex').write_text(totals_latex(nums))
    for log, suffix in ((True, ''), (False, '-linear')):
        figure(plotted(records), out / f'backends-scatter{suffix}', log)
        figure(records, out / f'backends-scatter-all{suffix}', log)
    print(summary(nums))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
