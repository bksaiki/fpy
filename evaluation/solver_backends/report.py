"""
The solver-backend evaluation's table and figures, from `run.py`'s results;
`run.py` calls it last.

    python evaluation/solver_backends/report.py evaluation/solver_backends/results

Writes into `report/` in the results directory, overwriting only these:

- `backends-totals.tex`: one row per backend, over the designs every backend
  compiled: how it finds a maximum (binary search or optimize), compile
  and solver time, the solver's share, and solver time per query;
- `sensitivity-scatter.{pdf,png}`: each query's time against the
  constraints in the component it was asked of, from every design's median
  run, a panel per backend, log-log.

The summary, with the designs whose C++ differs between backends, and those
whose counts differ between one backend's repeats, is printed.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

NAMES = {'z3': 'z3', 'z3-opt': 'z3 (Optimize)', 'cvc5': 'cvc5', 'highs': 'HiGHS'}
"""Each backend's label."""

METHODS = {'z3': 'Binary search', 'z3-opt': 'Optimize', 'cvc5': 'Binary search',
           'highs': 'Optimize'}
"""How each backend finds a maximum: bisecting over satisfiability queries,
or asking the solver to optimize."""

ROW_NAMES = {**NAMES, 'z3-opt': 'z3'}
"""Each backend's label in the table, whose method column tells the two z3
rows apart."""

COLORS = {'z3': '#1f1f1e', 'z3-opt': '#8a5cc7', 'cvc5': '#2a78d6', 'highs': '#d6772a'}
MARKERS = {'z3': 'o', 'z3-opt': 's', 'cvc5': '^', 'highs': 'D'}
INK, MUTED, GRID = '#1f1f1e', '#6b6a64', '#e4e3dc'

REPORT_DIR = 'report'
"""Where in the results directory the report goes, apart from the run's data."""

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


def counts_vary(r: dict[str, Any]) -> bool:
    """Whether *r*'s repeats stated different constraints or asked different
    queries: the analysis took another path in some of them, after a backend
    call hit its limit, so they did not time the same work."""
    return len({(x['stats']['constraints'], x['stats']['queries'])
                for x in r.get('runs', []) if 'cpp_sha256' in x}) > 1


def backends_of(records: list[dict[str, Any]]) -> list[str]:
    return list(records[0]['backends']) if records else []


def queries(records: list[dict[str, Any]], b: str, log: bool) -> tuple[list[int], list[float]]:
    """Every query backend *b* answered in *records*' median runs: the
    constraints it saw, and its time.  On log axes, only those with both
    above zero: a constant objective sees no constraint."""
    xs: list[int] = []
    ys: list[float] = []
    for rec in records:
        if (r := _ok(rec, b)) is None or 'trace' not in r:
            continue
        for c, s in zip(r['trace']['constraints'], r['trace']['seconds']):
            if not log or (c > 0 and s > 0):
                xs.append(c)
                ys.append(s)
    return xs, ys


def figure(records: list[dict[str, Any]], path: Path, log: bool = True) -> None:
    """Each query's time against the constraints it saw, a panel per
    backend on shared axes, log where *log* and linear otherwise, as *path*
    with `.pdf` and with `.png`.

    A panel each, not one cloud: tens of thousands of points a backend
    overlap until only the last drawn shows."""
    bs = backends_of(records)
    cols = 2 if len(bs) > 1 else 1
    rows = -(-len(bs) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 2.8 * rows),
                             sharex=True, sharey=True, squeeze=False)
    for ax in axes.flat[len(bs):]:
        ax.set_visible(False)
    shown = 0
    for ax, b in zip(axes.flat, bs):
        xs, secs = queries(records, b, log)
        shown += len(xs)
        ys = [1000 * s for s in secs]
        # rasterized: a vector PDF of this many points is slow to open
        ax.scatter(xs, ys, s=3, color=COLORS.get(b, INK), alpha=0.25,
                   linewidths=0, rasterized=True)
        ax.set_title(NAMES.get(b, b), color=INK, fontsize=10)
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
    ax0 = axes.flat[0]
    if log and shown:   # a log axis over no point is an error
        ax0.set_xscale('log')
        ax0.set_yscale('log')
        # 10, 100, 1,000 rather than powers of ten
        plain = FuncFormatter(lambda v, _: f'{v:,.10g}')
        ax0.xaxis.set_major_formatter(plain)
        ax0.yaxis.set_major_formatter(plain)
    elif not log:
        ax0.set_xlim(left=0)
        ax0.set_ylim(bottom=0)
    fig.supxlabel('Constraints in the queried component', color=INK, fontsize=10)
    fig.supylabel('Query time (ms)', color=INK, fontsize=10)
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
    varies = {
        b: [rec['design'] for rec in records if (r := _ok(rec, b)) and counts_vary(r)]
        for b in bs
    }
    return {
        'designs': len(records), 'compared': len(common), 'reference': ref,
        'totals': totals, 'cpp_differs': differs, 'failed': failed, 'counts_vary': varies,
    }


def totals_latex(nums: dict[str, Any]) -> str:
    """One row per backend, totalled over the designs every backend compiled.

    No constraint or query counts: they are the same in every row while the
    backends agree, since the store poses the same problem, and asks the
    same questions of it, whichever answers.  The summary prints them once.
    """
    lines = [
        '\\begin{tabular}{llrrrr}',
        '\\toprule',
        'Backend & Solver method & Compile & Solver & Solver share & Time/query \\\\',
        '\\midrule',
    ]
    for b, t in nums['totals'].items():
        q = t['queries']
        per_query = 1000 * t['solver_seconds'] / q if q else 0.0
        lines.append(
            f'{ROW_NAMES.get(b, b)} & {METHODS.get(b, "")} & {t["compile_seconds"]:.0f}\\,s & {t["solver_seconds"]:.0f}\\,s & '
            f'{100 * t["solver_share"]:.0f}\\% & {per_query:.1f}\\,ms \\\\')
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
    if not any(nums['counts_vary'].values()):
        lines.append('counts the same across every backend\'s repeats')
    for b, ds in nums['counts_vary'].items():
        if ds:
            lines.append(f'counts vary across repeats under {NAMES.get(b, b)} '
                         f'(a call hit its limit): {", ".join(ds)}')
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
    dest = out / REPORT_DIR
    dest.mkdir(exist_ok=True)
    (dest / 'backends-totals.tex').write_text(totals_latex(nums))
    figure(records, dest / 'sensitivity-scatter')
    print(summary(nums))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
