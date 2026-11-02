"""
The precision evaluation's table and figure, from `run.py`'s results.

    python evaluation/analysis_precision/report.py results/
    python evaluation/analysis_precision/report.py results/ results/adversarial/ -o combined/

Writes `precision-tops.tex` (top sites per architecture, without and with digit
bounds), `precision-coverage.tex` (how much of each bound digit bounds leave was
observed, per architecture) and `precision-slack.{pdf,png}` (each bounded site's
slack with digit bounds, per dimension) to the first results directory, or `-o
DIR`, and prints summary numbers.  Several directories -- other seeds or input sources -- are
combined: a site's observations are their union, so its slack in each
dimension is the least any run saw.
"""

import argparse
import json
import math
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from observe import SLACK
from run import Analysis

ARCHS = {
    'volta': 'Volta', 'turing': 'Turing', 'ampere': 'Ampere', 'ada': 'Ada',
    'hopper': 'Hopper', 'blackwell': 'Blackwell', 'rtx_blackwell': 'RTX Blackwell',
    'cdna1': 'CDNA1', 'cdna2': 'CDNA2', 'cdna3': 'CDNA3', 'fma': 'FMA chain',
}
"""Each architecture's tag in a design's name, and its row label, in row order."""

DIMENSIONS = {'digits': 'Significant digits', 'lsb': 'Lowest digit', 'mag': 'Magnitude'}
"""Each slack reported, and its panel title.  The highest digit, `msb`, is
left out for the magnitude: `log2 b - log2 max|v|`, which sees a bound `b`
that is not a power of two."""

_SLACK = {**SLACK, 'mag': lambda static, seen: static - seen}

BINS = 8
"""Slacks of this many bits or more share the last bin."""

BLUE, INK, MUTED, GRID = '#2a78d6', '#1f1f1e', '#6b6a64', '#e4e3dc'
LARGE, SMALL = {'fontsize': 14, 'color': INK}, {'fontsize': 12, 'color': MUTED}


@dataclass
class Row:
    label: str
    designs: int = 0
    sites: int = 0
    top_format: int = 0
    top_digit_bound: int = 0


def arch(design: str) -> str:
    """The architecture tag of *design*: `nv.hopper.bf16.f32` -> `hopper`."""
    return 'fma' if '(fma)' in design else design.split('.')[1]


def _result(site: dict[str, Any], a: Analysis) -> dict[str, Any] | None:
    return site.get(a.value)


def load(results: list[Path]) -> tuple[list[dict[str, Any]], list[str]]:
    """The records in *results*, each design's combined across directories,
    and the designs that failed in any."""
    by_design: dict[str, list[dict[str, Any]]] = {}
    for d in results:
        for p in sorted(d.glob('*.json')):
            rec = json.loads(p.read_text())
            by_design.setdefault(rec['design'], []).append(rec)
    failed = [n for n, rs in by_design.items() if any('error' in r for r in rs)]
    return [combine(rs) for n, rs in by_design.items() if n not in failed], failed


def _site_key(site: dict[str, Any]) -> tuple:
    return tuple(site['calls']), site['loc'], site.get('rhs'), site['name']


_JOIN = {'digits': max, 'lsb': min, 'msb': max, 'mag': max}
"""How each observed parameter combines over runs."""


def _union(a: dict[str, Any] | None, b: dict[str, Any] | None) -> dict[str, Any] | None:
    """One analysis's result at a site over two runs' observations."""
    if a is None or b is None:
        return a or b
    if a['static'] != b['static']:
        raise ValueError('runs disagree on a static bound')
    seen = a['observed'] or b['observed']
    if a['observed'] and b['observed']:
        seen = {k: f(a['observed'][k], b['observed'][k]) for k, f in _JOIN.items()}
    return {**a, 'executed': a['executed'] or b['executed'],
            'sound': a['sound'] and b['sound'], 'observed': seen}


def combine(records: list[dict[str, Any]]) -> dict[str, Any]:
    """One design's runs as one record over all their observations."""
    if len({r.get('anf', False) for r in records}) > 1:
        raise ValueError(f'{records[0]["design"]}: runs with and without --anf')
    sites: dict[tuple, dict[str, Any]] = {}
    for rec in records:
        for site in rec['sites']:
            k = _site_key(site)
            if k not in sites:
                sites[k] = site
                continue
            sites[k] = {**site, **{a.value: _union(sites[k].get(a.value), site.get(a.value))
                                   for a in Analysis}}
    return {**records[0], 'sites': list(sites.values()), 'n': sum(r['n'] for r in records)}


def table(records: Iterable[dict[str, Any]]) -> list[Row]:
    """One row per architecture present, in `ARCHS` order, then the total."""
    rows = {tag: Row(label) for tag, label in ARCHS.items()}
    total = Row('Total')
    for rec in records:
        for row in (rows[arch(rec['design'])], total):
            row.designs += 1
            for site in rec['sites']:
                f, d = _result(site, Analysis.FORMAT), _result(site, Analysis.DIGIT_BOUND)
                row.sites += 1
                row.top_format += bool(f and f['top'])
                row.top_digit_bound += bool(d and d['top'])
    return [r for r in rows.values() if r.designs] + [total]


def latex(rows: list[Row]) -> str:
    lines = [
        r'\begin{tabular}{lrrrr}', r'\toprule',
        r'Architecture & Designs & Sites & $\mathbb{R}^*$ (FI) & $\mathbb{R}^*$ (FI+DB) \\',
        r'\midrule',
    ]
    for r in rows:
        if r.label == 'Total':
            lines.append(r'\midrule')
        lines.append(f'{r.label} & {r.designs} & {r.sites} & {r.top_format} & '
                     f'{r.top_digit_bound} \\\\')
    return '\n'.join([*lines, r'\bottomrule', r'\end{tabular}', ''])


def _measured(rec: dict[str, Any]) -> Iterable[tuple[dict[str, Any], dict[str, Any]]]:
    """The static and observed parameters of each of *rec*'s sites that digit
    bounds leave bounded and that ran soundly with a finite nonzero value."""
    for site in rec['sites']:
        r = _result(site, Analysis.DIGIT_BOUND)
        if r and not r['top'] and r['sound'] and r['static'] and r['observed']:
            yield r['static'], r['observed']


def slacks(records: Iterable[dict[str, Any]]) -> dict[str, list[int | float]]:
    """Each dimension's slack at every measured site."""
    out: dict[str, list[int | float]] = {d: [] for d in DIMENSIONS}
    for rec in records:
        for st, seen in _measured(rec):
            for d in DIMENSIONS:
                out[d].append(_SLACK[d](st[d], seen[d]))
    return out


COVERAGE = {
    'digits': ('Precision', lambda st, seen: seen['digits'] / st['digits']),
    'lsb': ('Quantum', lambda st, seen: 2.0 ** (st['lsb'] - seen['lsb'])),
    'mag': ('Magnitude', lambda st, seen: 2.0 ** (seen['mag'] - st['mag'])),
}
"""Each coverage -- how much of what the static bound allows was observed, in
(0, 1], or 0 where the bound is unbounded -- by its column title, one per
parameter of a format A(p, exp, b): significant digits p' / p, smallest
quantum 2^exp / 2^lsb', and largest magnitude max|v| / b."""


def coverage(records: Iterable[dict[str, Any]]) -> list[tuple[str, int, dict[str, float]]]:
    """Each architecture's mean coverages over its measured sites, in `ARCHS`
    order, then the total: `(label, sites, {coverage: mean})`."""
    seen: dict[str, list[dict[str, float]]] = {tag: [] for tag in ARCHS}
    for rec in records:
        seen[arch(rec['design'])] += [{c: f(st, ob) for c, (_, f) in COVERAGE.items()}
                                     for st, ob in _measured(rec)]
    rows = [(ARCHS[tag], cs) for tag, cs in seen.items() if cs]
    rows.append(('Total', [c for cs in seen.values() for c in cs]))
    return [(label, len(cs), {c: sum(x[c] for x in cs) / len(cs) for c in COVERAGE})
            for label, cs in rows]


def coverage_latex(rows: list[tuple[str, int, dict[str, float]]]) -> str:
    titles = ' & '.join(title for title, _ in COVERAGE.values())
    lines = [r'\begin{tabular}{lr' + 'r' * len(COVERAGE) + '}', r'\toprule',
             f'Architecture & Sites & {titles} \\\\', r'\midrule']
    for label, n, means in rows:
        if label == 'Total':
            lines.append(r'\midrule')
        cells = ' & '.join(f'{100 * means[c]:.0f}\\%' for c in COVERAGE)
        lines.append(f'{label} & {n} & {cells} \\\\')
    return '\n'.join([*lines, r'\bottomrule', r'\end{tabular}', ''])


def _bin(v: float) -> int:
    """Bin 0 holds exact bounds, bin k slacks in (k - 1, k] bits."""
    return BINS if math.isinf(v) else min(math.ceil(v), BINS)


def figure(data: dict[str, list[int | float]], out: Path) -> None:
    fig, axes = plt.subplots(1, len(DIMENSIONS), figsize=(4 * len(DIMENSIONS), 3.4),
                             sharey=True, layout='constrained')
    labels = [str(b) for b in range(BINS)] + [f'≥{BINS}']
    for ax, (d, title) in zip(axes, DIMENSIONS.items(), strict=True):
        counts = [0] * (BINS + 1)
        for v in data[d]:
            counts[_bin(v)] += 1
        ax.bar(range(BINS + 1), counts, width=0.8, color=BLUE, zorder=2)
        ax.set_title(title, loc='left', **LARGE)
        ax.set_xticks(range(BINS + 1), labels)
        ax.set_xlabel('slack (bits)', **LARGE)
        ax.tick_params(colors=SMALL['color'], labelsize=SMALL['fontsize'])
        ax.grid(axis='y', color=GRID, linewidth=0.8, zorder=0)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel('sites', **LARGE)
    for ext in ('.pdf', '.png'):
        fig.savefig(out.with_suffix(ext), dpi=200)
    plt.close(fig)


def summary(records: list[dict[str, Any]], data: dict[str, list[int | float]]) -> str:
    unsound = sum(not r['sound'] for rec in records for site in rec['sites']
                  for a in Analysis if (r := _result(site, a)))
    n = len(data['digits'])
    tight = sum(all(data[d][i] == 0 for d in DIMENSIONS) for i in range(n))
    inputs = sum(r['n'] for r in records)
    lines = [f'{len(records)} designs, {inputs} inputs, {unsound} unsound results',
             f'{n} bounded sites (digit bounds); tight in all dimensions: {tight} ({tight / max(n, 1):.0%})']
    for d, title in DIMENSIONS.items():
        zero = sum(v == 0 for v in data[d])
        one = sum(v <= 1 for v in data[d])
        lines.append(f'  {title}: slack 0 at {zero / max(n, 1):.0%}, ≤1 at {one / max(n, 1):.0%}')
    return '\n'.join(lines)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('results', type=Path, nargs='+', help="run.py's output directories")
    ap.add_argument('-o', '--out', type=Path, help='where to write (default the first RESULTS)')
    args = ap.parse_args(argv)
    out = args.out or args.results[0]
    out.mkdir(parents=True, exist_ok=True)

    records, failed = load(args.results)
    if not records:
        ap.error(f'no results in {args.results}')
    (out / 'precision-tops.tex').write_text(latex(table(records)))
    (out / 'precision-coverage.tex').write_text(coverage_latex(coverage(records)))
    data = slacks(records)
    figure(data, out / 'precision-slack')
    print(summary(records, data))
    if failed:
        print(f'failed, left out: {", ".join(failed)}')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
