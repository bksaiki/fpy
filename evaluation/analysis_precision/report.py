"""
The precision evaluation's table and figure, from `run.py`'s results.

    python evaluation/analysis_precision/report.py results/

Writes `precision-tops.tex` (top sites per architecture, without and with digit
bounds) and `precision-slack.{pdf,png}` (each bounded site's slack with digit
bounds, per dimension) to the results directory, or `-o DIR`, and prints
summary numbers.
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
from run import Analysis

ARCHS = {
    'volta': 'Volta', 'turing': 'Turing', 'ampere': 'Ampere', 'ada': 'Ada',
    'hopper': 'Hopper', 'blackwell': 'Blackwell', 'rtx_blackwell': 'RTX Blackwell',
    'cdna1': 'CDNA1', 'cdna2': 'CDNA2', 'cdna3': 'CDNA3', 'fma': 'FMA chain',
}
"""Each architecture's tag in a design's name, and its row label, in row order."""

DIMENSIONS = {'digits': 'Significant digits', 'lsb': 'Lowest digit', 'msb': 'Highest digit'}
"""Each slack in a site's result, and its panel title."""

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
    unexecuted: int = 0


def arch(design: str) -> str:
    """The architecture tag of *design*: `nv.hopper.bf16.f32` -> `hopper`."""
    return 'fma' if '(fma)' in design else design.split('.')[1]


def _result(site: dict[str, Any], a: Analysis) -> dict[str, Any] | None:
    return site.get(a.value)


def load(results: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """The records in *results*, and the designs that failed."""
    records = [json.loads(p.read_text()) for p in sorted(results.glob('*.json'))]
    return [r for r in records if 'error' not in r], [r['design'] for r in records if 'error' in r]


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
                row.unexecuted += not (d or f)['executed']
    return [r for r in rows.values() if r.designs] + [total]


def latex(rows: list[Row]) -> str:
    lines = [
        r'\begin{tabular}{lrrrrr}', r'\toprule',
        r'Architecture & Designs & Sites & $\mathbb{R}^*$ (FI) & $\mathbb{R}^*$ (FI+DB) & Not run \\',
        r'\midrule',
    ]
    for r in rows:
        if r.label == 'Total':
            lines.append(r'\midrule')
        lines.append(f'{r.label} & {r.designs} & {r.sites} & {r.top_format} & '
                     f'{r.top_digit_bound} & {r.unexecuted} \\\\')
    return '\n'.join([*lines, r'\bottomrule', r'\end{tabular}', ''])


def slacks(records: Iterable[dict[str, Any]]) -> dict[str, list[int | float]]:
    """Each dimension's slack at every executed, sound site digit bounds
    leave bounded."""
    out: dict[str, list[int | float]] = {d: [] for d in DIMENSIONS}
    for rec in records:
        for site in rec['sites']:
            r = _result(site, Analysis.DIGIT_BOUND)
            if r and not r['top'] and r['executed'] and r['sound'] and r['digits'] is not None:
                for d in DIMENSIONS:
                    out[d].append(r[d])
    return out


def _bin(v: float) -> int:
    return BINS if math.isinf(v) else min(int(v), BINS)


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
    lines = [f'{len(records)} designs, {unsound} unsound results',
             f'{n} bounded sites (digit bounds); tight in all dimensions: {tight} ({tight / max(n, 1):.0%})']
    for d, title in DIMENSIONS.items():
        zero = sum(v == 0 for v in data[d])
        one = sum(v <= 1 for v in data[d])
        lines.append(f'  {title}: slack 0 at {zero / max(n, 1):.0%}, ≤1 at {one / max(n, 1):.0%}')
    return '\n'.join(lines)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('results', type=Path, help="run.py's output directory")
    ap.add_argument('-o', '--out', type=Path, help='where to write (default RESULTS)')
    args = ap.parse_args(argv)
    out = args.out or args.results
    out.mkdir(parents=True, exist_ok=True)

    records, failed = load(args.results)
    if not records:
        ap.error(f'no results in {args.results}')
    (out / 'precision-tops.tex').write_text(latex(table(records)))
    data = slacks(records)
    figure(data, out / 'precision-slack')
    print(summary(records, data))
    if failed:
        print(f'failed, left out: {", ".join(failed)}')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
