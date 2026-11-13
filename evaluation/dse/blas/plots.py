"""
The BLAS evaluation's figure and numbers, from `explore.py`'s table, drawn as
`plot_grid.py` draws the grid's KL.

    python evaluation/dse/blas/plots.py    # -> evaluation/results/blas/plots/k.{png,pdf}; numbers on stdout
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import ticker

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / 'evaluation' / 'results'
sys.path.insert(0, str(ROOT / 'evaluation' / 'dse' / 'llm'))
sys.path.append(str(ROOT / 'evaluation'))

# `plot_grid.py`'s look: each generation's colour and marker, each join's
# line, and its sizes at print
from common.plot import save
from plot_grid import (
    BASES,
    EDGE,
    HEIGHT,
    INK,
    JOINS,
    KEY_GAP,
    LARGE,
    LINE,
    MARKER,
    RING,
    RING_LINE,
    SCALE,
    _look,
    key,
    legend,
    on_ticks,
    panel,
)

# each panel's input formats
PANELS = {'BF16': ('BF16',), 'FP8': ('E4M3', 'E4M3FNUZ')}
# where each panel's legend has room: above the lines, and FP8's plateaus
LEGENDS = {'BF16': ('upper right', 1), 'FP8': ('upper right', 1)}
WIDTH = 2.8
"""Inches: two of `plot_grid.py`'s square panels at its height; its key may
run past them (the saved figure is cropped to it)."""


def load(results: Path) -> tuple[list[dict], dict[str, dict[str, dict]]]:
    """Each graded point of `explore.py`'s table, with its errors by case
    `(test, n, s)`; and the FP32 FMA loop's and the ±1 ULP floor's, by
    baseline and format."""
    points: dict[str, dict] = {}
    bases: dict[str, dict[str, dict]] = {'fma': {}, 'floor': {}}
    with (results / 'blas' / 'blas-grid.csv').open() as f:
        for r in csv.DictReader(f):
            if r['status'] != 'ok':
                continue
            case = (r['test'], int(r['n']), int(r['s']))
            d = points.setdefault(r['design'], {
                'anchor': r['anchor'], 'format': r['format'], 'is_anchor': r['is_anchor'] == 'True',
                'F': int(r['F']), 'join': r['join'], 'errors': {}})
            d['errors'][case] = float(r['error'])
            for b, base in bases.items():
                if r.get(f'{b}_error'):   # older tables have no floor
                    base.setdefault(r['format'], {'errors': {}})['errors'][case] = float(r[f'{b}_error'])
    return list(points.values()), bases


def size(d: dict) -> int:
    """The matrix size d was graded at, the largest if several."""
    return max(m for _, m, _ in d['errors'])


def worst(d: dict) -> float:
    """d's worst error at its :func:`size`, over both tests and every s."""
    n = size(d)
    return max(e for (_, m, _), e in d['errors'].items() if m == n)


def figure(points: list[dict], floor: dict[str, dict]) -> plt.Figure:
    """The worst error at the graded size n against k, a panel per input format: a line per
    generation and join, each generation's own design ringed; and the ±1 ULP
    *floor*'s worst error on the same matrices, the lowest of the panel's
    formats', as `plot_grid.py`'s.  Drawn as its figure, at its height."""
    fig = plt.figure(figsize=(WIDTH, HEIGHT), layout='constrained')
    fig.get_layout_engine().set(h_pad=KEY_GAP)
    axes = fig.subplots(1, len(PANELS))
    for (f, fmts), ax in zip(PANELS.items(), axes):
        mine = [d for d in points if d['format'] in fmts]
        anchors = [a for a in BASES if any(d['anchor'] == a for d in mine)]
        for a in anchors:
            color, marker = _look(a)
            for join, (style, _) in JOINS.items():
                pts = sorted((d['F'], math.log2(worst(d)), d['is_anchor']) for d in mine
                             if d['anchor'] == a and d['join'] == join)
                ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, linestyle=style,
                        linewidth=LINE, marker=marker, markersize=MARKER, markeredgecolor='white',
                        markeredgewidth=EDGE)
                ax.scatter([p[0] for p in pts if p[2]], [p[1] for p in pts if p[2]], s=RING,
                           facecolor='none', edgecolor=color, linewidth=RING_LINE, zorder=3)
        if lows := [worst(floor[f]) for f in fmts if f in floor]:
            ax.axhline(math.log2(min(lows)), color=INK, linestyle=':', linewidth=LINE, zorder=1)
        legend(ax, anchors, *LEGENDS[f])
        ax.set_title(f, loc='left', pad=6 * SCALE, **LARGE)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
        panel(ax)
    axes[0].set_ylabel(r'$\log_2 \eta$', labelpad=4 * SCALE, **LARGE)
    key(fig, '±1 ULP')
    return fig


def numbers(points: list[dict], bases: dict[str, dict[str, dict]]) -> None:
    """Each generation's own design's, the FP32 FMA loop's and the ±1 ULP
    floor's log2 worst error at the graded size, as the figure plots it."""
    for d in points:
        if d['is_anchor']:
            print(f'{d["anchor"]:24} {math.log2(worst(d)):7.2f}')
    for name, b in (('FP32 FMA loop', 'fma'), ('±1 ULP floor', 'floor')):
        for f, e in bases[b].items():
            print(f'{name + ", " + f:24} {math.log2(worst(e)):7.2f}')


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=str(__doc__).splitlines()[1])
    ap.add_argument('--results', type=Path, default=RESULTS,
                    help="the outputs' root, as `explore.py`'s (default: %(default)s)")
    args = ap.parse_args(argv)
    points, bases = load(args.results)
    fig = figure(points, bases['floor'])
    on_ticks(fig)
    save(fig, args.results / 'blas' / 'plots' / 'k', tight=True)
    numbers(points, bases)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
