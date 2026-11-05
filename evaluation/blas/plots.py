"""
The BLAS evaluation's figure and numbers, from `explore.py`'s table, drawn as
`plot_grid.py` draws the grid's KL.

    python evaluation/blas/plots.py    # -> evaluation/results/blas/plots/k.{png,pdf}; numbers on stdout
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

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'evaluation' / 'results'
sys.path.insert(0, str(ROOT / 'examples' / 'mmasim' / 'serve'))

# `plot_grid.py`'s look: each generation's colour and marker, each join's
# line, its colours and its text styles
from plot_grid import BASES, GRID, INK, JOINS, LARGE, MUTED, SMALL, _look

U = 2.0 ** -24
N = 1024
# each panel's input formats; the FP32 FMA loop drawn is the first's
PANELS = {'BF16': ('BF16',), 'FP8': ('E4M3', 'E4M3FNUZ')}
# where each panel's legend has room, and its columns: above FP8's plateaus, two
LEGENDS = {'BF16': ('upper right', 1), 'FP8': ('upper right', 2)}


def load(results: Path) -> tuple[list[dict], dict[str, dict]]:
    """Each graded point of `explore.py`'s table, with its errors by case
    `(test, n, s)`; and the FP32 FMA loop's, by format."""
    points: dict[str, dict] = {}
    fma: dict[str, dict] = {}
    with (results / 'blas' / 'blas-grid.csv').open() as f:
        for r in csv.DictReader(f):
            if r['status'] != 'ok':
                continue
            case = (r['test'], int(r['n']), int(r['s']))
            d = points.setdefault(r['design'], {
                'anchor': r['anchor'], 'format': r['format'], 'is_anchor': r['is_anchor'] == 'True',
                'F': int(r['F']), 'join': r['join'], 'errors': {}})
            d['errors'][case] = float(r['error'])
            fma.setdefault(r['format'], {'errors': {}})['errors'][case] = float(r['fma_error'])
    return list(points.values()), fma


def worst(d: dict, n: int = N) -> float:
    """d's worst error at n, over both tests and every s."""
    return max(e for (_, m, _), e in d['errors'].items() if m == n)


def figure(points: list[dict], fma: dict[str, dict]) -> plt.Figure:
    """The worst error at n = N against k, a panel per input format: a line per
    generation and join, each generation's own design ringed; the FP32 FMA
    loop's worst error on the same matrices, measured, and an FP32 dot
    product's worst case `n u`."""
    # 4 inches a panel, as `plot_grid.py`'s figure, so the two print at one scale
    fig, axes = plt.subplots(1, len(PANELS), figsize=(4 * len(PANELS), 4.2), layout='constrained')
    fig.get_layout_engine().set(h_pad=0.06)
    for (f, fmts), ax in zip(PANELS.items(), axes):
        mine = [d for d in points if d['format'] in fmts]
        anchors = [a for a in BASES if any(d['anchor'] == a for d in mine)]
        for a in anchors:
            color, marker = _look(a)
            for join, (style, _) in JOINS.items():
                pts = sorted((d['F'], math.log2(worst(d)), d['is_anchor']) for d in mine
                             if d['anchor'] == a and d['join'] == join)
                ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, linestyle=style,
                        linewidth=2, marker=marker, markersize=6, markeredgecolor='white')
                ax.scatter([p[0] for p in pts if p[2]], [p[1] for p in pts if p[2]], s=200,
                           facecolor='none', edgecolor=color, linewidth=1.5, zorder=3)
        ax.axhline(math.log2(worst(fma[fmts[0]])), color=MUTED, linewidth=1.5, zorder=1)
        ax.axhline(math.log2(N * U), color=INK, linestyle=':', linewidth=2, zorder=1)
        ax.legend(handles=[plt.Line2D([], [], color=c, marker=m, linewidth=2, markeredgecolor='white',
                                      label=BASES[a][1]) for a in anchors for c, m in [_look(a)]],
                  loc=LEGENDS[f][0], ncol=LEGENDS[f][1], fontsize=SMALL['fontsize'], labelcolor=SMALL['color'],
                  frameon=False, columnspacing=0.8)
        ax.set_title(f'{f}, n = {N}', loc='left', **LARGE)
        ax.set_xlabel('k (bits)', **LARGE)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, steps=[1, 2, 4, 5, 10]))
        ax.tick_params(colors=SMALL['color'], labelsize=SMALL['fontsize'])
        ax.grid(color=GRID, linewidth=0.8)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        # limits widened to the nearest ticks
        lo, hi = ax.get_ylim()
        ticks = ax.yaxis.get_major_locator().tick_values(lo, hi)
        ax.set_ylim(max(t for t in ticks if t <= lo), min(t for t in ticks if t >= hi))
    axes[0].set_ylabel(r'log$_2$ worst $|\hat C - C| \,/\, (|A|\,|B|)$', **LARGE)
    (_, with_), (_, after) = JOINS.values()
    # two rows, `c added` over `FP32 FMA worst case`: a legend fills column by column
    handles = [plt.Line2D([], [], linestyle='None', label='c added'),
               plt.Line2D([], [], linestyle='None', label='FP32 FMA worst case'),
               plt.Line2D([], [], color=MUTED, ls=JOINS['inside'][0], linewidth=2, label=with_),
               plt.Line2D([], [], color=MUTED, linewidth=1.5, label='measured'),
               plt.Line2D([], [], color=MUTED, ls=JOINS['separate'][0], linewidth=2, label=after),
               plt.Line2D([], [], color=INK, linestyle=':', linewidth=2, label='theoretical')]
    leg = fig.legend(handles=handles, loc='outside upper center', ncol=3,
                     fontsize=SMALL['fontsize'], labelcolor=SMALL['color'], frameon=False,
                     handlelength=2.4, columnspacing=1.2)
    for header in (0, 1):
        leg.get_texts()[header].set(fontweight='bold', **LARGE)
    return fig


def numbers(points: list[dict], fma: dict[str, dict]) -> None:
    """Each generation's own design's and the FP32 FMA loop's log2 worst
    error at n = N, as the figure plots it."""
    for d in points:
        if d['is_anchor']:
            print(f'{d["anchor"]:24} {math.log2(worst(d)):7.2f}')
    for f, b in fma.items():
        print(f'{"FP32 FMA loop, " + f:24} {math.log2(worst(b)):7.2f}')


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=str(__doc__).splitlines()[1])
    ap.add_argument('--results', type=Path, default=RESULTS,
                    help="the outputs' root, as `explore.py`'s (default: %(default)s)")
    args = ap.parse_args(argv)
    points, fma = load(args.results)
    out = args.results / 'blas' / 'plots'
    out.mkdir(parents=True, exist_ok=True)
    fig = figure(points, fma)
    for ext in ('png', 'pdf'):
        fig.savefig(out / f'k.{ext}', dpi=200)
    numbers(points, fma)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
