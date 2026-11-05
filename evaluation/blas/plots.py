"""
The BLAS evaluation's figure and numbers, from `explore.py`'s table, drawn as
`plot_grid.py` draws the grid's KL.

    python evaluation/blas/plots.py    # -> evaluation/results/blas/plots/{k,compile,compile-sorted,compile-sorted-gen}.{png,pdf}; numbers on stdout
"""

import argparse
import csv
import math
import statistics
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


def load(results: Path) -> tuple[list[dict], dict[str, dict]]:
    """Each graded point of `explore.py`'s table, with its errors by case
    `(test, n, s)`; and the FP32 FMA loop's, by format."""
    points: dict[str, dict] = {}
    fma: dict[str, dict] = {}
    with (results / 'blas' / 'blas-grid.csv').open() as f:
        for r in csv.DictReader(f):
            # refused, or not graded (`explore.py --no-grade`)
            if r['status'] != 'ok' or not r.get('test'):
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


def eta(d: dict) -> float:
    """d's worst error over `n u`, FP32's worst case, at any n: GtB's grade A below 1."""
    return max(e / (n * U) for (_, n, _), e in d['errors'].items())


def figure(points: list[dict], fma: dict[str, dict]) -> plt.Figure:
    """The worst error at n = N against k, a panel per input format: a line per
    generation and join, each generation's own design ringed; the FP32 FMA
    loop's worst error on the same matrices, measured, and an FP32 dot
    product's worst case `n u`, below which a design is GtB's grade A."""
    fig, axes = plt.subplots(1, len(PANELS), figsize=(13, 4.2), layout='constrained')
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
                  fontsize=SMALL['fontsize'], labelcolor=SMALL['color'], frameon=False)
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
    handles = [plt.Line2D([], [], linestyle='None', label='c added'),
               *(plt.Line2D([], [], color=MUTED, ls=s, linewidth=2, label=lab) for s, lab in JOINS.values()),
               plt.Line2D([], [], linestyle='None', label='FP32 FMA worst case'),
               plt.Line2D([], [], color=MUTED, linewidth=1.5, label='measured'),
               plt.Line2D([], [], color=INK, linestyle=':', linewidth=2, label='theoretical')]
    leg = fig.legend(handles=handles, loc='outside upper center', ncol=len(handles),
                     fontsize=SMALL['fontsize'], labelcolor=SMALL['color'], frameon=False,
                     handlelength=2.4, columnspacing=1.2)
    for header in (0, 3):
        leg.get_texts()[header].set(fontweight='bold', **LARGE)
    return fig


def compile_times(results: Path) -> list[dict]:
    """Each point of `explore.py`'s table, graded or refused, with the seconds
    FPy took to compile it to C++."""
    points: dict[str, dict] = {}
    with (results / 'blas' / 'blas-grid.csv').open() as f:
        for r in csv.DictReader(f):
            points.setdefault(r['design'], {
                'anchor': r['anchor'], 'format': r['format'], 'is_anchor': r['is_anchor'] == 'True',
                'F': int(r['F']), 'join': r['join'], 'refused': r['status'] == 'refused',
                'seconds': float(r['fpy_seconds'])})
    return list(points.values())


def compile_figure(points: list[dict]) -> plt.Figure:
    """FPy's compile time, a bar per design, a panel per input format: grouped
    by generation, and within each by join, then k; hatched: `c` added after
    the products; a cross on the axis: failed, its time not a compile's; each
    generation's own design ringed."""
    groups = {f: [a for a in BASES if any(d['anchor'] == a and d['format'] in fmts for d in points)]
              for f, fmts in PANELS.items()}
    widths = [sum(len([d for d in points if d['anchor'] == a]) + 2 for a in anchors) for anchors in groups.values()]
    fig, axes = plt.subplots(1, len(PANELS), figsize=(13, 4.2), layout='constrained', sharey=True,
                             gridspec_kw={'width_ratios': widths})
    for (f, anchors), ax in zip(groups.items(), axes):
        x, ticks = 0, []
        for a in anchors:
            color, _ = _look(a)
            mine = sorted((d for d in points if d['anchor'] == a),
                          key=lambda d: (list(JOINS).index(d['join']), d['F']))
            for d in mine:
                if d['refused']:
                    ax.scatter(x, 0, s=40, marker='x', color=color, linewidth=1.5, clip_on=False, zorder=3)
                    x += 1
                    continue
                hatch = '////' if d['join'] == 'separate' else None
                ax.bar(x, d['seconds'], width=0.7, color='white' if hatch else color,
                       edgecolor=color, hatch=hatch, linewidth=0.9)
                if d['is_anchor']:
                    ax.scatter(x, d['seconds'] + 1.5, s=100, facecolor='none', edgecolor=color, linewidth=1.5)
                x += 1
            ticks.append(x - (len(mine) + 1) / 2)
            x += 2
        ax.set_xticks(ticks, [BASES[a][1] for a in anchors])
        ax.set_xlim(-1.5, x - 1.5)
        ax.set_title(f, loc='left', **LARGE)
        ax.tick_params(colors=SMALL['color'], labelsize=SMALL['fontsize'])
        ax.tick_params(axis='x', length=0)
        ax.grid(axis='y', color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
    # limits widened to the nearest ticks
    hi = axes[0].get_ylim()[1]
    axes[0].set_ylim(0, min(t for t in axes[0].yaxis.get_major_locator().tick_values(0, hi) if t >= hi))
    axes[0].set_ylabel('FPy to C++ (s)', **LARGE)
    swatch = {'with the products': {'facecolor': MUTED}, 'after the products': {'facecolor': 'white', 'hatch': '////'}}
    handles = [plt.Line2D([], [], linestyle='None', label='c added'),
               *(plt.Rectangle((0, 0), 1, 1, edgecolor=MUTED, label=lab, **swatch[lab]) for _, lab in JOINS.values()),
               plt.Line2D([], [], linestyle='None', label=' '),
               plt.Line2D([], [], linestyle='None', marker='x', markersize=7, markeredgewidth=1.5, color=MUTED,
                          label='failed')]
    leg = fig.legend(handles=handles, loc='outside upper center', ncol=len(handles),
                     fontsize=SMALL['fontsize'], labelcolor=SMALL['color'], frameon=False,
                     handlelength=2.4, columnspacing=1.2)
    leg.get_texts()[0].set(fontweight='bold', **LARGE)
    return fig


def sorted_figure(points: list[dict], by_generation: bool) -> plt.Figure:
    """FPy's compile time, a bar per compiled design sorted by it, drawn as the
    performance evaluation's sorted speedups: in one colour, or in each
    generation's; hatched when `c` is added after the products; each
    generation's own design ringed, and the median a dotted line."""
    mine = sorted((d for d in points if not d['refused']), key=lambda d: d['seconds'])
    fig, ax = plt.subplots(figsize=(13, 4.2), layout='constrained')
    colors = [_look(d['anchor'])[0] if by_generation else MUTED for d in mine]
    for x, (d, color) in enumerate(zip(mine, colors)):
        hatch = '////' if d['join'] == 'separate' else None
        ax.bar(x, d['seconds'], width=0.8, color='white' if hatch else color, edgecolor=color, hatch=hatch,
               linewidth=0.9)
    ax.set_xticks([])
    ax.set_xlim(-0.7, len(mine) - 0.3)
    ax.set_xlabel(f'{len(mine)} designs, sorted by compile time', **LARGE)
    ax.tick_params(colors=SMALL['color'], labelsize=SMALL['fontsize'])
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    # limits widened to the nearest ticks
    hi = ax.get_ylim()[1]
    top = min(t for t in ax.yaxis.get_major_locator().tick_values(0, hi) if t >= hi)
    ax.set_ylim(0, top)
    ax.set_ylabel('FPy to C++ (s)', **LARGE)
    median = statistics.median(d['seconds'] for d in mine)
    ax.axhline(median, color=INK, linestyle=':', linewidth=2)
    ax.text(0, median + top / 60, f'median {median:.1f} s', va='bottom', **SMALL | {'color': INK})
    for x, (d, color) in enumerate(zip(mine, colors)):
        if d['is_anchor']:
            ax.scatter(x, d['seconds'] + top / 25, s=100, facecolor='none', edgecolor=color, linewidth=1.5)
    if by_generation:
        names = dict.fromkeys(BASES[a][0] for a in BASES)
        handles = [plt.Rectangle((0, 0), 1, 1, color=_look(next(a for a in BASES if BASES[a][0] == g))[0], label=g)
                   for g in names]
        ax.legend(handles=handles, loc='upper left', ncol=2, fontsize=SMALL['fontsize'],
                  labelcolor=SMALL['color'], frameon=False, handlelength=1.2, handleheight=1.0)
    swatch = {'with the products': {'facecolor': MUTED}, 'after the products': {'facecolor': 'white', 'hatch': '////'}}
    handles = [plt.Line2D([], [], linestyle='None', label='c added'),
               *(plt.Rectangle((0, 0), 1, 1, edgecolor=MUTED, label=lab, **swatch[lab]) for _, lab in JOINS.values())]
    leg = fig.legend(handles=handles, loc='outside upper center', ncol=len(handles),
                     fontsize=SMALL['fontsize'], labelcolor=SMALL['color'], frameon=False,
                     handlelength=2.4, columnspacing=1.2)
    leg.get_texts()[0].set(fontweight='bold', **LARGE)
    return fig


def numbers(points: list[dict], fma: dict[str, dict]) -> None:
    """Each generation's own design's and the FP32 FMA loop's `eta`, and how
    many designs are grade A."""
    for d in points:
        if d['is_anchor']:
            print(f'{d["anchor"]:24} {eta(d):9.3g}')
    for f, b in fma.items():
        print(f'{"FP32 FMA loop, " + f:24} {eta(b):9.3g}')
    print(f'grade A: {sum(eta(d) < 1 for d in points)} of {len(points)}')


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=str(__doc__).splitlines()[1])
    ap.add_argument('--results', type=Path, default=RESULTS,
                    help="the outputs' root, as `explore.py`'s (default: %(default)s)")
    args = ap.parse_args(argv)
    points, fma = load(args.results)
    out = args.results / 'blas' / 'plots'
    out.mkdir(parents=True, exist_ok=True)
    times = compile_times(args.results)
    figs = {'compile': compile_figure(times), 'compile-sorted': sorted_figure(times, False),
            'compile-sorted-gen': sorted_figure(times, True)}
    if points:
        figs['k'] = figure(points, fma)
    for name, f in figs.items():
        for ext in ('png', 'pdf'):
            f.savefig(out / f'{name}.{ext}', dpi=200)
    if points:
        numbers(points, fma)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
