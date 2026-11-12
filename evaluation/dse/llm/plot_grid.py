"""
The grid's plots, from `sweep.py --grid`'s table.

KL(ref ‖ design) against `F` (the paper's `k`), a line per generation (by colour and marker)
and when `c` is added (solid: with the products, dashed: after them): per
input format, a panel observed (free, as deployed), then amplified (frozen,
quantization held to ref); each generation's own design ringed, one that does
not read `F` (CDNA2's) a point alone.  Observed, designs carry their standard
errors.  Writes `<results>/plots/<table>.png` and `.pdf`.

    python plot_grid.py --table sweep-grid-Qwen3.5-0.8B.csv
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

sys.path.append(str(Path(__file__).resolve().parents[2]))

from common.plot import FAMILIES as HARDWARE
from common.plot import GRID, INK, MUTED, TEXTWIDTH, save

RESULTS = Path(__file__).resolve().parents[2] / 'results' / 'llm'
MARKERS = ('o', 's', '^', 'D', 'v')
FAMILIES = {'fp8-row': 'FP8', 'fp8-row:fnuz': 'FP8', 'bf16': 'BF16', 'mxfp8': 'FP8'}
"""Each scheme's input format, its panels' title."""

PANELS = (('observed', 'free_kl', 'k0_kl'), ('amplified', 'kl', 'floor_kl'))
"""Each panel: its header, its KL column, and its ±1 ULP column."""
SCALE = TEXTWIDTH / 16
"""The paper's figure was drawn 16 inches wide and printed at the text
width; it is now drawn at print size, every size below scaled by this."""
LARGE, SMALL = {'fontsize': 14 * SCALE, 'color': INK}, {'fontsize': 12 * SCALE, 'color': MUTED}
"""The two text styles: titles, axis labels and the key; ticks and legends."""
LINE, MARKER, EDGE = 2 * SCALE, 6 * SCALE, 0.8 * SCALE
"""Points: data lines, their markers, and axes, ticks and grid lines."""
RING, RING_LINE = 200 * SCALE ** 2, 1.5 * SCALE
"""Points squared and points: the ring marking an existing design's point."""
KEY_GAP = 0.03
"""Inches: layout padding that leaves the key about 0.06 in above the panel
titles, as the paper printed it."""
HEIGHT = 1.6
"""Inches: the figure's, and the BLAS figure's, for square panels."""
GENERATIONS = ('Blackwell', 'Hopper', 'Ada', 'CDNA3', 'CDNA2')
"""Each generation (a family of `HARDWARE`), its marker by position."""
BASES = {'nv.hopper.bf16.f32': ('Hopper', 'Hopper'), 'nv.ampere.bf16.f32': ('Ada', 'Ada'),
         'amd.cdna3.bf16': ('CDNA3', 'CDNA3'), 'amd.cdna2.bf16': ('CDNA2', 'CDNA2'),
         'nv.hopper.e4m3.f32': ('Hopper', 'Hopper'),
         'nv.ada.e4m3.f32': ('Ada', 'Ada'), 'amd.cdna3.fp8': ('CDNA3', 'CDNA3'),
         'nv.blackwell.mx.e4m3': ('Blackwell', 'Blackwell')}
"""Each anchor: its generation, and its legend name (MMA-Sim's Blackwell
BF16 is Hopper's, Ada's Ampere's)."""
JOINS = {'inside': ('-', 'with the products'), 'separate': ('--', 'after the products')}
"""Each join: its line style and legend label."""


def figure(rows: list[dict], out: Path) -> None:
    rows = [r for r in rows if r['status'] == 'ok']
    family = {r['anchor']: FAMILIES.get(r['scheme'], r['scheme']) for r in rows}
    families = sorted(set(family.values()))
    fig = plt.figure(figsize=(TEXTWIDTH, HEIGHT), layout='constrained')
    fig.get_layout_engine().set(h_pad=KEY_GAP)
    axes = fig.subplots(1, len(families) * len(PANELS))
    for i, f in enumerate(families):
        mine = axes[i * len(PANELS):(i + 1) * len(PANELS)]
        anchors = [a for a in BASES if family.get(a) == f]
        for a in anchors:
            _lines(mine, [r for r in rows if r['anchor'] == a], a)
        legend(mine[0], anchors, 'upper right')
        for ax, (title, _, ref) in zip(mine, PANELS):
            ax.set_title(f'{f} ({title})', loc='left', pad=6 * SCALE, **LARGE)
            # one ±1 ULP line, the lowest of the bases' (their quantizations differ)
            floor = min(float(r[ref]) for r in rows if family[r['anchor']] == f and r.get(ref))
            ax.axhline(math.log10(floor), color=INK, linestyle=':', linewidth=LINE, zorder=1)
            panel(ax)
    axes[0].set_ylabel(r'$\log_{10}$ KL(ref ‖ design)', labelpad=4 * SCALE, **LARGE)
    key(fig, '±1 ULP')
    on_ticks(fig)
    save(fig, out, tight=True)


def on_ticks(fig: plt.Figure) -> None:
    """*fig*'s panels' k-axes from their first k to their last, both ticks,
    as their y-axes, with points on the edge drawn whole.  After the layout,
    fixed: a last tick's label past a panel's edge would otherwise narrow
    every panel."""
    fig.canvas.draw()
    fig.set_layout_engine('none')
    for ax in fig.axes:
        lo, hi = ax.dataLim.intervalx
        ticks = ax.xaxis.get_major_locator().tick_values(lo, hi)
        ax.set_xlim(max(t for t in ticks if t <= lo), min(t for t in ticks if t >= hi))
        for a in (*ax.lines, *ax.collections):
            a.set_clip_on(False)


def panel(ax: plt.Axes) -> None:
    """A k-axis panel: square, its ticks, grid and y limits (widened to the
    nearest ticks); :func:`on_ticks` ends its k-axis on ticks."""
    ax.set_box_aspect(1)
    ax.set_xlabel('k (bits)', labelpad=4 * SCALE, **LARGE)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, steps=[1, 2, 4, 5, 10]))
    ax.tick_params(colors=SMALL['color'], labelsize=SMALL['fontsize'], length=3.5 * SCALE, width=EDGE,
                   pad=3.5 * SCALE)
    ax.grid(color=GRID, linewidth=EDGE)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_linewidth(EDGE)
    lo, hi = ax.get_ylim()
    ticks = ax.yaxis.get_major_locator().tick_values(lo, hi)
    ax.set_ylim(max(t for t in ticks if t <= lo), min(t for t in ticks if t >= hi))


def legend(ax: plt.Axes, anchors: list[str], loc: str, ncol: int = 1) -> None:
    """*anchors*' generations in *ax*, at its empty *loc*."""
    ax.legend(handles=[plt.Line2D([], [], color=c, marker=m, linewidth=LINE, markersize=MARKER,
                                  markeredgecolor='white', markeredgewidth=EDGE, label=BASES[a][1])
                       for a in anchors for c, m in [_look(a)]],
              loc=loc, ncol=ncol, fontsize=SMALL['fontsize'], labelcolor=SMALL['color'], frameon=False,
              columnspacing=0.8)


def key(fig: plt.Figure, dotted: str) -> None:
    """The joins' line styles and the *dotted* line, above the panels."""
    handles = [plt.Line2D([], [], linestyle='', label='c added')]
    handles += [plt.Line2D([], [], color=MUTED, linestyle=s, linewidth=LINE, label=lab) for s, lab in JOINS.values()]
    handles += [plt.Line2D([], [], linestyle='', label=' ')]
    handles += [plt.Line2D([], [], color=INK, linestyle=':', linewidth=LINE, label=dotted)]
    leg = fig.legend(handles=handles, loc='outside upper center', ncol=len(handles), fontsize=LARGE['fontsize'],
                     labelcolor=LARGE['color'], frameon=False, handlelength=2.4, columnspacing=1.2)
    leg.get_texts()[0].set(fontweight='bold')


def _look(anchor: str) -> tuple[str, str]:
    """*anchor*'s colour and marker, its generation's."""
    g = BASES[anchor][0]
    return HARDWARE[g][0], MARKERS[GENERATIONS.index(g)]


def _lines(axes: list[plt.Axes], rows: list[dict], anchor: str) -> None:
    """One anchor's lines in its format's panels, observed and amplified:
    a line per join."""
    color, marker = _look(anchor)
    for ax, (_, col, _) in zip(axes, PANELS):
        for join, (style, _) in JOINS.items():
            pts = sorted((int(r['F']), float(r[col]), float(r.get(f'{col}_se') or 0), r['is_anchor'] == 'True')
                         for r in rows if r['join'] == join and r.get(col))
            if not pts:
                continue
            assert len({p[0] for p in pts}) == len(pts), f'{anchor}: two points at one F'
            xs, ys = [p[0] for p in pts], [math.log10(p[1]) for p in pts]
            ax.plot(xs, ys, color=color, linestyle=style, linewidth=LINE, marker=marker, markersize=MARKER,
                    markeredgecolor='white', markeredgewidth=EDGE)
            if col == 'free_kl':
                lo = [y - math.log10(max(p[1] - p[2], p[1] / 10)) for y, p in zip(ys, pts)]
                hi = [math.log10(p[1] + p[2]) - y for y, p in zip(ys, pts)]
                ax.errorbar(xs, ys, yerr=[lo, hi], fmt='none', ecolor=color, elinewidth=SCALE, capsize=2 * SCALE)
            for x, y, p in zip(xs, ys, pts):
                if p[3]:
                    ax.scatter(x, y, s=RING, facecolor='none', edgecolor=color, linewidth=RING_LINE, zorder=3)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('--results', type=Path, default=RESULTS,
                    help="the outputs' root (default: %(default)s)")
    ap.add_argument('--table', required=True, help='the table, under <results>/sweep')
    args = ap.parse_args(argv)
    table = args.results / 'sweep' / args.table
    with table.open() as f:
        rows = list(csv.DictReader(f))
    out = args.results / 'plots' / f'{table.stem}.png'
    figure(rows, out)
    print(out)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
