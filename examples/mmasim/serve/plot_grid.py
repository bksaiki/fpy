"""
The grid's plots, from `sweep.py --grid`'s table.

KL(ref ‖ design) against `F` (the paper's `k`), a line per generation (by colour and marker)
and when `c` is added (solid: with the products, dashed: after them): per
input format, a panel observed (free, as deployed), then amplified (frozen,
quantization held to ref); each generation's own design ringed, one that does
not read `F` (CDNA2's) a point alone.  Observed, designs carry their standard
errors.  Writes `<results>/plots/<table>.png` and `.pdf`.

    python serve/plot_grid.py --table sweep-grid-Qwen3.5-0.8B.csv
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

RESULTS = Path('results')
COLORS = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4')
MARKERS = ('o', 's', '^', 'D', 'v')
INK, MUTED, GRID = '#1f1f1e', '#6b6a64', '#e4e3dc'
FAMILIES = {'fp8-row': 'FP8', 'fp8-row:fnuz': 'FP8', 'bf16': 'BF16', 'mxfp8': 'FP8'}
"""Each scheme's input format, its panels' title."""

PANELS = (('observed', 'free_kl', 'k0_kl'), ('amplified', 'kl', 'floor_kl'))
"""Each panel: its header, its KL column, and its ±1 ulp column."""
GENERATIONS = ('Blackwell', 'Hopper', 'Ada', 'CDNA3', 'CDNA2')
"""Each generation, its colour and marker by position."""
BASES = {'nv.hopper.bf16.f32': ('Hopper', 'Hopper/Blackwell'), 'nv.ampere.bf16.f32': ('Ada', 'Ada'),
         'amd.cdna3.bf16': ('CDNA3', 'CDNA3'), 'amd.cdna2.bf16': ('CDNA2', 'CDNA2'),
         'nv.hopper.e4m3.f32': ('Hopper', 'Hopper'),
         'nv.ada.e4m3.f32': ('Ada', 'Ada'), 'amd.cdna3.fp8': ('CDNA3', 'CDNA3'),
         'nv.blackwell.mx.e4m3': ('Blackwell', 'Blackwell')}
"""Each anchor: its generation, and its legend name (MMA-Sim's Blackwell
BF16 is Hopper's, Ada's Ampere's)."""
LARGE, SMALL = {'fontsize': 14, 'color': INK}, {'fontsize': 12, 'color': MUTED}
"""The two text styles: titles and axis labels; ticks and legends."""
JOINS = {'inside': ('-', 'with the products'), 'separate': ('--', 'after the products')}
"""Each join: its line style and legend label."""


def figure(rows: list[dict], out: Path) -> None:
    rows = [r for r in rows if r['status'] == 'ok']
    family = {r['anchor']: FAMILIES.get(r['scheme'], r['scheme']) for r in rows}
    families = sorted(set(family.values()))
    fig = plt.figure(figsize=(4 * len(families) * len(PANELS), 4.2), layout='constrained')
    fig.get_layout_engine().set(h_pad=0.06)
    for i, (f, sub) in enumerate(zip(families, fig.subfigures(1, len(families)))):
        axes = sub.subplots(1, len(PANELS), squeeze=False)[0]
        mine = [a for a in BASES if family.get(a) == f]
        for a in mine:
            _lines(axes, [r for r in rows if r['anchor'] == a], a)
        handles = [plt.Line2D([], [], color=c, marker=m, linewidth=2, markeredgecolor='white', label=BASES[a][1])
                   for a in mine for c, m in [_look(a)]]
        axes[0].legend(handles=handles, fontsize=SMALL['fontsize'], labelcolor=SMALL['color'], frameon=False,
                       loc='upper right')
        for ax, (title, _, ref) in zip(axes, PANELS):
            ax.set_title(f'{f} ({title})', loc='left', **LARGE)
            ax.set_xlabel('k (bits)', **LARGE)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, steps=[1, 2, 4, 5, 10]))
            ax.tick_params(colors=SMALL['color'], labelsize=SMALL['fontsize'])
            ax.grid(color=GRID, linewidth=0.8)
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
            # one ±1 ulp line, the lowest of the bases' (their quantizations differ)
            floor = min(float(r[ref]) for r in rows if family[r['anchor']] == f and r.get(ref))
            ax.axhline(math.log10(floor), color=INK, linestyle=':', linewidth=2, zorder=1)
            # limits widened to the nearest ticks
            lo, hi = ax.get_ylim()
            ticks = ax.yaxis.get_major_locator().tick_values(lo, hi)
            ax.set_ylim(max(t for t in ticks if t <= lo), min(t for t in ticks if t >= hi))
        if i == 0:
            axes[0].set_ylabel('log10 KL(ref ‖ design)', **LARGE)
    handles = [plt.Line2D([], [], linestyle='', label='c added')]
    handles += [plt.Line2D([], [], color=MUTED, linestyle=s, linewidth=2, label=lab) for s, lab in JOINS.values()]
    handles += [plt.Line2D([], [], linestyle='', label=' ')]
    handles += [plt.Line2D([], [], color=INK, linestyle=':', linewidth=2, label='±1 ulp')]
    leg = fig.legend(handles=handles, loc='outside upper center', ncol=len(handles), fontsize=SMALL['fontsize'],
                     labelcolor=SMALL['color'], frameon=False, handlelength=2.4, columnspacing=1.2)
    leg.get_texts()[0].set(fontweight='bold', **LARGE)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ('.png', '.pdf'):
        fig.savefig(out.with_suffix(ext), dpi=200)
    plt.close(fig)


def _look(anchor: str) -> tuple[str, str]:
    """*anchor*'s colour and marker, its generation's."""
    g = GENERATIONS.index(BASES[anchor][0])
    return COLORS[g], MARKERS[g]


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
            ax.plot(xs, ys, color=color, linestyle=style, linewidth=2, marker=marker, markersize=6,
                    markeredgecolor='white')
            if col == 'free_kl':
                lo = [y - math.log10(max(p[1] - p[2], p[1] / 10)) for y, p in zip(ys, pts)]
                hi = [math.log10(p[1] + p[2]) - y for y, p in zip(ys, pts)]
                ax.errorbar(xs, ys, yerr=[lo, hi], fmt='none', ecolor=color, elinewidth=1, capsize=2)
            for x, y, p in zip(xs, ys, pts):
                if p[3]:
                    ax.scatter(x, y, s=200, facecolor='none', edgecolor=color, linewidth=1.5, zorder=3)


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
