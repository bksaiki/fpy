"""
The grid's plots, from `sweep.py --grid`'s table.

KL(ref ‖ design) against `F`, a line per generation (by colour and marker)
and join (how `+C` is added: solid its own, dashed the other vendor's): a
panel per input format, as deployed (free), then with quantization held to
ref (frozen); each generation's own design ringed, one that does not read `F`
(CDNA2's) a level line.  Each group's panels share a scale; as deployed, designs
carry their standard errors.  Writes `<results>/plots/<table>.png` and `.pdf`.

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
from plot_sweep import COLORS, FAMILIES, GRID, INK, MARKERS, MUTED, RESULTS

PANELS = (('deployed', 'free_kl', 'k0_kl'), ('held to ref', 'kl', 'floor_kl'))
"""Each panel: its header, its KL column, and its ±1 ulp column."""
GENERATIONS = ('Blackwell', 'Hopper', 'Ada', 'CDNA3', 'CDNA2')
"""Each generation, its colour and marker by position."""
BASES = {'nv.hopper.bf16.f32': ('Hopper', 'Hopper/Blackwell'), 'nv.ampere.bf16.f32': ('Ada', 'Ada'),
         'amd.cdna3.bf16': ('CDNA3', 'CDNA3'), 'amd.cdna2.bf16': ('CDNA2', 'CDNA2'),
         'nv.blackwell.e4m3.f32': ('Blackwell', 'Blackwell'), 'nv.hopper.e4m3.f32': ('Hopper', 'Hopper'),
         'nv.ada.e4m3.f32': ('Ada', 'Ada'), 'amd.cdna3.fp8': ('CDNA3', 'CDNA3')}
"""Each anchor: its generation, and its legend name (MMA-Sim's Blackwell
BF16 is Hopper's, Ada's Ampere's)."""
JOINS = {False: ('-', 'own'), True: ('--', "other vendor's")}
"""Each join, by whether swapped: its line style and legend label."""


def _swapped(r: dict) -> bool:
    """Is *r* its anchor with the other vendor's join?"""
    return r['value'].endswith('join=swapped')


def figure(rows: list[dict], out: Path) -> None:
    rows = [r for r in rows if r['status'] == 'ok']
    family = {r['anchor']: FAMILIES.get(r['scheme'], r['scheme']) for r in rows}
    families = sorted(set(family.values()))
    fig = plt.figure(figsize=(4 * len(families) * len(PANELS), 4.2), layout='constrained')
    grids = []
    for sub, (title, _, _) in zip(fig.subfigures(1, len(PANELS)), PANELS):
        sub.suptitle(title, fontsize=11, color=INK)
        grids.append(sub.subplots(1, len(families), squeeze=False, sharey=True)[0])
    for i, f in enumerate(families):
        axes = [g[i] for g in grids]
        mine = sorted({a for a in family if family[a] == f}, key=lambda a: list(BASES).index(a) if a in BASES else 99)
        for a in mine:
            _lines(axes, [r for r in rows if r['anchor'] == a], a)
        handles = [plt.Line2D([], [], color=COLORS[g], marker=MARKERS[g], linewidth=2, markeredgecolor='white',
                              label=BASES.get(a, (a, a))[1])
                   for a in mine for g in [GENERATIONS.index(BASES[a][0]) if a in BASES else 0]]
        axes[0].legend(handles=handles, fontsize=8, frameon=False, loc='upper right')
        for ax in axes:
            ax.set_title(f, fontsize=10, color=INK, loc='left')
            if i == 0:
                ax.set_ylabel('log10 KL(ref ‖ design)', fontsize=9, color=INK)
    handles = [plt.Line2D([], [], linestyle='', label='+C')]
    handles += [plt.Line2D([], [], color=MUTED, linestyle=s, linewidth=2, label=lab) for s, lab in JOINS.values()]
    handles += [plt.Line2D([], [], linestyle='', label=' ')]
    handles += [plt.Line2D([], [], color=MUTED, linestyle=':', linewidth=1, label='±1 ulp')]
    leg = fig.legend(handles=handles, loc='outside upper center', ncol=len(handles), fontsize=8,
                     frameon=False, handlelength=2.4, columnspacing=1.2)
    leg.get_texts()[0].set_fontweight('bold')
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ('.png', '.pdf'):
        fig.savefig(out.with_suffix(ext), dpi=200)
    plt.close(fig)


def _lines(axes: list[plt.Axes], rows: list[dict], anchor: str) -> None:
    """One anchor's lines in its row's panels, as deployed and held to ref:
    its own join and the other vendor's."""
    g = GENERATIONS.index(BASES[anchor][0]) if anchor in BASES else 0
    color, marker = COLORS[g], MARKERS[g]
    for ax, (_, col, ref) in zip(axes, PANELS):
        for swapped, (style, _) in JOINS.items():
            pts = sorted((int(r['F']), float(r[col]), float(r.get(f'{col}_se') or 0), r['is_anchor'] == 'True')
                         for r in rows if _swapped(r) == swapped and r.get(col))
            if not pts:
                continue
            assert len({p[0] for p in pts}) == len(pts), f'{anchor}: two points at one F'
            xs, ys = [p[0] for p in pts], [math.log10(p[1]) for p in pts]
            if len(pts) == 1:
                # a design that does not read `F`: a level line
                ax.axhline(ys[0], color=color, linestyle=style, linewidth=1.5, zorder=1)
            ax.plot(xs, ys, color=color, linestyle=style, linewidth=2, marker=marker, markersize=4,
                    markeredgecolor='white')
            if col == 'free_kl':
                lo = [y - math.log10(max(p[1] - p[2], p[1] / 10)) for y, p in zip(ys, pts)]
                hi = [math.log10(p[1] + p[2]) - y for y, p in zip(ys, pts)]
                ax.errorbar(xs, ys, yerr=[lo, hi], fmt='none', ecolor=color, elinewidth=1, capsize=2)
            for x, y, p in zip(xs, ys, pts):
                if p[3]:
                    ax.scatter(x, y, s=130, facecolor='none', edgecolor=color, linewidth=1.5, zorder=3)
        ax.axhline(math.log10(float(rows[0][ref])), color=MUTED, linestyle=':', linewidth=1, zorder=1)
        ax.set_xlabel('F (bits)', fontsize=9, color=INK)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, steps=[1, 2, 4, 5, 10]))
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.grid(color=GRID, linewidth=0.8)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)


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
