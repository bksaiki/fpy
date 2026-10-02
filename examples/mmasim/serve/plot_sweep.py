"""
The sweep's graphs from `sweep.py`'s table alone: for each local metric, a
panel per parameter, how each anchor's design measures as that parameter
varies and the rest hold.  A line per anchor (its colour and marker); its own
value ringed; a point the compiler refused an x on the panel's floor.  A categorical parameter's points are not joined:
its values have no order.  Writes `<results>/plots/sweep/<metric>.png`.

    python serve/plot_sweep.py                      # from examples/mmasim
    python serve/plot_sweep.py --metrics backward_normwise flips
"""

import argparse
import csv
import math
import sys
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import transforms

RESULTS = Path('results')

COLORS = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4')
"""The anchors' hues, in this order (the dataviz reference's categorical
slots 1-5)."""
MARKERS = ('o', 's', '^', 'D', 'v')
INK, MUTED, GRID = '#1f1f1e', '#6b6a64', '#e4e3dc'

U = 2.0 ** -24

METRICS: dict[str, tuple[str, Callable[[float], float]]] = {
    'backward_normwise': ('normwise backward error (log2)', lambda v: v),
    'backward': ('componentwise backward error, mean of log2', lambda v: v),
    'backward_max': ('componentwise backward error, max (log2)', lambda v: v),
    'backward_p99': ('componentwise backward error, p99 (log2)', lambda v: v),
    'normwise': ('normwise relative error (log2)', lambda v: math.log2(v) if v > 0 else math.nan),
    'ulp': ('ULP bits, mean', lambda v: v),
    'magnitude_bias': ('magnitude bias (u)', lambda v: v / U),
    'bias': ('bias (u)', lambda v: v / U),
    'flips': ('flipped codes (log10 share)', lambda v: math.log10(v) if v > 0 else math.nan),
}
"""Each metric as plotted: its label, and the transform from `local.py`'s
value."""

DEFAULT = ('backward_normwise', 'backward', 'backward_max', 'magnitude_bias')

SYMLOG = {'bias', 'magnitude_bias'}
"""Metrics signed and spanning decades: a symmetric log axis, linear within 1 u."""

LOG2 = {'L'}
"""Parameters that double: a log2 axis."""


def _value(v: str) -> int | str:
    try:
        return int(v)
    except ValueError:
        return v


def panel(ax: plt.Axes, rows: list[dict], param: str, metric: str, anchors: list[str]) -> None:
    """*metric* against *param*, a line per anchor of *anchors* that sweeps it."""
    f = METRICS[metric][1]
    mine = [r for r in rows if r['param'] == param]
    values = list(dict.fromkeys(_value(r['value']) for r in mine))
    numeric = all(isinstance(v, int) for v in values)
    order = sorted(values) if numeric else values
    pos = {v: (math.log2(v) if param in LOG2 else v) if numeric else i for i, v in enumerate(order)}
    swept = [a for a in anchors if any(r['anchor'] == a for r in mine)]
    for a, color, marker in zip(anchors, COLORS, MARKERS):
        pts = [r for r in mine if r['anchor'] == a]
        if not pts:
            continue
        ok = sorted((r for r in pts if r['status'] == 'ok' and r.get(metric)),
                    key=lambda r: pos[_value(r['value'])])
        xs = [pos[_value(r['value'])] for r in ok]
        ys = [f(float(r[metric])) for r in ok]
        ax.plot(xs, ys, color=color, linewidth=2 if numeric else 0, marker=marker, markersize=6,
                zorder=2, markeredgecolor='white', markeredgewidth=1)
        for r, x, y in zip(ok, xs, ys):
            if r['is_anchor'] == 'True':
                ax.scatter(x, y, s=150, facecolor='none', edgecolor=color, linewidth=1.5, zorder=3)
        # side by side, so the anchors' marks at one value do not hide each other
        shift = transforms.offset_copy(ax.get_xaxis_transform(), fig=ax.figure, units='points',
                                       x=7 * (swept.index(a) - (len(swept) - 1) / 2))
        for r in pts:
            if r['status'] != 'ok':
                ax.plot(pos[_value(r['value'])], 0.03, marker='x', color=color, markersize=6,
                        transform=shift, clip_on=False, zorder=3)
    ticks = [pos[v] for v in order]
    keep = _spaced(ticks) if numeric and param not in LOG2 else [True] * len(ticks)
    ax.set_xticks(ticks, [str(v) if k else '' for v, k in zip(order, keep)], fontsize=8)
    if not numeric:
        ax.set_xlim(-0.5, len(order) - 0.5)
    if metric in SYMLOG:
        ax.set_yscale('symlog', linthresh=1)
    ax.set_title(param, fontsize=10, color=INK, loc='left')
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(color=GRID, linewidth=0.8)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)


def _spaced(ticks: list[float]) -> list[bool]:
    """Which of *ticks* to label: none within a twentieth of the span of the
    last one labeled."""
    gap = (max(ticks) - min(ticks)) / 20 if ticks else 0
    keep, last = [], -math.inf
    for t in ticks:
        keep.append(t - last >= gap)
        last = t if keep[-1] else last
    return keep


def figure(rows: list[dict], metric: str, out: Path) -> None:
    """One metric's grid of panels, a panel per parameter swept."""
    anchors = list(dict.fromkeys(r['anchor'] for r in rows))
    params = list(dict.fromkeys(r['param'] for r in rows))
    cols = min(3, len(params))
    nrows = math.ceil(len(params) / cols)
    fig, axes = plt.subplots(nrows, cols, figsize=(4.2 * cols, 3.2 * nrows + 0.8), squeeze=False,
                             constrained_layout=True)
    for ax, param in zip(axes.flat, params):
        panel(ax, rows, param, metric, anchors)
    for ax in list(axes.flat)[len(params):]:
        ax.set_visible(False)
    fig.supylabel(METRICS[metric][0], fontsize=10, color=INK)
    handles = [plt.Line2D([], [], color=c, marker=m, linewidth=2, label=f'{a} ({s})')
               for a, c, m, s in zip(anchors, COLORS, MARKERS,
                                     [next(r['scheme'] for r in rows if r['anchor'] == a) for a in anchors])]
    handles += [plt.Line2D([], [], color=MUTED, marker='o', markersize=11, markerfacecolor='none',
                           linestyle='', label="the anchor's own value"),
                plt.Line2D([], [], color=MUTED, marker='x', linestyle='', label='refused by the compiler')]
    fig.legend(handles=handles, loc='outside upper center', ncol=min(4, len(handles)), fontsize=8,
               frameon=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('--results', type=Path, default=RESULTS,
                    help="the outputs' root (default: %(default)s)")
    ap.add_argument('--metrics', nargs='*', choices=list(METRICS), default=list(DEFAULT),
                    help=f'metrics to plot (default: {" ".join(DEFAULT)})')
    args = ap.parse_args(argv)
    with (args.results / 'sweep' / 'sweep.csv').open() as f:
        rows = list(csv.DictReader(f))
    for metric in args.metrics:
        out = args.results / 'plots' / 'sweep' / f'{metric}.png'
        figure(rows, metric, out)
        print(out)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
