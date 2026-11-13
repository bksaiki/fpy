"""
The evaluations' shared plot style.  Figures are drawn at their printed
size, `TEXTWIDTH` wide (the paper includes them at `\textwidth`), with
`LARGE` titles and axis labels and `SMALL` ticks, legends and notes, so
every figure's text prints at one size.

Hardware families: each vendor has its own fill (NVIDIA solid, AMD
hatched, FMA chains dotted) and a ramp of colors by generation, oldest
lightest, so families stay apart under color blindness and in grayscale.
"""

import math
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[1] / '.matplotlib'))

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator

INK, MUTED, GRID = '#1f1f1e', '#6b6a64', '#e4e3dc'
ACCENT = '#0868ac'
"""The one color of a plot with no groups: the families' darkest blue."""
LARGE = {'fontsize': 7, 'color': INK}
"""Titles and axis labels."""
SMALL = {'fontsize': 6, 'color': MUTED}
"""Ticks, legends and notes."""

TEXTWIDTH = 5.5
"""Inches: the paper's text width."""
ROW = (TEXTWIDTH, 1.3)
"""Inches: a figure of one row of panels, such as two bar charts: each about
3:1, as Fig. 12 first printed."""
GAP = 0.03
"""Between bar charts side by side: a fraction of a panel's width."""

LINE, THIN = 1.0, 0.5
"""Points: data and reference lines; grid lines and axes."""
MARKER = 3.5
"""Points: a line's markers."""
RING = 36
"""Points squared: the ring marking an existing design's point."""

DPI = 200

FAMILIES = {
    'Volta / Turing': ('#f0f9e8', ''), 'Ampere': ('#bae4bc', ''), 'Ada': ('#7bccc4', ''),
    'Hopper': ('#43a2ca', ''), 'Blackwell': ('#0868ac', ''),
    'CDNA1': ('#fee8c8', '////'), 'CDNA2': ('#fdbb84', '////'), 'CDNA3': ('#e34a33', '////'),
    'FMA chain': ('#969696', '....'),
}
"""Hardware families, in legend order, with their color and hatch."""

EDGE = '#1a1a19'
"""Bars' outline and hatch color: it keeps the lightest bars visible."""

LINE_WIDTH = 0.4
"""Points: bars' outlines and hatches."""

matplotlib.rcParams.update({
    'hatch.linewidth': LINE_WIDTH,   # legend swatches ignore a bar's own
    'axes.linewidth': THIN, 'xtick.major.width': THIN, 'ytick.major.width': THIN,
    'xtick.minor.width': THIN, 'ytick.minor.width': THIN, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'xtick.major.pad': 2, 'ytick.major.pad': 2,
    'figure.constrained_layout.w_pad': 0.02, 'figure.constrained_layout.h_pad': 0.02,
    'figure.constrained_layout.wspace': 0.02, 'figure.constrained_layout.hspace': 0.02,
})

OUTLIER = 2.0
"""A speedup this many times the next is cut off (:func:`speedups`)."""
TICKS = 4
"""The most tick intervals on a bar chart's y-axis (:func:`round_axis`)."""


def family(design: str) -> str:
    """*design*'s hardware family, of `FAMILIES`."""
    if 'fma' in design:   # `fp32.fma`, or GSM8K's `fp32 (fma)`
        return 'FMA chain'
    arch = design.split('.')[1]
    if arch.startswith('cdna'):
        return arch.upper()
    return {'volta': 'Volta / Turing', 'turing': 'Volta / Turing',
            'rtx_blackwell': 'Blackwell'}.get(arch, arch.capitalize())


def bar_style(f: str) -> dict[str, object]:
    """Matplotlib patch arguments for a bar of family *f*."""
    color, hatch = FAMILIES[f]
    return {'facecolor': color, 'hatch': hatch, 'edgecolor': EDGE, 'linewidth': LINE_WIDTH}


def geomean(xs: list[float]) -> float:
    return math.exp(sum(map(math.log, xs)) / len(xs))


def tidy(ax: plt.Axes, grid: str = 'y') -> None:
    """*ax*'s ticks, *grid* lines and spines in the shared style."""
    ax.tick_params(colors=SMALL['color'], labelsize=SMALL['fontsize'])
    ax.grid(axis=grid, color=GRID, linewidth=THIN)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)


def legend(at: plt.Figure | plt.Axes, styles: dict[str, dict[str, object]], rows: int = 1) -> None:
    """A swatch per group of *styles* (its bar style), in *rows*: above a
    figure's panels, or in an axes' upper left."""
    where = {'loc': 'outside upper center'} if isinstance(at, plt.Figure) else {'loc': 'upper left'}
    at.legend([plt.Rectangle((0, 0), 1, 1, **s) for s in styles.values()], list(styles), **where,
              ncol=math.ceil(len(styles) / rows), fontsize=SMALL['fontsize'], labelcolor=SMALL['color'],
              frameon=False, handlelength=1.0, handleheight=1.0, handletextpad=0.4, columnspacing=0.8)


def family_legend(at: plt.Figure | plt.Axes, designs: list[str], rows: int = 1) -> None:
    """A swatch per family of *designs*, in `FAMILIES`' order."""
    shown = {family(d) for d in designs}
    legend(at, {f: bar_style(f) for f in FAMILIES if f in shown}, rows)


def round_axis(ax: plt.Axes, top: float) -> float:
    """*ax*'s y-axis from 0 to *top*, ticked at the least round step (1, 2,
    2.5, 3, 4 or 5 × 10^n) that fits at most `TICKS` intervals; *top*."""
    e = 10 ** math.floor(math.log10(top / TICKS))
    step = next(m * e for m in (1, 2, 2.5, 3, 4, 5, 10) if top / (m * e) <= TICKS)
    ax.set_ylim(0, top)
    ax.yaxis.set_major_locator(MultipleLocator(step))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    return top


def speedups(ax: plt.Axes, values: dict[str, float]) -> None:
    """*values*' speedups by design, ascending, colored by family, with
    their geometric mean and 1× dotted.  A speedup `OUTLIER` times the
    next is cut off, its bar broken and labeled."""
    order = sorted(values, key=values.get)
    heights = [values[d] for d in order]
    for i, (d, h) in enumerate(zip(order, heights, strict=True)):
        ax.bar(i, h, 0.9, **bar_style(family(d)))
    mean = geomean(heights)
    ax.axhline(mean, color=INK, linestyle=':', linewidth=LINE)
    ax.text(-0.5, mean, f'geomean {mean:.1f}×', va='bottom', **SMALL | {'color': INK})
    ax.axhline(1.0, color=INK, linestyle='--', linewidth=THIN)
    cut = len(heights) > 1 and heights[-1] > OUTLIER * heights[-2]
    top = round_axis(ax, 1.15 * heights[-2] if cut else 1.05 * heights[-1])   # room for the cut's label
    if cut:   # a slanted gap marks the break
        i, y, d, gap = len(heights) - 1, 0.9 * top, 0.02 * top, 0.008 * top
        ax.fill([i - 0.5, i + 0.5, i + 0.5, i - 0.5], [y - d, y + d, y + d + gap, y - d + gap],
                color='white', zorder=3)
        ax.annotate(f'{heights[-1]:.0f}×', (i, top), xytext=(-4, -2), textcoords='offset points',
                    ha='right', va='top', **SMALL | {'color': INK})
    ax.set_xticks([])
    ax.set_xlim(-0.7, len(order) - 0.3)
    ax.set_xlabel('Designs', **LARGE)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}×'))
    tidy(ax)


def speedup_figure(panels: list[dict[str, float]], path: Path, inside: int | None = None) -> None:
    """A row of :func:`speedups` panels, one per *panels*, untitled (the
    caption names them), as *path* `.png` and `.pdf`; the legend in panel
    *inside*'s empty upper left, if given, else above."""
    fig, axes = plt.subplots(1, len(panels), figsize=ROW, layout='constrained', squeeze=False)
    fig.get_layout_engine().set(wspace=GAP)
    for ax, values in zip(axes[0], panels, strict=True):
        speedups(ax, values)
    axes[0][0].set_ylabel('Speedup', **LARGE)
    designs = [d for v in panels for d in v]
    if inside is None:
        family_legend(fig, designs)
    else:
        family_legend(axes[0][inside], designs, rows=4)
    save(fig, path)


def save(fig: plt.Figure, path: Path, tight: bool = False, dpi: int = DPI) -> None:
    """*fig* as *path* `.png` and `.pdf` (rasterized parts at *dpi*), then
    closed; *tight*, cropped to its content (keys past its edges included)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    crop = {'bbox_inches': 'tight', 'pad_inches': 0.01} if tight else {}
    for ext in ('.png', '.pdf'):
        fig.savefig(path.with_suffix(ext), dpi=dpi, **crop)
    plt.close(fig)
