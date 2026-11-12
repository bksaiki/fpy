"""
The compilation-time figures, from `compile_times.py`'s table: a panel per
backend, a bar per design sorted by its time; coloured by each design's GPU
architecture, by its input format, or by the MMA-Sim arithmetic model it
transcribes (as the performance evaluation's figure), or in one colour; and
by architecture, each bar's digit-bound solver time solid and the rest
faded.  Beside them, `compile-times-paper.json`: each backend's medians, and
each arithmetic model's range.

    python evaluation/compilation/plots.py    # -> evaluation/results/compilation/plots/{arch,format,models,solid,solver}.{png,pdf}
"""

import argparse
import csv
import json
import statistics
import sys
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'evaluation' / 'results'
sys.path.append(str(ROOT / 'evaluation'))

from common.plot import (
    FAMILIES,
    GAP,
    INK,
    LARGE,
    LINE,
    ROW,
    SMALL,
    bar_style,
    family,
    legend,
    round_axis,
    save,
    tidy,
)

BACKENDS = {'cpp': 'C++', 'triton': 'Triton'}
# the performance figure's one colour
SOLID = '#eb6834'


def _parts(design: str) -> tuple[str, list[str]]:
    """*design*'s architecture and the rest of its name: `nv.hopper.bf16.f32`
    -> `hopper`, `[bf16, f32]`; `fp32 (fma)` -> `fma`, `[fp32]`."""
    if design.endswith('(fma)'):
        return 'fma', [design.split()[0]]
    _, arch, *rest = design.split('.')
    return arch, rest


def input_format(design: str) -> str:
    """*design*'s input format, as the performance evaluation's figure groups
    them: block-scaled formats together, an FP8-FP4 mix (older results) as FP4."""
    arch, rest = _parts(design)
    if arch == 'fma':
        return rest[0].upper()
    if rest[0] in ('mx', 'mxfp4', 'nvfp4'):
        return 'MX / NVFP4'
    if 'e2m1' in rest:
        return 'FP4'
    if {'e3m2', 'e2m3'} & set(rest):
        return 'FP6'
    if {'e4m3', 'e5m2', 'fp8', 'bf8'} & set(rest):
        return 'FP8'
    return {'tf32': 'TF32', 'bf16': 'BF16', 'bf16_1k': 'BF16', 'f16': 'FP16'}[rest[0]]


def arithmetic(design: str) -> str:
    """The MMA-Sim arithmetic model *design* transcribes."""
    arch, rest = _parts(design)
    name = '.'.join(rest)
    if arch == 'fma':
        return 'FMA'
    if arch == 'cdna1':
        return 'E-FDPA'
    if arch == 'cdna2':
        return 'FTZ mul-add'
    if arch == 'cdna3':
        return 'GTR-FDPA' if '8' in name else 'TR-FDPA'
    if name in ('nvfp4', 'mxfp4'):
        return 'GST-FDPA'
    if name.startswith('mx.'):
        return 'ST-FDPA'
    return 'T-FDPA'


def _solid(colors: dict[str, str]) -> dict[str, dict[str, object]]:
    return {g: {'color': c} for g, c in colors.items()}


# each colouring: its grouping, and each group's bar style in legend order;
# the architectures the hardware families' (`common/plot.py`)
COLORINGS: dict[str, tuple[Callable[[str], str], dict[str, dict[str, object]]]] = {
    'arch': (family, {f: bar_style(f) for f in FAMILIES}),
    'format': (input_format, _solid({'FP64': '#2a78d6', 'FP32': '#eb6834', 'TF32': '#1baf7a', 'BF16': '#eda100',
                              'FP16': '#e87ba4', 'FP8': '#008300', 'FP6': '#0f8fa8', 'FP4': '#6250d6',
                              'MX / NVFP4': '#e34948'})),
    'models': (arithmetic, _solid({'T-FDPA': '#2a78d6', 'ST-FDPA': '#eb6834', 'GST-FDPA': '#1baf7a',
                            'TR-FDPA': '#eda100', 'GTR-FDPA': '#e87ba4', 'E-FDPA': '#e34948',
                            'FTZ mul-add': '#008300', 'FMA': '#6250d6'})),
}


FADED = 0.35
"""The opacity of a bar's time outside the solver, in the solver figure."""
KEY = ('Hopper', 'Ada', 'AMD CDNA3')
"""The families striping the solver figure's key: any family's style."""

STATS = ('seconds', 'constraints', 'queries', 'solver_seconds')
"""Each compiled design's numbers: an older table has only its seconds."""


def load(results: Path) -> dict[str, list[dict]]:
    """Each backend's compiled designs, with their numbers."""
    by: dict[str, list[dict]] = {b: [] for b in BACKENDS}
    with (results / 'compilation' / 'compile-times.csv').open() as f:
        for r in csv.DictReader(f):
            if r['status'] == 'ok':
                by[r['backend']].append({'design': r['design']} | {
                    k: float(r[k]) for k in STATS if r.get(k)})
    return by


def summary(by: dict[str, list[dict]]) -> dict:
    """Each backend's median over its compiled designs, and each arithmetic
    model's least and greatest."""
    out = {}
    for b, designs in by.items():
        stats = [k for k in STATS if all(k in d for d in designs)]
        models: dict[str, list[dict]] = {}
        for d in designs:
            models.setdefault(arithmetic(d['design']), []).append(d)
        out[b] = {
            'designs': len(designs),
            'median': {k: statistics.median(d[k] for d in designs) for k in stats} | (
                {'solver_share': statistics.median(d['solver_seconds'] / d['seconds'] for d in designs)}
                if 'solver_seconds' in stats else {}),
            'models': {m: {'designs': len(ds),
                           'range': {k: [min(d[k] for d in ds), max(d[k] for d in ds)] for k in stats}}
                       for m, ds in sorted(models.items())},
        }
    return out


def figure(by: dict[str, list[dict]], coloring: str | None) -> plt.Figure:
    """A panel per backend, a bar per compiled design sorted by its time,
    coloured by *coloring* (one colour if `None`; by architecture, the
    solver's time solid and the rest faded, if `'solver'`), and the median a
    dotted line."""
    split = coloring == 'solver'
    group, palette = COLORINGS['arch' if split else coloring] if coloring else (None, {})
    fig, axes = plt.subplots(1, len(BACKENDS), figsize=ROW, layout='constrained')
    fig.get_layout_engine().set(wspace=GAP)
    for b, ax in zip(BACKENDS, axes):
        mine = sorted(by[b], key=lambda d: d['seconds'])
        for i, d in enumerate(mine):
            style = palette[group(d['design'])] if group else {'color': SOLID}
            if split:
                ax.bar(i, d['solver_seconds'], width=0.9, **style)
                ax.bar(i, d['seconds'] - d['solver_seconds'], bottom=d['solver_seconds'], width=0.9,
                       **style, alpha=FADED)
            else:
                ax.bar(i, d['seconds'], width=0.9, **style)
        median = statistics.median(d['seconds'] for d in mine)
        ax.axhline(median, color=INK, linestyle=':', linewidth=LINE)
        ax.text(-0.5, median, f'median {median:.1f} s', va='bottom', **SMALL | {'color': INK})
        ax.set_xticks([])
        ax.set_xlim(-0.7, len(mine) - 0.3)
        ax.set_xlabel('Designs', **LARGE)
        round_axis(ax, 1.05 * mine[-1]['seconds'])
        tidy(ax)
    axes[0].set_ylabel('Compilation Time (s)', **LARGE)
    if group:   # in the first panel's empty upper left
        shown = {group(d['design']) for ds in by.values() for d in ds}
        legend(axes[0], {g: s for g, s in palette.items() if g in shown}, rows=4)
    if split:   # in the last panel's empty upper left
        stripes = [tuple(Rectangle((0, 0), 1, 1, **palette[f], alpha=a) for f in KEY) for a in (1, FADED)]
        axes[-1].legend(stripes, ['Solver', 'Rest of compilation'], loc='upper left',
                        handler_map={tuple: HandlerTuple(ndivide=None, pad=0)}, fontsize=SMALL['fontsize'],
                        labelcolor=SMALL['color'], frameon=False, handlelength=1.6, handleheight=1.0,
                        handletextpad=0.4)
    return fig


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=str(__doc__).splitlines()[1])
    ap.add_argument('--results', type=Path, default=RESULTS,
                    help="the outputs' root, as `compile_times.py`'s (default: %(default)s)")
    args = ap.parse_args(argv)
    by = load(args.results)
    out = args.results / 'compilation' / 'plots'
    out.mkdir(parents=True, exist_ok=True)
    timed = all('solver_seconds' in d for ds in by.values() for d in ds)   # not in older tables
    for name in (*COLORINGS, 'solid', *(['solver'] if timed else [])):
        save(figure(by, None if name == 'solid' else name), out / name)

    def num(k: str, v: float) -> str:
        return f'{v:.1f}' if k.endswith('seconds') else f'{v:.0%}' if k.endswith('share') else f'{v:,.0f}'

    paper = summary(by)
    (args.results / 'compilation' / 'compile-times-paper.json').write_text(json.dumps(paper, indent=1))
    for b, label in BACKENDS.items():
        s = paper[b]
        print(f'{label}: {s["designs"]} designs; median ' + ', '.join(
            f'{num(k, v)} {k.replace("_", " ")}' for k, v in s['median'].items()))
        for m, r in s['models'].items():
            print(f'  {m:12} {r["designs"]:2} designs  ' + '  '.join(
                f'{k.replace("_", " ")} {num(k, lo)}-{num(k, hi)}' for k, (lo, hi) in r['range'].items()))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
