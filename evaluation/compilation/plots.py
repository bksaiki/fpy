"""
The compilation-time figures, from `compile_times.py`'s table: a panel per
backend, a bar per design sorted by its time; coloured by each design's GPU
architecture, by its input format, or by the MMA-Sim arithmetic model it
transcribes (as the performance evaluation's figure), or in one colour.

    python evaluation/compilation/plots.py    # -> evaluation/results/compilation/plots/{arch,format,models,solid}.{png,pdf}
"""

import argparse
import csv
import statistics
import sys
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'evaluation' / 'results'
sys.path.insert(0, str(ROOT / 'evaluation' / 'dse' / 'llm'))

# `plot_grid.py`'s look: its grid colour and text styles
from plot_grid import GRID, INK, LARGE, SMALL

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


def architecture(design: str) -> str:
    """*design*'s GPU architecture, Volta's and Turing's together and RTX
    Blackwell's with Blackwell's."""
    arch, _ = _parts(design)
    return {'volta': 'Volta/Turing', 'turing': 'Volta/Turing', 'ampere': 'Ampere', 'ada': 'Ada',
            'hopper': 'Hopper', 'blackwell': 'Blackwell', 'rtx_blackwell': 'Blackwell', 'cdna1': 'CDNA1',
            'cdna2': 'CDNA2', 'cdna3': 'CDNA3', 'fma': 'FMA loop'}[arch]


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


# each colouring: its grouping, and each group's colour in legend order. The
# architectures keep the LLM and BLAS figures' colours; the formats and models
# the performance figure's.
COLORINGS: dict[str, tuple[Callable[[str], str], dict[str, str]]] = {
    'arch': (architecture, {'Volta/Turing': '#6250d6', 'Ampere': '#008300', 'Ada': '#1baf7a',
                            'Hopper': '#eb6834', 'Blackwell': '#2a78d6', 'CDNA1': '#e34948',
                            'CDNA2': '#e87ba4', 'CDNA3': '#eda100', 'FMA loop': '#6b6a64'}),
    'format': (input_format, {'FP64': '#2a78d6', 'FP32': '#eb6834', 'TF32': '#1baf7a', 'BF16': '#eda100',
                              'FP16': '#e87ba4', 'FP8': '#008300', 'FP6': '#0f8fa8', 'FP4': '#6250d6',
                              'MX / NVFP4': '#e34948'}),
    'models': (arithmetic, {'T-FDPA': '#2a78d6', 'ST-FDPA': '#eb6834', 'GST-FDPA': '#1baf7a',
                            'TR-FDPA': '#eda100', 'GTR-FDPA': '#e87ba4', 'E-FDPA': '#e34948',
                            'FTZ mul-add': '#008300', 'FMA': '#6250d6'}),
}


def load(results: Path) -> dict[str, list[dict]]:
    """Each backend's compiled designs, with their seconds."""
    by: dict[str, list[dict]] = {b: [] for b in BACKENDS}
    with (results / 'compilation' / 'compile-times.csv').open() as f:
        for r in csv.DictReader(f):
            if r['status'] == 'ok':
                by[r['backend']].append({'design': r['design'], 'seconds': float(r['seconds'])})
    return by


def figure(by: dict[str, list[dict]], coloring: str | None) -> plt.Figure:
    """A panel per backend, a bar per compiled design sorted by its time,
    coloured by *coloring* (one colour if `None`), and the median a dotted
    line."""
    group, palette = COLORINGS[coloring] if coloring else (None, {})
    fig, axes = plt.subplots(1, len(BACKENDS), figsize=(13, 4.2), layout='constrained', sharey=True)
    for (b, label), ax in zip(BACKENDS.items(), axes):
        mine = sorted(by[b], key=lambda d: d['seconds'])
        colors = [palette[group(d['design'])] if group else SOLID for d in mine]
        ax.bar(range(len(mine)), [d['seconds'] for d in mine], width=0.9, color=colors)
        median = statistics.median(d['seconds'] for d in mine)
        ax.axhline(median, color=INK, linestyle=':', linewidth=2)
        ax.text(-0.5, median, f'median {median:.1f} s', va='bottom', **SMALL | {'color': INK})
        ax.set_title(f'{label}, {len(mine)} designs', loc='left', **LARGE)
        ax.set_xticks([])
        ax.set_xlim(-0.7, len(mine) - 0.3)
        ax.set_xlabel('Designs', **LARGE)
        ax.tick_params(colors=SMALL['color'], labelsize=SMALL['fontsize'])
        ax.grid(axis='y', color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel('Compilation Time (s)', **LARGE)
    if group:
        shown = {group(d['design']) for ds in by.values() for d in ds}
        handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=g) for g, c in palette.items() if g in shown]
        axes[0].legend(handles=handles, loc='upper left', ncol=2, fontsize=SMALL['fontsize'],
                       labelcolor=SMALL['color'], frameon=False, handlelength=1.2, handleheight=1.0)
    return fig


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=str(__doc__).splitlines()[1])
    ap.add_argument('--results', type=Path, default=RESULTS,
                    help="the outputs' root, as `compile_times.py`'s (default: %(default)s)")
    args = ap.parse_args(argv)
    by = load(args.results)
    out = args.results / 'compilation' / 'plots'
    out.mkdir(parents=True, exist_ok=True)
    for name in (*COLORINGS, 'solid'):
        fig = figure(by, None if name == 'solid' else name)
        for ext in ('png', 'pdf'):
            fig.savefig(out / f'{name}.{ext}', dpi=200)
    for b, label in BACKENDS.items():
        times = [d['seconds'] for d in by[b]]
        print(f'{label:7} {len(times)} designs, median {statistics.median(times):.1f} s, '
              f'max {max(times):.1f} s, sum {sum(times):.0f} s')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
