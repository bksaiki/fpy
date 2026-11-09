"""
FPy's speedup over MMA-Sim, per design: `report.py`'s `gsm8k-speed.json`,
a bar per design, sorted, colored by MMA-Sim's kind of operation.

    python evaluation/gsm8k/plot.py evaluation/results/gsm8k

Writes `gsm8k-designs.pdf` and `.png` beside the records.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(_ROOT / 'evaluation' / 'common' / 'mma'), str(_ROOT / 'examples' / 'mmasim')]

import matplotlib
import mmasim_matmul

matplotlib.use('Agg')
import matplotlib.pyplot as plt

KINDS = {'MMA_T_FDPA': 'T-FDPA (NVIDIA)', 'MMA_ST_FDPA': 'ST-FDPA (MX)', 'MMA_GST_FDPA': 'GST-FDPA (NVFP4, MXFP4)',
         'MMA_TR_FDPA': 'TR-FDPA (CDNA3)', 'MMA_GTR_FDPA': 'GTR-FDPA (CDNA3 FP8)',
         'MMA_FTZ_MUL_ADD': 'FTZ mul-add (CDNA2)', 'MMA_FMA': 'FMA (FP32, FP64)'}
"""Each MMA-Sim operation's label, in legend order."""

COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6']
"""The reference categorical palette's first 7 slots, in order."""

SIZE = (7.0, 2.3)
"""Inches: the paper's other speedup figure's (504 x 165.6 pt)."""

INK, MUTED, GRID = '#1a1a19', '#6b6a64', '#e4e3df'


def kind(design: str) -> str:
    return KINDS[type(mmasim_matmul.operation(design)).__name__]


def plot(designs: list[dict], out: Path) -> None:
    ds = sorted(designs, key=lambda d: d['speedup'])
    color = dict(zip(KINDS.values(), COLORS, strict=True))
    fig, ax = plt.subplots(figsize=SIZE)
    x = range(len(ds))
    ax.bar(x, [d['speedup'] for d in ds], width=0.8, color=[color[kind(d['design'])] for d in ds],
           edgecolor='white', linewidth=0.4, zorder=3)
    geomean = statistics.geometric_mean(d['speedup'] for d in ds)
    ax.axhline(geomean, color=MUTED, linewidth=0.8, linestyle=(0, (3, 2)), zorder=2)
    ax.text(-0.5, geomean, f'geomean {geomean:.0f}×', fontsize=6, color=MUTED, va='bottom')
    ax.set_xticks([])
    ax.set_xlim(-0.6, len(ds) - 0.4)
    ax.margins(y=0)
    ax.set_xlabel('Designs', fontsize=8, color=INK)
    ax.set_ylabel('Speedup', fontsize=8, color=INK)
    ax.yaxis.set_major_formatter(lambda v, _: f'{v:.0f}×')
    ax.tick_params(axis='y', labelsize=7, colors=INK)
    ax.grid(axis='y', color=GRID, linewidth=0.5, zorder=0)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(MUTED)
    present = [k for k in KINDS.values() if any(kind(d['design']) == k for d in ds)]
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=color[k]) for k in present], labels=present,
              fontsize=6, frameon=False, loc='upper left', ncol=2)
    fig.tight_layout(pad=0.2)
    for ext in ('pdf', 'png'):
        fig.savefig(out / f'gsm8k-designs.{ext}', dpi=200)
    plt.close(fig)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('results', type=Path, help="report.py's output directory")
    out = ap.parse_args(argv).results
    plot(json.loads((out / 'gsm8k-speed.json').read_text())['designs'], out)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
