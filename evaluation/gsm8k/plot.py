"""
FPy's speedup over MMA-Sim on the GPU, per design, from `report.py`'s
`gsm8k-speed.json`, in the shared speedup figure (`common/plot.py`).

    python evaluation/gsm8k/plot.py evaluation/results/gsm8k

Writes `gsm8k-speedup.pdf` and `.png` beside the records.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from common.plot import speedup_figure


def speedups(out: Path) -> dict[str, float]:
    """Each design's speedup in *out*'s `gsm8k-speed.json`."""
    return {d['design']: d['speedup'] for d in json.loads((out / 'gsm8k-speed.json').read_text())['designs']}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('results', type=Path, help="report.py's output directory")
    out = ap.parse_args(argv).results
    speedup_figure([speedups(out)], out / 'gsm8k-speedup')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
