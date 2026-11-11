"""
FPy's speedup over MMA-Sim on the CPU (left) and on the GPU (right), as
the compilation figure's: from the `cpu_performance` and `gsm8k`
evaluations' results.  `run.py` draws it once both exist.

    python evaluation/performance/plots.py --results evaluation/results   # -> <results>/performance/perf-speedup.{png,pdf}
"""

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

HERE = Path(__file__).resolve().parents[1]
sys.path.append(str(HERE))

from common.plot import speedup_figure


def _module(path: Path) -> ModuleType:
    """*path* imported under a name of its own: both evaluations' modules
    are `report.py` and `plot.py`."""
    spec = importlib.util.spec_from_file_location(f'{path.parent.name}_{path.stem}', path)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('--results', type=Path, required=True, help="the evaluations' results root")
    root = ap.parse_args(argv).results
    cpu = _module(HERE / 'cpu_performance' / 'report.py')
    gpu = _module(HERE / 'gsm8k' / 'plot.py')
    speedup_figure([cpu.headline(cpu.load(root / 'cpu_performance')), gpu.speedups(root / 'gsm8k')],
                   root / 'performance' / 'perf-speedup', inside=0)   # the CPU's bars leave its top left empty
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
