"""Puts the evaluation's modules and the MMA-Sim designs on the import path."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT), str(_ROOT.parent / 'common' / 'mma'), str(_ROOT.parents[1] / 'examples' / 'mmasim')]
sys.path.append(str(_ROOT.parent))  # last: `config`, not shadowing this `run`
