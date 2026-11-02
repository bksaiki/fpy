"""Puts the evaluation's modules and MMA-Sim's serving harness on the import path."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_MMASIM = _ROOT.parents[1] / 'examples' / 'mmasim'
sys.path[:0] = [str(_ROOT), str(_MMASIM / 'serve'), str(_MMASIM)]
