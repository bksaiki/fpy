"""Puts the evaluation's modules on the import path (`run` adds the rest)."""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_HERE), str(_HERE.parent / 'common' / 'mma')]
