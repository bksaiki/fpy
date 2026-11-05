"""Puts the harness and the models on the import path."""

import sys
from pathlib import Path

_MMA = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(_MMA), str(_MMA.parents[2] / 'examples' / 'mmasim')]

from fixtures import model, registry, tokens
