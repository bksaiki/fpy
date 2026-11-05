"""Puts the scripts, the harness and the models on the import path."""

import sys
from pathlib import Path

_LLM = Path(__file__).resolve().parents[1]
_MMA = _LLM.parents[1] / 'common' / 'mma'
sys.path[:0] = [str(_LLM), str(_MMA), str(_MMA / 'core' / 'tests'),
                str(_LLM.parents[2] / 'examples' / 'mmasim')]

from fixtures import model, registry, tokens
