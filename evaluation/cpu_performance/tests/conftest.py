"""Puts the evaluation's modules on the import path (`run` adds the rest)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
