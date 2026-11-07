"""Puts the evaluation's modules and the MMA-Sim designs on the import path;
the designs' operators, built once."""

import sys
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT), str(_ROOT.parent / 'common' / 'mma'), str(_ROOT.parents[1] / 'examples' / 'mmasim')]
sys.path.append(str(_ROOT.parent))  # last: `config`, not shadowing this `run`

BUILD = _ROOT.parent / '.torch_extensions' / 'cpu_performance'
"""Kept between sessions, so that only changed designs recompile."""


@pytest.fixture(scope='session')
def manifest() -> dict[str, Any]:
    import run

    return run.build(BUILD)


@pytest.fixture(scope='session')
def cases(manifest: dict[str, Any]) -> list:
    import cases

    return cases.cases(manifest)
