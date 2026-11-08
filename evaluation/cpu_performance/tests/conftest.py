"""Puts the evaluation's modules on the import path (`run` adds the rest);
the designs' operators, built once."""

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(scope='session')
def manifest() -> dict[str, Any]:
    import run

    return run.build()


@pytest.fixture(scope='session')
def cases(manifest: dict[str, Any]) -> list:
    import cases

    return cases.cases(manifest)
