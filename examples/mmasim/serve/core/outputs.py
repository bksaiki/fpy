"""
Writing results: whole or not at all, and with what they came from.
"""

import csv
import json
import os
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def write_json(path: Path, obj: Any) -> None:
    """*obj* as JSON at *path*, replacing it at once: a run that dies midway
    leaves the old file or none, never part of one."""
    _replace(path, lambda f: json.dump(obj, f, indent=2))


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """*rows* as CSV at *path* (columns: every key, first seen first), as
    :func:`write_json` replaces a file."""
    fields = list(dict.fromkeys(k for r in rows for k in r))

    def write(f: Any) -> None:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(rows)
    _replace(path, write)


def _replace(path: Path, write: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.tmp')
    with tmp.open('w', newline='') as f:
        write(f)
    os.replace(tmp, path)


def commit() -> dict[str, Any]:
    """The checkout's commit, and whether tracked files differ from it
    (`None`s outside a git checkout)."""
    def git(*args: str) -> str | None:
        try:
            done = subprocess.run(['git', *args], capture_output=True, text=True, check=True,
                                  cwd=Path(__file__).parent)
        except (OSError, subprocess.CalledProcessError):
            return None
        return done.stdout.strip()

    status = git('status', '--porcelain', '--untracked-files=no')
    return {'commit': git('rev-parse', 'HEAD'), 'dirty': None if status is None else bool(status)}
