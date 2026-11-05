"""
Writing results: whole or not at all, and with what they came from.
"""

import csv
import hashlib
import json
import os
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any


def write_json(path: Path, obj: Any) -> None:
    """*obj* as JSON at *path*, written atomically."""
    replace(path, lambda f: json.dump(obj, f, indent=2))


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """*rows* as CSV at *path*, written atomically; columns are every key,
    first seen first."""
    fields = list(dict.fromkeys(k for r in rows for k in r))

    def write(f: Any) -> None:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(rows)
    replace(path, write)


def replace(path: Path, write: Callable[[Any], None], binary: bool = False) -> None:
    """*path* replaced atomically by what *write* writes to an open file: a
    run that dies midway leaves the old file or none, never part of one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.tmp')
    with (tmp.open('wb') if binary else tmp.open('w', newline='')) as f:
        write(f)
    os.replace(tmp, path)


def sources(*paths: Path) -> str:
    """A digest of *paths*' contents, each a file or a directory's `.py` files."""
    files = sorted(f for p in paths for f in (p.rglob('*.py') if p.is_dir() else [p]))
    digest = hashlib.sha256()
    for f in files:
        digest.update(f.read_bytes())
    return digest.hexdigest()


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
