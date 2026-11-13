"""
Compilation tests for the MMA-Sim designs (`examples/mmasim/designs.py`):
each design to C++, compiled with `cc -c`, and to a Triton kernel.

    python -m tests.infra.backend.mmasim            # every design
    python -m tests.infra.backend.mmasim -k volta   # designs whose name matches
    python -m tests.infra.backend.mmasim --keep     # keep the C++ files
"""

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'examples' / 'mmasim'))

from designs import DESIGNS, Build, compile_matmul, cpp_compiler, prepare

from .cpp import _CPP_CMD, _CPP_OPTIONS

_EXPECTED_REFUSALS = {
    # `e_fdpa` sums exactly: no storage holds the sum (C++ needs 524 bits)
    'amd.cdna1.bf16',
    'amd.cdna1.f16',
}
"""Designs both backends refuse; one that compiles must be removed."""


def _compile_cpp(name: str, build: Build, out: Path) -> None:
    compiler = cpp_compiler()
    src = compiler.compile_module(prepare(build))
    path = out / (re.sub(r'\W+', '_', name) + '.cpp')
    path.write_text('\n'.join(compiler.headers()) + '\n' + compiler.helpers() + '\n' + src + '\n')
    cmd = _CPP_CMD + _CPP_OPTIONS + ['-c', '-o', str(path.with_suffix('.o')), str(path)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def _compile_triton(name: str, build: Build, out: Path) -> None:
    compile_matmul(build, None)


def test_compile_mmasim(select: str | None = None, keep: bool = False) -> list[str]:
    """Every design matching *select* to C++ and to Triton: the failures, as
    `<design> (<backend>): <error>`.  A refusal fails unless the design is in
    `_EXPECTED_REFUSALS`, where compiling fails instead."""
    designs = [(n, b) for n, b in DESIGNS if select is None or re.search(select, n)]
    failures = []
    with tempfile.TemporaryDirectory(prefix='fpy_mmasim_', delete=not keep) as d:
        if keep:
            print(f'Writing C++ to `{d}`')
        for name, build in designs:
            for backend, compile_ in (('C++', _compile_cpp), ('Triton', _compile_triton)):
                expected = name in _EXPECTED_REFUSALS
                try:
                    compile_(name, build, Path(d))
                    status = 'ok'
                    if expected:
                        failures.append(f'{name} ({backend}): compiles; remove it from _EXPECTED_REFUSALS')
                except subprocess.CalledProcessError as e:
                    status = 'cc failed'
                    failures.append(f'{name} ({backend}): {e.stderr.strip()}')
                except Exception as e:  # noqa: BLE001 -- a refusal is a result
                    status = type(e).__name__ + (' (expected)' if expected else '')
                    if not expected:
                        failures.append(f'{name} ({backend}): {type(e).__name__}: {e}')
                print(f'{name:28} {backend:6} {status}', flush=True)
    print(f'{len(designs)} designs, {len(failures)} failures')
    for f in failures:
        print(f'FAILED {f}')
    return failures


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compile the MMA-Sim designs to C++ and Triton')
    parser.add_argument('-k', dest='select', metavar='PATTERN', default=None,
                        help='only designs whose name matches PATTERN (a regex)')
    parser.add_argument('--keep', action='store_true', help='keep the C++ files')
    args = parser.parse_args()
    sys.exit(1 if test_compile_mmasim(args.select, args.keep) else 0)
