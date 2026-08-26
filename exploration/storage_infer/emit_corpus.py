"""Emit C++ for every compiling corpus function, for before/after comparison.

Storage inference is being replaced by a backend-independent analysis
(``docs/todos/storage-inference.md``).  The obligation is behavioural
equivalence, and the cheapest check is that the emitted code does not move:

    python -m exploration.storage_infer.emit_corpus /tmp/base
    ... change something ...
    python -m exploration.storage_infer.emit_corpus /tmp/new
    diff -r /tmp/base /tmp/new

A refusal is part of the behaviour, so a function that does not compile is
recorded with its error rather than skipped -- a change that makes a refusal
*disappear* is as much a difference as one that changes emitted code.
"""

import importlib
import sys
from pathlib import Path

import fpy2 as fp

from fpy2.backend.cpp.compiler import CppCompiler
from tests.infra.backend.cpp import _inst_type, _library_ignore
from tests.infra.examples import all_example_tests, all_unit_tests


def corpus():
    yield from all_unit_tests()
    yield from all_example_tests()
    for name in ('core', 'eft', 'vector', 'matrix'):
        mod = importlib.import_module(f'fpy2.libraries.{name}')
        for f in mod.__dict__.values():
            if isinstance(f, fp.Function) and f.name not in _library_ignore:
                yield f


def main(out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob('*.txt'):
        stale.unlink()

    emitted = refused = 0
    for i, f in enumerate(corpus()):
        # index-prefixed: two corpus entries may share a name
        path = out / f'{i:04d}_{f.name}.txt'
        try:
            ty = fp.analysis.TypeInfer.check(f.ast)
            args = [_inst_type(t) for t in ty.arg_types]
            path.write_text(
                CppCompiler().compile(f, ctx=fp.FP64, arg_types=args))
            emitted += 1
        except Exception as e:                        # noqa: BLE001
            path.write_text(f'REFUSED {type(e).__name__}: {e}\n')
            refused += 1
    print(f'{emitted} emitted, {refused} refused -> {out}')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'exploration/storage_infer/out')
