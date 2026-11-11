"""
Compiles the MMA-Sim designs to C++ with parts of the compiler switched off:
its analyses (`ANALYSES`) crossed with its rewrites (`REWRITES`).

    python evaluation/ablation/run.py           # everything, about 15 min
    python evaluation/ablation/run.py volta     # matching designs

Replaces the results directory (`evaluation/results/ablation/`, or `-o DIR`)
with `<design>.json` per design, `ablation-run.json`, and `report.py`'s
table.  Each compilation runs in a fresh process, which `configure` patches.
Exits 1 if a compilation fails other than by the compiler refusing it.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(_ROOT / 'evaluation' / 'common' / 'mma'), str(_ROOT / 'examples' / 'mmasim')]
sys.path.append(str(_ROOT / 'evaluation'))  # last: `config`, not shadowing this `run`

import config
from compile import DESIGNS, _filename, cpp_compiler, prepare

import fpy2 as fp
from fpy2.analysis import FormatInfer
from fpy2.backend.cpp import CppCompileError
from fpy2.backend.cpp.emitter import CppEmitter
from fpy2.transform import RescaleFixed, RoundElim, TransformDeclined

RESULTS = Path(__file__).resolve().parents[1] / 'results' / 'ablation'

ANALYSES = ('fi', 'fi+db')
"""Format inference alone, or with digit-bound analysis."""

REWRITES = ('none', 'split', 'unfold', 'split+unfold')
"""`split`: the rewrites separating arithmetic from rounding (`Elim`,
`Insert`).  `unfold`: those lowering roundings (`Special`, `Overflow`,
`FloatToFixed`, `Rescale`).  Both is the compiler as the other evaluations
run it."""

CELLS = [(a, r) for a in ANALYSES for r in REWRITES]


def _record(name: str) -> str:
    """The file design *name*'s record is written to."""
    return _filename(name).removesuffix('.cpp') + '.json'


def _foreign(out: Path) -> list[str]:
    """What in *out* a run did not write, and so must not be deleted."""
    if not out.exists():
        return []
    own = {_record(name) for name, _ in DESIGNS}
    return sorted(p.name for p in out.iterdir()
                  if not (p.name in own or p.name.startswith('ablation-')))


def _decline(*_args: Any, **_kwargs: Any) -> Any:
    raise TransformDeclined('switched off')


def configure(analyses: str, rewrites: str) -> fp.CppCompiler:
    """The compiler for the cell (*analyses*, *rewrites*), with what the cell
    leaves out patched off for the rest of this process.

    `Split` has no switch: `cpp_compiler` never runs it, and no design has a
    site for it.
    """
    if analyses == 'fi':
        analyze = FormatInfer.analyze
        FormatInfer.analyze = staticmethod(
            lambda func, **kw: analyze(func, **kw | {'use_digit_bounds': False}))
    if 'split' not in rewrites:
        RoundElim.apply = staticmethod(lambda func: func)
        # `Insert`: the emitter's widening of an exact operation to a native one
        CppEmitter._try_widen = lambda *_args, **_kwargs: None
    if 'unfold' in rewrites:
        return cpp_compiler()
    # `prepare` rescales too, ahead of the compiler's unfolding
    RescaleFixed.apply = staticmethod(_decline)
    return fp.CppCompiler(unfold=fp.CppCompiler.UnfoldMode.NONE, enable_fenv=False)


def compile_once(name: str, analyses: str, rewrites: str) -> dict[str, str]:
    """Design *name* compiled in one cell, in this process: the C++'s hash, or
    the compiler's refusal."""
    compiler = configure(analyses, rewrites)
    try:
        src = compiler.compile_module(prepare(dict(DESIGNS)[name]))
    except CppCompileError as e:
        return {'refused': str(e)}
    return {'cpp_sha256': hashlib.sha256(src.encode()).hexdigest()}


def _run_child(name: str, analyses: str, rewrites: str, timeout: float) -> dict[str, Any]:
    """:func:`compile_once` in a fresh process, killed past *timeout*
    seconds."""
    cmd = [sys.executable, __file__, '--child', name, analyses, rewrites]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return {'timeout': timeout}
    if proc.returncode != 0:
        return {'error': proc.stderr.strip() or f'exit {proc.returncode}'}
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        return {'error': f'no result on stdout: {proc.stdout[-500:]!r}'}


def measure(names: list[str], jobs: int, timeout: float, out: Path) -> list[str]:
    """Every design in *names* in every cell, into *out*: the designs with a
    cell that neither compiled nor was refused."""
    broken = []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {(n, a, r): pool.submit(_run_child, n, a, r, timeout)
                   for n in names for a, r in CELLS}
        width = max(map(len, names))
        for name in names:
            cells = {f'{a}/{r}': futures[(name, a, r)].result() for a, r in CELLS}
            (out / _record(name)).write_text(json.dumps({'design': name, 'cells': cells}, indent=1))
            compiled = [k for k, c in cells.items() if 'cpp_sha256' in c]
            print(f'{name:{width}}  compiles: {", ".join(compiled) or "nowhere"}', flush=True)
            if any('timeout' in c or 'error' in c for c in cells.values()):
                broken.append(name)
    return broken


def main(argv: list[str]) -> int:
    if argv[:1] == ['--child']:
        print(json.dumps(compile_once(*argv[1:4])))
        return 0

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('filter', nargs='*', metavar='SUBSTRING',
                    help='only designs whose name contains one of these')
    ap.add_argument('-o', '--out', metavar='DIR', type=Path, default=RESULTS,
                    help='the results directory, replaced (default %(default)s)')
    ap.add_argument('-j', '--jobs', type=int, default=4,
                    help='compilations at once (default %(default)s)')
    ap.add_argument('-t', '--timeout', type=float, default=config.ABLATION_TIMEOUT,
                    help='seconds one compilation may take (default %(default)s)')
    args = ap.parse_args(argv)

    names = [name for name, _ in DESIGNS
             if not args.filter or any(f in name for f in args.filter)]
    if not names:
        ap.error(f'no design matches {args.filter}')
    if foreign := _foreign(args.out):
        ap.error(f'{args.out} holds files a run did not write: {", ".join(foreign)}')
    shutil.rmtree(args.out, ignore_errors=True)
    args.out.mkdir(parents=True)

    (args.out / 'ablation-run.json').write_text(json.dumps({
        'analyses': ANALYSES, 'rewrites': REWRITES, 'jobs': args.jobs,
        'timeout': args.timeout, 'designs': names,
    }, indent=1))
    broken = measure(names, args.jobs, args.timeout, args.out)
    print(f'{len(names)} designs measured; {len(broken)} with a cell that failed or timed out'
          + (f': {", ".join(broken)}' if broken else ''))
    print('-- report')
    import report
    report.main([str(args.out)])
    return 1 if broken else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
