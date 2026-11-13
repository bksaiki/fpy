"""
Compiles every MMA-Sim design to C++, reporting where each one stops.

A roadmap tracker: the point is to see *which* refusal each design hits and
how the count moves.  With `-r`, each design that compiles is also run
against the interpreter.

    python evaluation/common/mma/compile.py           # one line per design
    python evaluation/common/mma/compile.py -v        # full error text
    python evaluation/common/mma/compile.py -e cdna2  # print the C++ of a design
    python evaluation/common/mma/compile.py -o out/   # write each one to out/
    python evaluation/common/mma/compile.py -j 8      # in 8 processes
    python evaluation/common/mma/compile.py -r 256    # also run 256 draws and
                                                # compare with the interpreter
"""

import argparse
import json
import math
import random
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from functools import cache, partial
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'examples' / 'mmasim'))

from designs import (  # noqa: F401 -- re-exported for the evaluations
    _CDNA3_FP8_PAIRS,
    _F8F6F4_PAIRS,
    _FP8_PAIRS,
    DESIGNS,
    Build,
    _length,
    cpp_compiler,
    prepare,
)

import fpy2 as fp
from fpy2.backend.cpp.types import CppList
from fpy2.backend.cpp.utils import CPP_HEADERS, CPP_HELPERS
from fpy2.number.context import SizedFormat
from fpy2.types import Type

_L = fp.types.ListType
_R = fp.types.RealType

_HARD_EVERY = 4
"""Every this many `--run` draws, each element is a hard case with
probability :data:`_HARD`."""

_HARD = 0.25

_CXX = ['c++', '-std=c++17', '-O0', '-w']

_DRIVER = r'''
#include <cstdio>
#include <cstdlib>
#include <type_traits>

// Reads a count, then that many argument lists; prints one result per line.
static double next_value() {
    double x;
    if (std::scanf("%lf", &x) != 1) std::exit(2);
    return x;
}
template <class T> struct In { static T get() { return static_cast<T>(next_value()); } };
template <class T, std::size_t N> struct In<std::array<T, N>> {
    static std::array<T, N> get() {
        std::array<T, N> xs;
        for (auto& x : xs) x = In<T>::get();
        return xs;
    }
};
template <class R, class... P> void drive(R (*fn)(P...)) {
    int n;
    if (std::scanf("%d", &n) != 1) std::exit(2);
    for (int i = 0; i < n; ++i) {
        // a braced list reads the arguments in order
        std::tuple<typename std::decay<P>::type...> args{In<typename std::decay<P>::type>::get()...};
        std::printf("%a\n", static_cast<double>(std::apply(fn, args)));
        std::fflush(stdout);
    }
}
int main() { drive(&@ENTRY@); }
'''


def compile_design(build: Build) -> tuple[str, dict]:
    """The C++ for one design (:func:`prepare`), and its entry's signature;
    raises whatever refused it.

    The signature gives the entry's name, and the C++ element type of each
    parameter and the result, with a `size` for a fixed-size array.
    """
    return compile_module(prepare(build))


def compile_module(mod: fp.Module) -> tuple[str, dict]:
    """The C++ for prepared module *mod*, and its entry's signature, as
    :func:`compile_design` gives them."""
    compiler = cpp_compiler()
    src = compiler.compile_module(mod)
    entry = mod.public()[0]
    params, result = compiler.signature(entry, module=mod)
    sig = {'entry': entry.name, 'params': [_cpp_type(p) for p in params],
           'result': _cpp_type(result)}
    return src, sig


def _cpp_type(t) -> dict:
    """*t* as `{type}` for a scalar, or `{type, size}` for a list of them."""
    if isinstance(t, CppList):
        return {'type': t.elt.format(), 'size': t.size}
    return {'type': t.format()}

def _namespace(name: str) -> str:
    """Return a unique C++ namespace for a design name."""
    chars = [c if c.isalnum() else "_" for c in name]
    words = filter(None, "".join(chars).split("_"))
    return "model_" + "_".join(words)

def in_processes(fn: Callable, items: Iterable, jobs: int) -> Iterator:
    """*fn* of each of *items*, in order: in *jobs* processes if more than one.
    *fn* and its results must pickle."""
    if jobs <= 1:
        yield from map(fn, items)
        return
    pool = ProcessPoolExecutor(jobs)
    try:
        yield from pool.map(fn, items)
    finally:
        pool.shutdown(cancel_futures=True)


def _fmt(t: Type) -> SizedFormat:
    """The format of *t*'s values, or of its elements'."""
    elt = t.elt if isinstance(t, _L) else t
    if not (isinstance(elt, _R) and isinstance(elt.fmt, SizedFormat)):
        raise TypeError(f'expected a real type of sized format, got {t}')
    return elt.fmt


@cache
def _hard_cases(fmt: SizedFormat) -> list[float]:
    """The values of *fmt* a random draw misses: the zeros, the smallest and
    largest magnitudes, the least normal, and the specials it has."""
    hi = fmt.to_ordinal(fmt.maxval())
    mags = [0.0, float(fmt.from_ordinal(1)), float(fmt.from_ordinal(hi)), math.inf, math.nan]
    if (normal := getattr(fmt, 'min_normal', None)) is not None:
        mags.append(float(normal()))
    out: dict[str, float] = {}
    for v in mags:
        for x in (v, -v):
            if fmt.representable_in(fp.Float.from_float(x)):
                out.setdefault(repr(x), x)
    return list(out.values())


def _sample(fmt: SizedFormat, rng: random.Random, hard: float = 0.0) -> float:
    """A value of *fmt*: a hard case with probability *hard*, else half over
    its whole range, half near one."""
    if hard and rng.random() < hard:
        return rng.choice(_hard_cases(fmt))
    hi = fmt.to_ordinal(fmt.maxval())
    try:
        fmt.from_ordinal(-1)
        lo = -hi
    except Exception:  # noqa: BLE001 -- an unsigned format
        lo = fmt.to_ordinal(fmt.minval())
    if rng.random() < 0.5:
        o = rng.randint(lo, hi)
    else:
        one = fp.Float.from_float(1.0)
        mid = fmt.to_ordinal(one) if fmt.representable_in(one) else hi // 2
        w = max(hi // 8, 1)
        o = rng.randint(max(mid - w, 0), min(mid + w, hi))
        o = -o if lo < 0 and rng.random() < 0.5 else o
    return float(fmt.from_ordinal(o))


def _vector(t: Type, rng: random.Random, hard: float = 0.0) -> list[float]:
    """A value of each of list type *t*'s elements."""
    return [_sample(_fmt(t), rng, hard) for _ in range(_length(t))]


def _row(t: Type, rng: random.Random, hard: float = 0.0) -> float | list[float]:
    """A value of *t*, or of each of its elements."""
    return _vector(t, rng, hard) if isinstance(t, _L) else _sample(_fmt(t), rng, hard)


def _same(got: float, want: float) -> bool:
    """Bit for bit: the same value and sign, or both NaN."""
    return (got == want and math.copysign(1, got) == math.copysign(1, want)
            or math.isnan(got) and math.isnan(want))


def run_design(name: str, src: str, build: Build, draws: int, seed: int) -> int:
    """How many of *draws* random inputs the C++ *src* of design *build* gets
    bit for bit.  Every :data:`_HARD_EVERY`-th draw is heavy in hard cases.
    A run that aborts -- a failed assert -- misses every draw after it."""
    func, arg_types = build()
    rng = random.Random(seed)
    inputs = [
        [_row(t, rng, _HARD if (i + 1) % _HARD_EVERY == 0 else 0.0) for t in arg_types]
        for i in range(draws)
    ]
    lines = [str(draws)] + [
        ' '.join(float.hex(x) for a in args for x in (a if isinstance(a, list) else [a]))
        for args in inputs
    ]
    with tempfile.TemporaryDirectory() as tmp:
        cpp, exe = Path(tmp) / 'run.cpp', Path(tmp) / 'run'
        entry = f'fpy_models::{_namespace(name)}::{func.name}'
        cpp.write_text(_translation_unit(name, src) + _DRIVER.replace('@ENTRY@', entry))
        built = subprocess.run([*_CXX, '-o', str(exe), str(cpp)],
                               capture_output=True, text=True, check=False)
        if built.returncode:
            raise RuntimeError(f'the driver does not build:\n{built.stderr}')
        # not checked: an abort still leaves the draws before it
        out = subprocess.run([str(exe)], input='\n'.join(lines), capture_output=True,
                             text=True, timeout=600, check=False).stdout
    got = [float.fromhex(s) for s in out.split()]
    want = [float(func(*args)) for args in inputs]
    return sum(_same(g, w) for g, w in zip(got, want))


def _compile_named(
    name: str, draws: int = 0, seed: int = 0,
) -> tuple[str | None, dict | None, str, str, int | None]:
    """The C++ for design *name*, its signature, and how many of *draws* it
    agrees on, or `None`s and the refusal's type and text."""
    build = dict(DESIGNS)[name]
    try:
        src, sig = compile_design(build)
        return src, sig, '', '', run_design(name, src, build, draws, seed) if draws else None
    except Exception as ex:  # noqa: BLE001 -- any refusal is a result
        return None, None, type(ex).__name__, str(ex), None


def _translation_unit(name: str, src: str) -> str:
    """Add shared headers and isolate one generated design's symbols."""
    body = CPP_HELPERS + '\n' + src
    return (
        '\n'.join(CPP_HEADERS)
        + f'\n\nnamespace fpy_models::{_namespace(name)} {{\n'
        + body
        + '\n}\n'
    )


def _filename(name: str) -> str:
    """*name* as a filename: `fp64 (fma)` -> `fp64_fma.cpp`."""
    keep = [c if (c.isalnum() or c in '._') else '_' for c in name]
    return ''.join(keep).strip('_').replace('__', '_') + '.cpp'


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('filter', nargs='*', metavar='SUBSTRING',
                    help='only designs whose name contains one of these')
    ap.add_argument('-v', '--verbose', action='store_true',
                    help='full error text for a design that does not compile')
    ap.add_argument('-j', '--jobs', type=int, default=1,
                    help='compile in this many processes (default 1)')
    ap.add_argument('-r', '--run', metavar='DRAWS', type=int, default=0,
                    help='run DRAWS random inputs and compare every output, '
                         'bit for bit, with the interpreter')
    ap.add_argument('-s', '--seed', type=int, default=0,
                    help='seed for the inputs --run draws (default 0)')
    # where the C++ goes: stdout or files, not both
    dest = ap.add_mutually_exclusive_group()
    dest.add_argument('-e', '--emit', action='store_true',
                      help='print the C++ of each design that compiles')
    dest.add_argument('-o', '--out', metavar='DIR', type=Path,
                      help='write each compiled design to DIR/<name>.cpp, and '
                           'their signatures to DIR/manifest.json')
    args = ap.parse_args(argv)

    names = [
        name for name, _ in DESIGNS
        if not args.filter or any(f in name for f in args.filter)
    ]
    if not names:
        ap.error(f'no design matches {args.filter}')
    if args.run and shutil.which(_CXX[0]) is None:
        ap.error(f'--run needs a C++ compiler: no `{_CXX[0]}`')
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)

    width = max(len(name) for name in names)
    ok = agree = 0
    manifest: dict[str, Any] = {'models': [], 'refused': {}}
    results = in_processes(partial(_compile_named, draws=args.run, seed=args.seed),
                           names, args.jobs)
    for name, (src, sig, kind, why, ran) in zip(names, results):
        if src is None:
            detail = why if args.verbose else why.split('\n')[0][:110]
            print(f'{name:{width}}  {kind}: {detail}')
            manifest['refused'][name] = f'{kind}: {why.splitlines()[0] if why else ""}'
            continue
        manifest['models'].append({'name': name, 'file': _filename(name),
                                   'namespace': f'fpy_models::{_namespace(name)}', **sig})
        ok += 1
        note = ''
        if ran is not None:
            agree += ran == args.run
            note = f'  {ran}/{args.run} agree'
        if args.out is not None:
            path = args.out / _filename(name)
            path.write_text(_translation_unit(name, src))
            note += f'  -> {path}'
        print(f'{name:{width}}  OK{note}')
        if args.emit:
            print(f'\n// ==== {name} ====\n{_translation_unit(name, src)}\n')
    if args.out is not None:
        (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'\n{ok}/{len(names)} compile')
    if args.run:
        print(f'{agree}/{len(names)} agree on every draw')
    return 0 if ok == len(names) and agree == (len(names) if args.run else 0) else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
