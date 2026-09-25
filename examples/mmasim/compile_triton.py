"""
Compiles every MMA-Sim design to a Triton kernel, reporting where each one stops.

A design is one dot product, which has no loop to tile, so each is wrapped in
a matmul: an `m x k` by `n x k` product, `B` given transposed so that a column
is a row, with `out[i][j]` one call.  `m = n = 1` is the dot product itself.
`m`, `n` and `k` are kernel arguments, so one kernel runs at any; `k` is
compiled in for a design taking scales, which are one per call.
Asserts are dropped: a kernel cannot raise.

    python examples/mmasim/compile_triton.py              # one line per design
    python examples/mmasim/compile_triton.py -v           # full error text
    python examples/mmasim/compile_triton.py -e volta     # print a design's kernel
    python examples/mmasim/compile_triton.py -o out/      # write each one to out/
    python examples/mmasim/compile_triton.py -r 64        # also launch 64 draws and
                                                          # compare with the interpreter
    python examples/mmasim/compile_triton.py -m 4 -n 4 -r 1   # a 4 x 4 matmul
    python examples/mmasim/compile_triton.py -r 8 --autotune  # the launch picks
                                                          # its block and warps
    python examples/mmasim/compile_triton.py -j 8         # compile in 8 processes
"""

import argparse
import math
import random
import sys
from collections.abc import Callable
from functools import cache, partial
from pathlib import Path
from typing import TYPE_CHECKING

sys.path.insert(0, str(Path(__file__).parent))

from compile import DESIGNS, _filename, in_processes

import fpy2 as fp
from fpy2.backend.triton import KernelSource, TritonCompiler, launch, unavailable
from fpy2.backend.triton.launcher import MODULE_PREAMBLE
from fpy2.backend.triton.storage import choose_storage_scalar
from fpy2.backend.triton.types import TritonScalar
from fpy2.number.context import Format, SizedFormat
from fpy2.types import Type
from fpy2.utils import NamedId

if TYPE_CHECKING:
    import torch

_L = fp.types.ListType
_R = fp.types.RealType

Build = Callable[[], tuple[fp.Function, list[Type]]]
"""A design's builder: the design, and its argument types."""

_BLOCK = 64
_BLOCK_M = 4
"""`--run`'s tile height: taller than the default `-m`, so the mask on the
rows past the end is exercised."""

_HARD_EVERY = 4
"""Every this many `--run` draws, each element is a hard case with
probability :data:`_HARD`."""

_HARD = 0.25


def _fmt(t: Type) -> SizedFormat:
    """The format of *t*'s values, or of its elements'."""
    elt = t.elt if isinstance(t, _L) else t
    if not (isinstance(elt, _R) and isinstance(elt.fmt, SizedFormat)):
        raise TypeError(f'expected a real type of sized format, got {t}')
    return elt.fmt


def _length(t: Type) -> int:
    """*t*'s length, where it is a list of known length."""
    if not (isinstance(t, _L) and isinstance(t.length, int)):
        raise TypeError(f'expected a list of known length, got {t}')
    return t.length


def _matmul(design: fp.Function, arity: int) -> fp.Function:
    """*design* as a matmul: `out[i][j]` is row `i` of `A` against row `j` of
    `BT`, and a scale is indexed the way its operand is."""
    if arity == 3:
        @fp.fpy(ctx=fp.REAL)
        def matmul3(A, BT, C, out, BLOCK):
            for i in range(len(out)):
                row = out[i]
                for j in range(len(row)):
                    row[j] = design(A[i], BT[j], C[i][j])
            return out
        return matmul3
    if arity == 5:
        @fp.fpy(ctx=fp.REAL)
        def matmul5(A, BT, C, xs, ys, out, BLOCK):
            for i in range(len(out)):
                row = out[i]
                for j in range(len(row)):
                    row[j] = design(A[i], BT[j], C[i][j], xs[i], ys[j])
            return out
        return matmul5
    raise ValueError(f'no matmul wrapper for {arity} arguments')


def _at_depth(arg_types: list[Type], k: int | None) -> list[Type]:
    """*arg_types* with the dot product *k* long: a multiple of the design's
    own length, with a list of scales growing alike."""
    a, b, c, *scales = arg_types
    assert isinstance(a, _L) and isinstance(b, _L)
    k0 = _length(a)
    if k is None or k == k0:
        return arg_types
    if k % k0:
        raise ValueError(f'k = {k} is not a multiple of the design\'s {k0}')
    grown: list[Type] = []
    for t in scales:
        if not isinstance(t, _L):
            raise TypeError(f'one scale per call: k must be {k0}')
        grown.append(_L(t.elt, _length(t) * k // k0))
    return [_L(a.elt, k), _L(b.elt, k), c, *grown]


def compile_matmul(
    build: Build, k: int | None,
) -> tuple[KernelSource, fp.Function, list[Type]]:
    """The kernel for one design as a matmul over *k*, the design, and its
    argument types; raises whatever refused it.  The sizes are the kernel's
    to be told, so one kernel runs at any -- but for a design's scales, whose
    number fixes `k`."""
    design, arg_types = build()
    arg_types = _at_depth(arg_types, k)
    a, b, c, *scales = arg_types
    assert isinstance(a, _L) and isinstance(b, _L)
    m, n = NamedId('m'), NamedId('n')
    if not scales:
        a, b = _L(a.elt, NamedId('k')), _L(b.elt, NamedId('k'))
    square = _L(_L(c, n), m)
    kernel = TritonCompiler(
        drop_asserts=True, unfold=TritonCompiler.UnfoldMode.ROUNDINGS,
    ).compile(
        _matmul(design, len(arg_types)), ctx=fp.REAL,
        arg_types=[_L(a, m), _L(b, n), square,
                   *(_L(t, d) for t, d in zip(scales, (m, n))),
                   square, _R(fp.INTEGER)],
    )
    return kernel, design, arg_types


def _kernel_named(name: str, k: int | None) -> tuple[KernelSource | None, str, str]:
    """The kernel for design *name*, or `None` and the refusal's type and text."""
    try:
        return compile_matmul(dict(DESIGNS)[name], k)[0], '', ''
    except Exception as ex:  # noqa: BLE001 -- any refusal is a result
        return None, type(ex).__name__, str(ex)


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


def _dtype(fmt: Format) -> 'torch.dtype':
    import torch
    scalar = choose_storage_scalar(fmt)
    dtypes = {TritonScalar.F16: torch.float16, TritonScalar.F32: torch.float32,
              TritonScalar.F64: torch.float64}
    if scalar not in dtypes:
        raise ValueError(f'no tensor dtype for {scalar}')
    return dtypes[scalar]


def _vector(t: Type, rng: random.Random, hard: float = 0.0) -> list[float]:
    """A value of each of list type *t*'s elements."""
    return [_sample(_fmt(t), rng, hard) for _ in range(_length(t))]


def _row(t: Type, rng: random.Random, hard: float = 0.0) -> float | list[float]:
    """A value of *t*, or of each of its elements."""
    return _vector(t, rng, hard) if isinstance(t, _L) else _sample(_fmt(t), rng, hard)


def run_matmul(kernel: KernelSource, design: fp.Function, arg_types: list[Type],
               m: int, n: int, trials: int, seed: int,
               block: int | None = _BLOCK, hard_every: int = _HARD_EVERY,
               block_m: int = _BLOCK_M) -> int:
    """How many of the *m* x *n* outputs, over *trials* draws, the kernel gets
    bit for bit.  Every *hard_every*-th draw is heavy in hard cases."""
    import torch

    rng = random.Random(seed)
    a, b, c, *scales = arg_types
    dtype = _dtype(_fmt(c))
    agree = 0
    for trial in range(trials):
        e = _HARD if (trial + 1) % hard_every == 0 else 0.0
        A = [_row(a, rng, e) for _ in range(m)]
        BT = [_row(b, rng, e) for _ in range(n)]
        C = [[_row(c, rng, e) for _ in range(n)] for _ in range(m)]
        S = [[_row(t, rng, e) for _ in range(d)] for t, d in zip(scales, (m, n))]
        out = torch.zeros(m, n, dtype=dtype).cuda()
        launch(kernel, [_tensor(A, a), _tensor(BT, b), _tensor(C, c),
                        *(_tensor(s, t) for s, t in zip(S, scales)), out],
               block=block, block_m=block_m)
        want = [
            float(design(A[i], BT[j], C[i][j], *((S[0][i], S[1][j]) if S else ())))
            for i in range(m) for j in range(n)
        ]
        agree += _agreeing(out.cpu().flatten().tolist(), want, dtype)
    return agree


def _tensor(values: list, t: Type) -> 'torch.Tensor':
    """*values* on the GPU, in the dtype that holds *t*'s elements."""
    import torch
    return torch.tensor(values, dtype=_dtype(_fmt(t))).cuda()


def _agreeing(got: list[float], want: list[float], dtype: 'torch.dtype') -> int:
    """How many of *got* are *want* bit for bit, once *want* is in *dtype*:
    the same value and sign, or both NaN."""
    import torch
    want = torch.tensor(want, dtype=dtype).tolist()
    return sum(
        bool(g == w) and math.copysign(1, g) == math.copysign(1, w)
        or (math.isnan(g) and math.isnan(w))
        for g, w in zip(got, want)
    )


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('filter', nargs='*', metavar='SUBSTRING',
                    help='only designs whose name contains one of these')
    ap.add_argument('-v', '--verbose', action='store_true',
                    help='full error text for a design that does not compile')
    ap.add_argument('-m', type=int, default=1, help="A's rows (default 1)")
    ap.add_argument('-n', type=int, default=1, help="B's columns (default 1)")
    ap.add_argument('-k', type=int, default=None,
                    help="the dot product's length (default: the design's own)")
    ap.add_argument('-r', '--run', metavar='DRAWS', type=int, default=0,
                    help='launch DRAWS random inputs and compare every output, '
                         'bit for bit, with the interpreter')
    ap.add_argument('-s', '--seed', type=int, default=0,
                    help='seed for the inputs --run draws')
    ap.add_argument('-j', '--jobs', type=int, default=1,
                    help='compile in this many processes (default 1); '
                         '--run launches from this one')
    ap.add_argument('--autotune', action='store_true',
                    help=f'let each launch pick its block and warps by timing '
                         f'them, rather than a block of {_BLOCK}')
    dest = ap.add_mutually_exclusive_group()
    dest.add_argument('-e', '--emit', action='store_true',
                      help='print the kernel of each design that compiles')
    dest.add_argument('-o', '--out', metavar='DIR', type=Path,
                      help='write each compiled design to DIR/<name>.py')
    args = ap.parse_args(argv)

    designs = [
        (name, build) for name, build in DESIGNS
        if not args.filter or any(f in name for f in args.filter)
    ]
    if not designs:
        ap.error(f'no design matches {args.filter}')
    if args.run and (why := unavailable()) is not None:
        ap.error(f'--run needs a GPU: {why}')
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)

    total = args.run * args.m * args.n
    width = max(len(name) for name, _ in designs)
    ok = agree = 0
    kernels = in_processes(partial(_kernel_named, k=args.k),
                           [name for name, _ in designs], args.jobs)
    for (name, build), (kernel, kind, why) in zip(designs, kernels):
        ran = None
        if kernel is not None and args.run:
            design, arg_types = build()
            try:
                ran = run_matmul(kernel, design, _at_depth(arg_types, args.k),
                                 args.m, args.n, args.run, args.seed,
                                 None if args.autotune else _BLOCK)
            except Exception as ex:  # noqa: BLE001 -- any refusal is a result
                kernel, kind, why = None, type(ex).__name__, str(ex)
        if kernel is None:
            detail = why if args.verbose else why.split('\n')[0][:110]
            print(f'{name:{width}}  {kind}: {detail}')
            continue
        ok += 1
        note = ''
        if ran is not None:
            agree += ran == total
            note = f'  {ran}/{total} agree'
        if args.out is not None:
            path = args.out / _filename(name).replace('.cpp', '.py')
            path.write_text(MODULE_PREAMBLE + kernel.source)
            note += f'  -> {path}'
        print(f'{name:{width}}  OK{note}')
        if args.emit:
            print(f'\n# ==== {name} ====\n{MODULE_PREAMBLE}{kernel.source}\n')
    print(f'\n{ok}/{len(designs)} compile')
    if args.run:
        print(f'{agree}/{len(designs)} agree on every output')
    return 0 if ok == len(designs) and agree == (len(designs) if args.run else 0) else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
