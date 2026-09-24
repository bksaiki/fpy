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
"""

import argparse
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from compile import DESIGNS, _filename

import fpy2 as fp
from fpy2.backend.triton import TritonCompiler, launch, unavailable
from fpy2.backend.triton.storage import choose_storage_scalar
from fpy2.backend.triton.types import TritonScalar
from fpy2.utils import NamedId

_L = fp.types.ListType
_R = fp.types.RealType

_BLOCK = 64

_PREAMBLE = (
    'import triton\n'
    'import triton.language as tl\n'
    'from triton.language.extra import libdevice\n\n\n'
)


def _matmul(design, arity: int):
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


def _at_depth(arg_types, k: int | None):
    """*arg_types* with the dot product *k* long: a multiple of the design's
    own length, with a list of scales growing alike."""
    a, b, c, *scales = arg_types
    k0 = a.length
    if k is None or k == k0:
        return arg_types
    if k % k0:
        raise ValueError(f'k = {k} is not a multiple of the design\'s {k0}')
    grown = []
    for t in scales:
        if not isinstance(t, _L):
            raise TypeError(f'one scale per call: k must be {k0}')
        grown.append(_L(t.elt, t.length * k // k0))
    return [_L(a.elt, k), _L(b.elt, k), c, *grown]


def compile_matmul(build, k: int | None):
    """The kernel for one design as a matmul over *k*, the design, and its
    argument types; raises whatever refused it.  The sizes are the kernel's
    to be told, so one kernel runs at any -- but for a design's scales, whose
    number fixes `k`."""
    design, arg_types = build()
    arg_types = _at_depth(arg_types, k)
    a, b, c, *scales = arg_types
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


def _sample(fmt, rng: random.Random) -> float:
    """A value of *fmt*: half over its whole range, half near one."""
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


def _dtype(fmt):
    import torch
    scalar = choose_storage_scalar(fmt)
    dtypes = {TritonScalar.F16: torch.float16, TritonScalar.F32: torch.float32,
              TritonScalar.F64: torch.float64}
    if scalar not in dtypes:
        raise ValueError(f'no tensor dtype for {scalar}')
    return dtypes[scalar]


def _row(t, rng):
    if isinstance(t, _L):
        return [_sample(t.elt.fmt, rng) for _ in range(t.length)]
    return _sample(t.fmt, rng)


def run_matmul(kernel, design, arg_types, m: int, n: int, trials: int, seed: int,
               block: int | None = _BLOCK) -> int:
    """How many of the *m* x *n* outputs, over *trials* draws, the kernel gets
    bit for bit."""
    import torch

    rng = random.Random(seed)
    a, b, c, *scales = arg_types
    dtype = _dtype(c.fmt)
    agree = 0
    for _ in range(trials):
        A = [_row(a, rng) for _ in range(m)]
        BT = [_row(b, rng) for _ in range(n)]
        C = [[_row(c, rng) for _ in range(n)] for _ in range(m)]
        S = [[_row(t, rng) for _ in range(d)] for t, d in zip(scales, (m, n))]
        out = torch.zeros(m, n, dtype=dtype).cuda()
        launch(kernel, [_tensor(A, a), _tensor(BT, b), _tensor(C, c),
                        *(_tensor(s, t) for s, t in zip(S, scales)), out],
               block=block)
        want = [
            float(design(A[i], BT[j], C[i][j], *((S[0][i], S[1][j]) if S else ())))
            for i in range(m) for j in range(n)
        ]
        agree += _agreeing(out.cpu().flatten().tolist(), want, dtype)
    return agree


def _tensor(values, t):
    """*values* on the GPU, in the dtype that holds *t*'s elements."""
    import torch
    return torch.tensor(values, dtype=_dtype((t.elt if isinstance(t, _L) else t).fmt)).cuda()


def _agreeing(got: list[float], want: list[float], dtype) -> int:
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
    for name, build in designs:
        try:
            kernel, design, arg_types = compile_matmul(build, args.k)
            ran = (run_matmul(kernel, design, arg_types, args.m, args.n,
                              args.run, args.seed,
                              None if args.autotune else _BLOCK)
                   if args.run else None)
        except Exception as ex:  # noqa: BLE001 -- any refusal is a result
            detail = str(ex) if args.verbose else str(ex).split('\n')[0][:110]
            print(f'{name:{width}}  {type(ex).__name__}: {detail}')
            continue
        ok += 1
        note = ''
        if ran is not None:
            agree += ran == total
            note = f'  {ran}/{total} agree'
        if args.out is not None:
            path = args.out / _filename(name).replace('.cpp', '.py')
            path.write_text(_PREAMBLE + kernel.source)
            note += f'  -> {path}'
        print(f'{name:{width}}  OK{note}')
        if args.emit:
            print(f'\n# ==== {name} ====\n{_PREAMBLE}{kernel.source}\n')
    print(f'\n{ok}/{len(designs)} compile')
    if args.run:
        print(f'{agree}/{len(designs)} agree on every output')
    return 0 if ok == len(designs) and agree == (len(designs) if args.run else 0) else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
