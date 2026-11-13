"""
Compiles every MMA-Sim design to a Triton kernel, reporting where each one stops.

A design is one dot product, which has no loop to tile, so each is wrapped in
a matmul: an `m x k` by `n x k` product, `B` given transposed so that a column
is a row, with `out[i][j]` one call.  `m = n = 1` is the dot product itself.
`m`, `n` and `k` are kernel arguments, so one kernel runs at any; a
block-scaled design runs as a chain of its instructions, each taking its
scales.  Asserts are dropped: a kernel cannot raise.

    python evaluation/common/mma/compile_triton.py              # one line per design
    python evaluation/common/mma/compile_triton.py -v           # full error text
    python evaluation/common/mma/compile_triton.py -e volta     # print a design's kernel
    python evaluation/common/mma/compile_triton.py -o out/      # write each one to out/
    python evaluation/common/mma/compile_triton.py -r 64        # also launch 64 draws and
                                                          # compare with the interpreter
    python evaluation/common/mma/compile_triton.py -m 4 -n 4 -r 1   # a 4 x 4 matmul
    python evaluation/common/mma/compile_triton.py -r 8 --autotune  # the launch picks
                                                          # its block and warps
    python evaluation/common/mma/compile_triton.py -j 8         # compile in 8 processes
    python evaluation/common/mma/compile_triton.py blackwell -r 8 -k 192   # three or six
                                                          # instructions a call
"""

import argparse
import random
import sys
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

sys.path.insert(0, str(Path(__file__).parent))

from compile import (
    _HARD,
    _HARD_EVERY,
    DESIGNS,
    _filename,
    _fmt,
    _length,  # noqa: F401 -- other modules read `ct._length`
    _row,
    _same,
    in_processes,
)
from designs import compile_matmul, typed

import fpy2 as fp
from fpy2.backend.triton import KernelSource, launch, unavailable
from fpy2.backend.triton.launcher import MODULE_PREAMBLE
from fpy2.backend.triton.storage import choose_storage_scalar
from fpy2.backend.triton.types import TritonScalar
from fpy2.number.context import Format
from fpy2.types import Type

if TYPE_CHECKING:
    import torch

_L = fp.types.ListType
_R = fp.types.RealType

_BLOCK = 64
_BLOCK_M = 4
"""`--run`'s tile height: taller than the default `-m`, so the mask on the
rows past the end is exercised."""


def _kernel_named(name: str, k: int | None) -> tuple[KernelSource | None, str, str]:
    """The kernel for design *name*, `k` checked against it, or `None` and the
    refusal's type and text."""
    try:
        return compile_matmul(dict(DESIGNS)[name], k)[0], '', ''
    except Exception as ex:  # noqa: BLE001 -- any refusal is a result
        return None, type(ex).__name__, str(ex)


def _dtype(fmt: Format) -> 'torch.dtype':
    import torch
    scalar = choose_storage_scalar(fmt)
    dtypes = {TritonScalar.F16: torch.float16, TritonScalar.F32: torch.float32,
              TritonScalar.F64: torch.float64}
    if scalar not in dtypes:
        raise ValueError(f'no tensor dtype for {scalar}')
    return dtypes[scalar]


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
    return sum(_same(g, w) for g, w in zip(got, want))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('filter', nargs='*', metavar='SUBSTRING',
                    help='only designs whose name contains one of these')
    ap.add_argument('-v', '--verbose', action='store_true',
                    help='full error text for a design that does not compile')
    ap.add_argument('-m', type=int, default=1, help="A's rows (default 1)")
    ap.add_argument('-n', type=int, default=1, help="B's columns (default 1)")
    ap.add_argument('-k', type=int, default=None,
                    help="the dot product's length --run draws (default: the design's own)")
    ap.add_argument('-r', '--run', metavar='DRAWS', type=int, default=0,
                    help='launch DRAWS random inputs and compare every output, '
                         'bit for bit, with the interpreter')
    ap.add_argument('-s', '--seed', type=int, default=0,
                    help='seed for the inputs --run draws (default 0)')
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
            design, arg_types = typed(build, args.k)
            try:
                ran = run_matmul(kernel, design, arg_types,
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
