"""
Compiles every MMA-Sim design to a Triton kernel, reporting where each one stops.

A design is one dot product, which has no loop to tile, so each is wrapped in
a matmul: an `m x k` by `n x k` product, `B` given transposed so that a column
is a row, with `out[i][j]` one call.  `m = n = 1` is the dot product itself.
`m`, `n` and `k` are kernel arguments, so one kernel runs at any; `k` is
compiled in for a design taking scales, which are one per call, and for
`--fuse`, a design over a longer `k` in one kernel.  Asserts are dropped: a
kernel cannot raise.

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
    python examples/mmasim/compile_triton.py --fuse 512 -r 8   # each over k = 512,
                                                          # in one kernel
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
    Build,
    _filename,
    _fmt,
    _length,
    _row,
    _same,
    in_processes,
)

import fpy2 as fp
from fpy2.backend.triton import KernelSource, TritonCompiler, launch, unavailable
from fpy2.backend.triton.launcher import MODULE_PREAMBLE
from fpy2.backend.triton.storage import choose_storage_scalar
from fpy2.backend.triton.types import TritonScalar
from fpy2.number.context import Format
from fpy2.types import Type
from fpy2.utils import NamedId

if TYPE_CHECKING:
    import torch

_L = fp.types.ListType
_R = fp.types.RealType

_BLOCK = 64
_BLOCK_M = 4
"""`--run`'s tile height: taller than the default `-m`, so the mask on the
rows past the end is exercised."""


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
    if not scales:
        a, b = _L(a.elt, NamedId('k')), _L(b.elt, NamedId('k'))
    return _compile(design, [a, b, c, *scales]), design, arg_types


def _compile(f: fp.Function, arg_types: list[Type]) -> KernelSource:
    """*f*, one output from `(A, B, C, *scales)` of *arg_types*, as a matmul
    kernel with `m` and `n` symbolic, a scale indexed as its operand."""
    a, b, c, *scales = arg_types
    m, n = NamedId('m'), NamedId('n')
    square = _L(_L(c, n), m)
    return TritonCompiler(
        drop_asserts=True, unfold=TritonCompiler.UnfoldMode.ROUNDINGS,
    ).compile(
        _matmul(f, len(arg_types)), ctx=fp.REAL,
        arg_types=[_L(a, m), _L(b, n), square,
                   *(_L(t, d) for t, d in zip(scales, (m, n))),
                   square, _R(fp.INTEGER)],
    )


def _chained(design: fp.Function, k0: int, per: int | None) -> fp.Function:
    """A block-scaled *design*, one instruction of *k0* elements, chained over
    `k`: instruction `t` takes its elements, its scales (one, or *per*), and
    the last one's result as its accumulator."""
    if per is None:
        @fp.fpy(ctx=fp.REAL)
        def chain1(A, B, c, xs, ys):
            d = c
            for t in range(len(xs)):
                i = t * k0
                d = design(A[i:i + k0], B[i:i + k0], d, xs[t], ys[t])
            return d
        return chain1
    g = k0 // per

    @fp.fpy(ctx=fp.REAL)
    def chain(A, B, c, xs, ys):
        d = c
        for t in range(0, len(xs), per):
            i = t * g
            d = design(A[i:i + k0], B[i:i + k0], d, xs[t:t + per], ys[t:t + per])
        return d
    return chain


def _promoted(design: fp.Function, block: int) -> fp.Function:
    """*design* over `k`, *block* elements at a time from a zero accumulator,
    each block's partial scaled into the result in FP32, `y + p * (s_x *
    s_w)`, as block-scaled FP8 promotes it."""
    @fp.fpy(ctx=fp.REAL)
    def promoted(A, B, c, xs, ys):
        y = c
        for t in range(len(xs)):
            i = t * block
            p = design(A[i:i + block], B[i:i + block], 0.0)
            with fp.FP32:
                y = y + p * (xs[t] * ys[t])
        return y
    return promoted


PROMOTE = 128
"""`k` per FP32 promotion of a design without scales of its own (`fp8-block`'s
block)."""


def fuse(build: Build, k: int) -> tuple[fp.Function, list[Type]]:
    """One design over a static *k*, and its argument types: a block-scaled
    one chained per instruction (:func:`_chained`), any other promoted every
    :data:`PROMOTE` of `k` with FP32 scales (:func:`_promoted`)."""
    design, (a, b, c, *scales) = build()
    assert isinstance(a, _L) and isinstance(b, _L)
    k0 = _length(a)
    if scales:
        if k % k0:
            raise ValueError(f'k = {k} is not a multiple of the design\'s {k0}')
        sx = scales[0]
        per = _length(sx) if isinstance(sx, _L) else None
        f, s, elt = _chained(design, k0, per), k // k0 * (per or 1), getattr(sx, 'elt', sx)
    else:
        if k % PROMOTE or PROMOTE % k0:
            raise ValueError(f'k = {k} is not a multiple of {PROMOTE}, or {PROMOTE} of {k0}')
        f, s, elt = _promoted(design, PROMOTE), k // PROMOTE, _R(fp.FP32)
    return f, [_L(a.elt, k), _L(b.elt, k), c, _L(elt, s), _L(elt, s)]


def compile_fused(build: Build, k: int) -> KernelSource:
    """:func:`fuse`'s function as a matmul kernel."""
    return _compile(*fuse(build, k))


def _kernel_named(name: str, k: int | None, fuse_k: int | None,
                  ) -> tuple[KernelSource | None, str, str]:
    """The kernel for design *name* (fused over *fuse_k*, if given), or `None`
    and the refusal's type and text."""
    try:
        build = dict(DESIGNS)[name]
        return (compile_matmul(build, k)[0] if fuse_k is None else compile_fused(build, fuse_k)), '', ''
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
                    help="the dot product's length (default: the design's own)")
    ap.add_argument('--fuse', metavar='K', type=int, default=None,
                    help=f'each design over K in one kernel: a block-scaled one chained per '
                         f'instruction, any other promoted every {PROMOTE} with FP32 scales')
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
    if args.fuse is not None and args.k is not None:
        ap.error('-k and --fuse K both give the length; pass one')
    if args.run and (why := unavailable()) is not None:
        ap.error(f'--run needs a GPU: {why}')
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)

    total = args.run * args.m * args.n
    width = max(len(name) for name, _ in designs)
    ok = agree = 0
    kernels = in_processes(partial(_kernel_named, k=args.k, fuse_k=args.fuse),
                           [name for name, _ in designs], args.jobs)
    for (name, build), (kernel, kind, why) in zip(designs, kernels):
        ran = None
        if kernel is not None and args.run:
            if args.fuse is None:
                design, arg_types = build()
                arg_types = _at_depth(arg_types, args.k)
            else:
                design, arg_types = fuse(build, args.fuse)
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
