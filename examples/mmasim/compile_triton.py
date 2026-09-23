"""
Compiles every MMA-Sim design to a Triton kernel, reporting where each one stops.

A design is one dot product, which has no loop to tile, so each is wrapped in
a batch of independent ones -- one per lane, which is what a warp does.
Asserts are dropped: a kernel cannot raise.

    python examples/mmasim/compile_triton.py           # one line per design
    python examples/mmasim/compile_triton.py -v        # full error text
    python examples/mmasim/compile_triton.py -e volta  # print a design's kernel
    python examples/mmasim/compile_triton.py -o out/   # write each one to out/
    python examples/mmasim/compile_triton.py -r 64     # also launch 64 rows and
                                                       # compare with the interpreter
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

_L = fp.types.ListType
_R = fp.types.RealType

_BLOCK = 16

_PREAMBLE = (
    'import triton\n'
    'import triton.language as tl\n'
    'from triton.language.extra import libdevice\n\n\n'
)


def _batched(design, arity: int):
    """*design* over a batch: row `r` of each argument is one call."""
    if arity == 3:
        @fp.fpy(ctx=fp.REAL)
        def batched3(As, Bs, cs, out, BLOCK):
            for r in range(len(out)):
                out[r] = design(As[r], Bs[r], cs[r])
            return out
        return batched3
    if arity == 5:
        @fp.fpy(ctx=fp.REAL)
        def batched5(As, Bs, cs, xs, ys, out, BLOCK):
            for r in range(len(out)):
                out[r] = design(As[r], Bs[r], cs[r], xs[r], ys[r])
            return out
        return batched5
    raise ValueError(f'no batch wrapper for {arity} arguments')


def compile_design(build, rows: int):
    """The kernel for one design over *rows* dot products, the design, and its
    argument types; raises whatever refused it."""
    design, arg_types = build()
    batch = [_L(t, rows) for t in arg_types]
    # the accumulator's format is the result's
    out = _L(arg_types[2], rows)
    kernel = TritonCompiler(
        drop_asserts=True, unfold=TritonCompiler.UnfoldMode.ROUNDINGS,
    ).compile(
        _batched(design, len(arg_types)), ctx=fp.REAL,
        arg_types=[*batch, out, _R(fp.INTEGER)],
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


def run_design(kernel, design, arg_types, rows: int, seed: int) -> int:
    """How many of *rows* random dot products the kernel gets bit for bit."""
    import torch

    rng = random.Random(seed)
    args = [[_row(t, rng) for _ in range(rows)] for t in arg_types]
    tensors = [
        torch.tensor(a, dtype=_dtype((t.elt if isinstance(t, _L) else t).fmt)).cuda()
        for a, t in zip(args, arg_types)
    ]
    dtype = _dtype(arg_types[2].fmt)
    out = torch.zeros(rows, dtype=dtype).cuda()
    launch(kernel, [*tensors, out], block=_BLOCK)

    got = out.cpu()
    want = torch.tensor(
        [float(design(*(a[r] for a in args))) for r in range(rows)],
        dtype=dtype,
    )
    return sum(
        bool(g == w) and math.copysign(1, g) == math.copysign(1, w)
        or (math.isnan(g) and math.isnan(w))
        for g, w in zip(got.tolist(), want.tolist())
    )


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('filter', nargs='*', metavar='SUBSTRING',
                    help='only designs whose name contains one of these')
    ap.add_argument('-v', '--verbose', action='store_true',
                    help='full error text for a design that does not compile')
    ap.add_argument('-r', '--run', metavar='ROWS', type=int, default=0,
                    help='launch ROWS dot products and compare each, bit for '
                         'bit, with the interpreter')
    ap.add_argument('-s', '--seed', type=int, default=0,
                    help='seed for the inputs --run draws')
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

    rows = args.run or _BLOCK
    width = max(len(name) for name, _ in designs)
    ok = agree = 0
    for name, build in designs:
        try:
            kernel, design, arg_types = compile_design(build, rows)
            ran = (run_design(kernel, design, arg_types, rows, args.seed)
                   if args.run else None)
        except Exception as ex:  # noqa: BLE001 -- any refusal is a result
            detail = str(ex) if args.verbose else str(ex).split('\n')[0][:110]
            print(f'{name:{width}}  {type(ex).__name__}: {detail}')
            continue
        ok += 1
        note = ''
        if ran is not None:
            agree += ran == rows
            note = f'  {ran}/{rows} agree'
        if args.out is not None:
            path = args.out / _filename(name).replace('.cpp', '.py')
            path.write_text(_PREAMBLE + kernel.source)
            note += f'  -> {path}'
        print(f'{name:{width}}  OK{note}')
        if args.emit:
            print(f'\n# ==== {name} ====\n{_PREAMBLE}{kernel.source}\n')
    print(f'\n{ok}/{len(designs)} compile')
    if args.run:
        print(f'{agree}/{len(designs)} agree on every row')
    return 0 if ok == len(designs) and agree == (len(designs) if args.run else 0) else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
