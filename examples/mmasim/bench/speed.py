"""
Runtime of each design's Triton matmul kernel.

Each design compiles once, with `m`, `n` and `k` as kernel arguments, and runs
an `m x k` by `n x k` product at several block sizes, timed by CUDA events.
Throughput counts `2 m n k` operations, as a matmul does, so the
`torch.matmul` rows -- not bit-exact -- are a ceiling to read against.

The inputs are random rather than drawn from each format: every branch is
flattened, so a kernel's time does not depend on its data.

    python examples/mmasim/bench/speed.py                   # every design
    python examples/mmasim/bench/speed.py volta nvfp4       # some
    python examples/mmasim/bench/speed.py -m 512 --blocks 64 128
"""

import argparse
import sys
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

sys.path.insert(0, str(Path(__file__).parent.parent))

import compile_triton as ct
from compile import DESIGNS

from fpy2.backend.triton import launch, unavailable
from fpy2.types import Type

if TYPE_CHECKING:
    import torch


def _timed(fn: Callable[[], object], reps: int) -> float:
    """Seconds per call of *fn*, after one warm-up."""
    import torch
    fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        fn()
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) / reps / 1e3


def _inputs(arg_types: list[Type], m: int, n: int, k: int) -> list['torch.Tensor']:
    """Random tensors in each argument's storage, and the output."""
    import torch
    a, b, c, *scales = arg_types
    g = torch.Generator(device='cuda').manual_seed(0)

    def rand(*shape: int, t: Type) -> torch.Tensor:
        return torch.randn(*shape, device='cuda', generator=g).to(ct._dtype(ct._fmt(t)))

    scale_args = [
        rand(d, *([ct._length(t)] if isinstance(t, ct._L) else []), t=t).abs() + 1
        for t, d in zip(scales, (m, n))
    ]
    out = torch.zeros(m, n, device='cuda', dtype=ct._dtype(ct._fmt(c)))
    return [rand(m, k, t=a), rand(n, k, t=b), rand(m, n, t=c), *scale_args, out]


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('filter', nargs='*', metavar='SUBSTRING',
                    help='only designs whose name contains one of these')
    ap.add_argument('-m', type=int, default=1024, help="A's rows (default 1024)")
    ap.add_argument('-n', type=int, default=None, help="B's columns (default m)")
    ap.add_argument('-k', type=int, default=256,
                    help='the dot product length, rounded down to a multiple of '
                         "the design's; a design taking scales keeps its own "
                         '(default 256)')
    ap.add_argument('--blocks', type=int, nargs='*', default=[16, 32, 64, 128],
                    help='block sizes to time (default 16 32 64 128)')
    ap.add_argument('--autotune', action='store_true',
                    help='also time the launch that picks its own block and warps')
    ap.add_argument('--reps', type=int, default=20, help='launches per timing')
    args = ap.parse_args(argv)
    if (why := unavailable()) is not None:
        ap.error(f'needs a GPU: {why}')
    import torch

    m, n = args.m, args.n or args.m
    print(f'{"design":22} {"k":>5} {"block":>5} {"ms":>9} {"GFLOP/s":>9}')
    for name, build in DESIGNS:
        if args.filter and not any(f in name for f in args.filter):
            continue
        try:
            kernel, _, arg_types = ct.compile_matmul(build, None)
        except Exception as ex:  # noqa: BLE001 -- a refusal is a result
            print(f'{name:22} {type(ex).__name__}')
            continue
        a, _, _, *scales = arg_types
        k0 = ct._length(a)
        k = k0 if scales else max(args.k // k0, 1) * k0
        tensors = _inputs(arg_types, m, n, k)
        for block in [*args.blocks, *([None] if args.autotune else [])]:
            t = _timed(partial(launch, kernel, tensors, block=block), args.reps)
            label = 'auto' if block is None else block
            print(f'{name:22} {k:5} {label:>5} {t * 1e3:9.3f} {2 * m * n * k / t / 1e9:9.1f}')

    torch.backends.cuda.matmul.allow_tf32 = False
    for dtype in (torch.float16, torch.float32):
        x = torch.randn(m, args.k, device='cuda', dtype=dtype)
        y = torch.randn(args.k, n, device='cuda', dtype=dtype)
        t = _timed(partial(torch.matmul, x, y), args.reps)
        label = f'torch.matmul {str(dtype).split(".")[1]}'
        print(f'{label:22} {args.k:5} {"":5} {t * 1e3:9.3f} {2 * m * n * args.k / t / 1e9:9.1f}')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
