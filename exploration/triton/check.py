"""Differential harness behind *What is implemented* in
`docs/todos/backend-triton.md`.

    python -m exploration.triton.check            # everything available here
    python -m exploration.triton.check --n 4096   # more samples

Two halves, deliberately separable:

**The contraction question needs no GPU.**  Whether fused multiply-add is
observable is a question about FPy's own semantics, and the interpreter answers
it: run each program beside its `fp.fma` twin and see whether they ever differ.
The prediction is that they never do where the product is exact and sometimes
do where it is not -- which, if it holds, means `enable_fp_fusion` is not a
flag to pin off globally but one the backend can *derive* per kernel from
`scalar_fits_in`, keeping the fusion where it is free.

**The rest needs a GPU**, and reports as skipped without one.  It compiles the
hand-written kernels, runs them against the interpreter and compares fp32 bit
patterns -- with the deliberately-wrong `dot_trap` included so a pass that
proves nothing is visible as such.
"""

import argparse
import random
import struct
import sys

import fpy2 as fp

from . import kernels as kmod
from .programs import (
    K,
    dot_exact_product,
    dot_exact_product_fma,
    dot_rounded_product,
    dot_rounded_product_fma,
)


def _bits32(x: float) -> int:
    """fp32 bit pattern of *x*, so NaN compares by payload and -0.0 != 0.0."""
    return struct.unpack('<I', struct.pack('<f', x))[0]


def _sample(rng: random.Random, ctx, n: int) -> list[list]:
    """*n* rows of `K` values, rounded into *ctx*.

    Mixed magnitudes on purpose: a product only rounds when the operands carry
    enough significand, and a sum only cancels when the terms are comparable.
    """
    rows = []
    for _ in range(n):
        row = []
        for _ in range(K):
            v = rng.uniform(-1, 1) * (2.0 ** rng.randint(-8, 8))
            row.append(ctx.round(v))
        rows.append(row)
    return rows


def contraction_report(n: int, seed: int) -> int:
    """Does fma contraction change the answer?  Interpreter only."""
    rng = random.Random(seed)
    print('== fma contraction (no GPU needed) ==')
    failures = 0
    cases = (
        ('exact product   (FP16 in, FP32 acc)', fp.FP16,
         dot_exact_product, dot_exact_product_fma, False),
        ('rounded product (FP32 throughout) ', fp.FP32,
         dot_rounded_product, dot_rounded_product_fma, True),
    )
    for label, ctx, plain, fused, expect_differ in cases:
        rows = _sample(rng, ctx, n)
        differ = 0
        for xs in rows:
            ys = _sample(rng, ctx, 1)[0]
            a, b = float(plain(xs, ys)), float(fused(xs, ys))
            if _bits32(a) != _bits32(b):
                differ += 1
        verdict = 'differs' if differ else 'identical'
        ok = (differ > 0) == expect_differ
        failures += not ok
        print(f'  {label}: {verdict} on {differ}/{n}'
              f'   [{"as predicted" if ok else "PREDICTION FAILED"}]')
    print()
    if not failures:
        print('  => enable_fp_fusion is derivable, not a global pin: safe'
              ' exactly where the product\n     fits its storage, which is'
              ' `scalar_fits_in` and already computed.\n')
    return failures


def gpu_report(n: int, seed: int) -> int:
    print('== kernels vs interpreter (needs GPU) ==')
    if not kmod.available:
        print(f'  SKIPPED - {kmod.why_unavailable()}')
        print('  install: uv pip install torch triton\n')
        return 0
    import torch

    if not torch.cuda.is_available():
        print('  SKIPPED - torch reports no CUDA device\n')
        return 0
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f'  device: {name}  sm_{cap[0]}{cap[1]}')
    if cap < (8, 0):
        print('  note: pre-Ampere - bf16 and TF32 are untestable here')

    rng = random.Random(seed)
    failures = 0
    for label, ctx, dtype, prog, kname in (
        ('dot_exact', fp.FP16, torch.float16, dot_exact_product, 'dot_exact'),
        ('dot_trap ', fp.FP16, torch.float16, dot_exact_product, 'dot_trap'),
        ('dot_fp32 ', fp.FP32, torch.float32, dot_rounded_product, 'dot_fp32'),
    ):
        xs = _sample(rng, ctx, n)
        ys = _sample(rng, ctx, n)
        want = [float(prog(a, b)) for a, b in zip(xs, ys)]

        xt = torch.tensor([[float(v) for v in r] for r in xs],
                          dtype=dtype, device='cuda')
        yt = torch.tensor([[float(v) for v in r] for r in ys],
                          dtype=dtype, device='cuda')
        out = torch.empty(n, dtype=torch.float32, device='cuda')
        grid = ((n + kmod.BLOCK - 1) // kmod.BLOCK,)
        kmod.kernels[kname][grid](
            xt, yt, out, n, K=K, BLOCK=kmod.BLOCK, enable_fp_fusion=False,
        )
        got = out.tolist()
        bad = sum(_bits32(a) != _bits32(b) for a, b in zip(want, got))
        # dot_trap is *meant* to disagree; a clean run there means the sample
        # never exercised the fp16 rounding and proves nothing.
        expect_bad = kname == 'dot_trap'
        ok = (bad > 0) == expect_bad
        failures += not ok
        print(f'  {label}: {bad}/{n} differ'
              f'   [{"as predicted" if ok else "PREDICTION FAILED"}]')
    fusion_report(n, seed)
    print()
    return failures


def fusion_report(n: int, seed: int) -> None:
    """Confirm on hardware what `contraction_report` predicted from semantics:
    fusion is unobservable where the product is exact and observable where it
    is not.  This is the claim the derived flag rests on."""
    import torch
    rng = random.Random(seed + 1)
    print('  -- enable_fp_fusion on hardware --')
    for label, ctx, dtype, prog, kname, expect_differ in (
        ('exact product  (dot_exact)', fp.FP16, torch.float16,
         dot_exact_product, 'dot_exact', False),
        ('rounded product (dot_fp32)', fp.FP32, torch.float32,
         dot_rounded_product, 'dot_fp32', True),
    ):
        xs = _sample(rng, ctx, n)
        ys = _sample(rng, ctx, n)
        want = [float(prog(a, b)) for a, b in zip(xs, ys)]
        xt = torch.tensor([[float(v) for v in r] for r in xs],
                          dtype=dtype, device='cuda')
        yt = torch.tensor([[float(v) for v in r] for r in ys],
                          dtype=dtype, device='cuda')
        grid = ((n + kmod.BLOCK - 1) // kmod.BLOCK,)
        res = {}
        for fuse in (True, False):
            out = torch.empty(n, dtype=torch.float32, device='cuda')
            kmod.kernels[kname][grid](
                xt, yt, out, n, K=K, BLOCK=kmod.BLOCK, enable_fp_fusion=fuse,
            )
            res[fuse] = sum(_bits32(a) != _bits32(b)
                            for a, b in zip(want, out.tolist()))
        ok = (res[True] > 0) == expect_differ and res[False] == 0
        print(f'     {label}: fused {res[True]}/{n} differ, '
              f'unfused {res[False]}/{n}'
              f'   [{"as predicted" if ok else "PREDICTION FAILED"}]')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--n', type=int, default=512, help='samples per case')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    f = contraction_report(a.n, a.seed) + gpu_report(a.n, a.seed)
    print('FAILURES:', f) if f else print('all predictions held')
    return 1 if f else 0


if __name__ == '__main__':
    sys.exit(main())
