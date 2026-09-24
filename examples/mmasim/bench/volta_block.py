"""
The volta T-FDPA chain written by hand in Triton, as a reference for the
generated kernel: one output per row of a `[BLOCK_R, L]` tile, lanes over a
block's `L` elements, reductions across them.

Checked bit for bit against the interpreter, special values included, then
timed at the work `speed.py` gives the generated kernel: `rows` dot products
of length `k`.

    python examples/mmasim/bench/volta_block.py
    python examples/mmasim/bench/volta_block.py --rows 65536 -k 1024
"""

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

E_ZERO = tl.constexpr(-131)   # -108 - F, the FP32-wide datapath
F = tl.constexpr(23)
EMIN_AB = tl.constexpr(-14)   # FP16
EMIN_C = tl.constexpr(-126)   # FP32


@triton.jit
def volta_dpa(A, B, C, OUT, R, K, BLOCK_R: tl.constexpr, L: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    lanes = tl.arange(0, L)
    live = rows < R
    d = tl.load(C + rows, mask=live, other=0.0)
    INF = float('inf')
    for kb in range(0, K, L):
        off = rows[:, None] * K + kb + lanes[None, :]
        a = tl.load(A + off, mask=live[:, None], other=0.0).to(tl.float32)
        b = tl.load(B + off, mask=live[:, None], other=0.0).to(tl.float32)
        p = a * b                                   # exact: 11 x 11 bits
        # special values (Sec. 4.2), each `any` a max across the lanes
        nonfinite = (tl.max((~(tl.abs(p) < INF)).to(tl.int32), axis=1) > 0) | ~(tl.abs(d) < INF)
        nan_in = (tl.max(((a != a) | (b != b)).to(tl.int32), axis=1) > 0) | (d != d)  # noqa: PLR0124
        infl = (tl.abs(a) == INF) | (tl.abs(b) == INF)
        inf_nan = tl.max((infl & (p != p)).to(tl.int32), axis=1) > 0  # noqa: PLR0124
        pos = (tl.max((infl & (p > 0)).to(tl.int32), axis=1) > 0) | (d == INF)
        neg = (tl.max((infl & (p < 0)).to(tl.int32), axis=1) > 0) | (d == -INF)
        special = tl.where(nan_in | inf_nan | (pos & neg), float('nan'),
                           tl.where(pos, INF, -INF))
        # exponents; a zero reads E_ZERO
        es = tl.where(p == 0, E_ZERO,
                      tl.maximum(libdevice.ilogb(a), EMIN_AB) + tl.maximum(libdevice.ilogb(b), EMIN_AB))
        ec = tl.where(d == 0, E_ZERO, tl.maximum(libdevice.ilogb(d), EMIN_C))
        q = tl.maximum(tl.max(es, axis=1), ec) - F      # the lsb every term keeps
        # truncate each term at 2^q and sum: exact, so in any order
        pt = libdevice.trunc(libdevice.ldexp(p.to(tl.float64), -q[:, None]))
        ct = libdevice.trunc(libdevice.ldexp(d.to(tl.float64), -q))
        s = libdevice.ldexp(tl.sum(pt, axis=1) + ct, q)
        # fdpa_round: overflow to infinity, otherwise RZ to FP32
        big = libdevice.ilogb(s) >= 128
        r = tl.where(big, tl.where(s < 0, -INF, INF), libdevice.double2float_rz(s))
        d = tl.where(nonfinite, special, r)
    tl.store(OUT + rows, d, mask=live)


def _run(A, B, c, out, block_r: int):
    rows, k = A.shape
    volta_dpa[(triton.cdiv(rows, block_r),)](A, B, c, out, rows, k,
                                            BLOCK_R=block_r, L=4)
    return out


def check(n: int = 256) -> int:
    """How many of *n* rows, specials among them, agree with the model."""
    import compile_triton as ct
    from compile import DESIGNS
    _, build = next(d for d in DESIGNS if 'volta' in d[0])
    design, (a_t, b_t, c_t) = build()
    rng = random.Random(0)
    picks = [float('inf'), -float('inf'), float('nan'), 0.0, -0.0]

    def spice(v):
        return rng.choice(picks) if rng.random() < 0.05 else v

    A = [[spice(v) for v in ct._row(a_t, rng)] for _ in range(n)]
    B = [[spice(v) for v in ct._row(b_t, rng)] for _ in range(n)]
    c = [spice(ct._row(c_t, rng)) for _ in range(n)]
    out = torch.empty(n, dtype=torch.float32, device='cuda')
    got = _run(torch.tensor(A).half().cuda(), torch.tensor(B).half().cuda(),
               torch.tensor(c).cuda(), out, 32)
    want = [float(design(A[r], B[r], c[r])) for r in range(n)]
    return ct._agreeing(got.cpu().tolist(), want, torch.float32)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('--rows', type=int, default=1024 * 1024,
                    help='dot products per launch (default 1024 x 1024)')
    ap.add_argument('-k', type=int, default=256, help='their length (default 256)')
    ap.add_argument('--reps', type=int, default=10)
    args = ap.parse_args(argv)

    agree = check()
    print(f'agree {agree}/256 with the interpreter')
    A = torch.randn(args.rows, args.k, device='cuda').half()
    B = torch.randn(args.rows, args.k, device='cuda').half()
    c = torch.randn(args.rows, device='cuda')
    out = torch.empty_like(c)
    for block_r in (32, 64, 128):
        _run(A, B, c, out, block_r)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.reps):
            _run(A, B, c, out, block_r)
        stop.record()
        torch.cuda.synchronize()
        t = start.elapsed_time(stop) / args.reps / 1e3
        print(f'BLOCK_R {block_r:4}  {t * 1e3:8.3f} ms  '
              f'{2 * args.rows * args.k / t / 1e9:8.1f} GFLOP/s')
    return 0 if agree == 256 else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
