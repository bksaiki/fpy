"""
Validation for the AMD MMA models in `amd.py`.

Two tiers:

1. Directed checks (need only `fpy2`): the paper's Eq. 10 / Table 8
   example, special values, and FTZ flushing semantics.

2. A randomized differential sweep against the MMA-Sim reference
   implementation (https://github.com/microsoft/MMA-Sim), comparing
   every model bit-for-bit over five input distributions: random
   bit-patterns, edge values, zero-mixed bit-patterns, standard
   normal, and normal scaled so that 1 becomes 2^emin. This tier
   needs the `mmasim` package (see requirements.txt):

       pip install -r requirements.txt
       python validate_amd.py [--trials N]
"""

import argparse
import math
import os
import random
import sys

# `amd.py` lives in the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fpy2 as fp
import amd
import nv

FAILURES = 0

def check(name, ok, detail=''):
    global FAILURES
    print(f'{"ok  " if ok else "FAIL"} {name}{"" if ok else "  " + detail}')
    if not ok:
        FAILURES += 1

###########################################################
# Tier 1: directed checks (fpy2 only)

def pad(xs, k):
    return xs + [0.0] * (k - len(xs))

def check_table8():
    # Eq. 10: d = 2^23 + (-2^23) + (-0.5) + (-0.25) + (-0.125) = -0.875 exactly
    a = [-2.0**13, -0.5, -0.25, -0.125]
    b = [2.0**10, 1.0, 1.0, 1.0]
    c = 2.0**23

    cases = [
        ('fp64 (fma)',              amd.make_fma_dpa(fp.FP64),                   4,  -0.875),
        ('cdna1.bf16 (e-fdpa)',     amd.make_e_fdpa(2),                          4,  -0.875),
        ('cdna1.f16 (e-fdpa)',      amd.make_e_fdpa(4),                          4,  -0.875),
        ('cdna2.bf16 (ftz, P=2)',   amd.make_ftz_addmul(fp.BF16, 2),             4,  -0.375),
        ('cdna2.bf16_1k (ftz, P=4)', amd.make_ftz_addmul(fp.BF16, 4),            4,  0.0),
        ('cdna2.f16 (ftz, P=4)',    amd.make_ftz_addmul(fp.FP16, 4),             4,  0.0),
        ('cdna3.f16 (tr-fdpa)',     amd.make_tr_fdpa(8, fp.FP16, fp.FP16),       8,  -0.5),
        ('cdna3.bf16 (tr-fdpa)',    amd.make_tr_fdpa(8, fp.BF16, fp.BF16),       8,  -0.5),
        ('cdna3.bf8 (gtr-fdpa)',    amd.make_gtr_fdpa(16, fp.S1E5M2, fp.S1E5M2), 16, -1.0),
    ]
    for name, model, k, expected in cases:
        got = float(model(pad(a, k), pad(b, k), c))
        check(f'table8 {name}', got == expected, f'got {got}, expected {expected}')

def check_directed():
    e2 = amd.make_e_fdpa(2)
    check('special NaN input', fp.isnan(e2([fp.nan(), 0.0], [1.0, 0.0], 0.0)))
    check('special inf * 0', fp.isnan(e2([float('inf'), 0.0], [0.0, 0.0], 0.0)))
    check('special inf - inf',
          fp.isnan(e2([float('inf'), 1.0], [1.0, float('-inf')], 0.0)))
    check('special c = inf', float(e2([1.0, 1.0], [1.0, 1.0], float('inf'))) == float('inf'))

    # CDNA2 FTZ: FP16 subnormal inputs flush to +0.0, so the product
    # vanishes; a subnormal FP32 accumulator flushes to +0.0 too
    ftz = amd.make_ftz_addmul(fp.FP16, 2)
    d = ftz([2.0**-24, 0.0], [1.0, 0.0], 1.0)
    check('ftz input flush', float(d) == 1.0, f'got {float(d)}')
    d = ftz([0.0, 0.0], [0.0, 0.0], -2.0**-130)
    check('ftz accumulator flush to +0', d == 0 and not fp.signbit(d))

    # CDNA3 asymmetric round-down (Sec. 6.2.4): the product sum is
    # rounded toward negative infinity when combined with c
    tr = amd.make_tr_fdpa(8, fp.FP16, fp.FP16)
    d = tr(pad([-2.0**13, -0.5, -0.25, -0.125], 8), pad([2.0**10, 1.0, 1.0, 1.0], 8), 2.0**23)
    check('tr-fdpa round-down', float(d) == -0.5, f'got {float(d)}')

def check_fma(trials):
    import struct
    from fractions import Fraction

    def rand_f64():
        return struct.unpack('<d', random.randbytes(8))[0]

    model = amd.make_fma_dpa(fp.FP64)
    fails = 0
    for _ in range(trials):
        k = random.choice([1, 2, 4, 8])
        a = [rand_f64() for _ in range(k)]
        b = [rand_f64() for _ in range(k)]
        c = rand_f64()
        ref = c
        for x, y in zip(a, b):
            try:
                ref = math.fma(x, y, ref)
            except OverflowError:
                # math.fma raises where IEEE returns an infinity;
                # recover its sign exactly and keep folding
                s = Fraction(x) * Fraction(y) + Fraction(ref)
                ref = math.inf if s > 0 else -math.inf
        mine = float(model(a, b, c))
        if not (ref == mine or (math.isnan(ref) and math.isnan(mine))):
            fails += 1
    check(f'fma_dpa vs math.fma ({trials} trials)', fails == 0, f'{fails} mismatches')

###########################################################
# Tier 2: randomized differential sweep vs MMA-Sim (needs torch)

def run_sweep(trials):
    import torch  # type: ignore[import-not-found]
    from mmasim.arithmetic import fdpa, ftz_mul_add  # type: ignore[import-not-found]

    torch.manual_seed(0)

    def rand_bits(n, dtype):
        """n random bit-patterns of the given dtype."""
        nbits = torch.finfo(dtype).bits
        if nbits == 16:
            raw = torch.randint(0, 2**16, (n,), dtype=torch.int32).to(torch.uint16)
        elif nbits == 32:
            return torch.randint(-2**31, 2**31, (n,), dtype=torch.int32).view(dtype)
        else:
            raw = torch.randint(0, 256, (n,), dtype=torch.uint8)
        return raw.view(dtype)

    NO_INF = (torch.float8_e4m3fn, torch.float8_e4m3fnuz, torch.float8_e5m2fnuz)

    def rand_edge(n, dtype):
        fi = torch.finfo(dtype)
        vals = [0.0, -0.0, fi.smallest_normal * fi.eps, -fi.smallest_normal * fi.eps,
                fi.smallest_normal, -fi.smallest_normal, fi.tiny, 1.0, -1.0,
                fi.max, -fi.max, fi.smallest_normal * 4.0]
        if dtype not in NO_INF:
            vals += [float('inf'), float('-inf')]
        vals += [float('nan')]
        return torch.tensor([random.choice(vals) for _ in range(n)]).to(dtype)

    def rand_mixed(n, dtype):
        """random bits, but ~1/3 of entries forced to +/-0."""
        t = rand_bits(n, dtype)
        for i in range(n):
            r = random.random()
            if r < 0.25:
                t[i] = 0.0
            elif r < 0.35:
                t[i] = -0.0
        return t

    def rand_normal(n, dtype):
        """standard normal, rounded into the input format."""
        return torch.randn(n).to(dtype)

    def rand_subnormal(n, dtype):
        """normal scaled so that 1 becomes 2^emin (subnormal-centered)."""
        emin = int(torch.log2(torch.tensor(torch.finfo(dtype).smallest_normal,
                                           dtype=torch.float64)))
        return (torch.randn(n, dtype=torch.float64) * 2.0 ** emin).float().to(dtype)

    GENS = [rand_bits, rand_edge, rand_mixed, rand_normal, rand_subnormal]

    def out_bits(t):
        return t.view(torch.int32).item() & 0xFFFFFFFF

    def same(ref_t, fpy_x):
        if torch.isnan(ref_t) and fp.isnan(fpy_x):
            return True
        mine = torch.tensor(float(fpy_x), dtype=torch.float32)
        return out_bits(ref_t) == out_bits(mine)

    def run_config(name, op, model, a_dtype, K, tf32=False):
        fails = 0
        for i in range(trials):
            gen = GENS[i % len(GENS)]
            a, b = gen(K, a_dtype), gen(K, a_dtype)
            c = gen(1, torch.float32)[0]
            ref = op.dpa(a.clone(), b.clone(), c.clone())
            if tf32:  # xf32: truncate inputs on the FPy side
                af = [x if math.isnan(x) else float(nv.RZ_TF32.round(x))
                      for x in a.float().tolist()]
                bf = [x if math.isnan(x) else float(nv.RZ_TF32.round(x))
                      for x in b.float().tolist()]
            else:
                af, bf = a.float().tolist(), b.float().tolist()
            mine = model(af, bf, float(c.float()))
            if not same(ref, mine):
                fails += 1
        check(f'sweep {name}: {trials - fails}/{trials}', fails == 0)

    TR = lambda L: fdpa.MMA_TR_FDPA(F=24, F2=31, rho='RNE-FP32', L_max=L)
    GTR = fdpa.MMA_GTR_FDPA(F=24, F2=31, rho='RNE-FP32', L_max=16)

    run_config('cdna1.bf16      L2 K8', fdpa.MMA_E_FDPA(2),
               amd.make_e_fdpa(2), torch.bfloat16, 8)
    run_config('cdna1.f16       L4 K16', fdpa.MMA_E_FDPA(4),
               amd.make_e_fdpa(4), torch.float16, 16)
    run_config('cdna2.bf16      P2 K8', ftz_mul_add.MMA_FTZ_MUL_ADD(2),
               amd.make_ftz_addmul(fp.BF16, 2), torch.bfloat16, 8)
    run_config('cdna2.bf16_1k   P4 K8', ftz_mul_add.MMA_FTZ_MUL_ADD(4),
               amd.make_ftz_addmul(fp.BF16, 4), torch.bfloat16, 8)
    run_config('cdna2.f16       P4 K16', ftz_mul_add.MMA_FTZ_MUL_ADD(4),
               amd.make_ftz_addmul(fp.FP16, 4), torch.float16, 16)
    run_config('cdna3.xf32      L4 K8', TR(4),
               amd.make_tr_fdpa(4, fp.TF32, fp.TF32), torch.float32, 8, tf32=True)
    run_config('cdna3.bf16      L8 K16', TR(8),
               amd.make_tr_fdpa(8, fp.BF16, fp.BF16), torch.bfloat16, 16)
    run_config('cdna3.f16       L8 K16', TR(8),
               amd.make_tr_fdpa(8, fp.FP16, fp.FP16), torch.float16, 16)
    run_config('cdna3.fp8       L16 K32', GTR,
               amd.make_gtr_fdpa(16, fp.S1E4M3, fp.S1E4M3), torch.float8_e4m3fnuz, 32)
    run_config('cdna3.bf8       L16 K32', GTR,
               amd.make_gtr_fdpa(16, fp.S1E5M2, fp.S1E5M2), torch.float8_e5m2fnuz, 32)

###########################################################

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials', type=int, default=1000,
                        help='trials per configuration in the sweep (default 1000)')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)

    check_table8()
    check_directed()
    check_fma(args.trials)

    try:
        run_sweep(args.trials)
    except ImportError as e:
        print(f'---- differential sweep skipped ({e}); pip install -r requirements.txt')

    if FAILURES:
        print(f'{FAILURES} check(s) FAILED')
        sys.exit(1)
    print('all checks passed')

if __name__ == '__main__':
    main()
