"""
Validation for the AMD MMA models in `amd.py`.

Two tiers:

1. Directed checks (need only `fpy2`): the paper's Eq. 10 / Table 8
   example, special values, FTZ flushing semantics, and the CDNA3
   round-down behavior. (The shared FMA chain is checked by
   `validate_nv.py`.)

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
import os
import random
import sys

# `amd.py` lives in the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fpy2 as fp
import amd

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
    for name, thunk, expected in amd.table8_cases():
        got = thunk()
        check(f'table8 {name}', got == expected, f'got {got}, expected {expected}')

def check_directed():
    e2 = amd.make_e_fdpa(2)
    check('special NaN input', fp.isnan(e2([fp.nan(), 0.0], [1.0, 0.0], 0.0)))
    check('special inf * 0', fp.isnan(e2([float('inf'), 0.0], [0.0, 0.0], 0.0)))
    check('special inf - inf',
          fp.isnan(e2([float('inf'), 1.0], [1.0, float('-inf')], 0.0)))
    check('special c = inf', float(e2([1.0, 1.0], [1.0, 1.0], float('inf'))) == float('inf'))

    # E-FDPA: an exactly-zero sum is always +0.0, even for
    # all-negative-zero summands; only a negative sum that underflows
    # in the output rounding yields -0.0
    d = e2([-0.0, 0.0], [1.0, -1.0], -0.0)
    check('e-fdpa zero sum is +0', d == 0 and not fp.signbit(d))
    d = e2([-2.0**-133, 0.0], [2.0**-126, 0.0], 0.0)
    check('e-fdpa negative underflow is -0', d == 0 and fp.signbit(d))

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

###########################################################
# Tier 2: randomized differential sweep vs MMA-Sim (needs torch)

def run_sweep(trials):
    import torch  # type: ignore[import-not-found]
    from mmasim.arithmetic import fdpa, ftz_mul_add  # type: ignore[import-not-found]
    import validate_common as vc

    torch.manual_seed(0)

    def run_config(name, op, model, a_dtype, K, tf32=False):
        fails = 0
        for i in range(trials):
            gen = vc.GENS[i % len(vc.GENS)]
            a, b = gen(K, a_dtype), gen(K, a_dtype)
            c = gen(1, torch.float32)[0]
            ref = op.dpa(a.clone(), b.clone(), c.clone())
            if tf32:  # xf32: truncate inputs on the FPy side
                af, bf = vc.tf32_list(a), vc.tf32_list(b)
            else:
                af, bf = a.float().tolist(), b.float().tolist()
            mine = model(af, bf, float(c.float()))
            if not vc.same(ref, mine):
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
