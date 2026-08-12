"""
Validation for the NVIDIA MMA models in `nv.py`.

Two tiers:

1. Directed checks (need only `fpy2`): the paper's Eq. 10 / Table 8
   example, special values, signed zeros, the `e_zero` tie-breaking
   corner case, and the FMA chain against `math.fma`.

2. A randomized differential sweep against the MMA-Sim reference
   implementation (https://github.com/microsoft/MMA-Sim), comparing
   every model bit-for-bit over five input distributions: random
   bit-patterns, edge values, zero-mixed bit-patterns, standard
   normal, and normal scaled so that 1 becomes 2^emin. This tier
   needs the `mmasim` package (see requirements.txt):

       pip install -r requirements.txt
       python validate_nv.py [--trials N]
"""

import argparse
import math
import os
import random
import struct
import sys

# `nv.py` lives in the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fpy2 as fp
import nv

FAILURES = 0

def check(name, ok, detail=''):
    global FAILURES
    print(f'{"ok  " if ok else "FAIL"} {name}{"" if ok else "  " + detail}')
    if not ok:
        FAILURES += 1

###########################################################
# Tier 1: directed checks (fpy2 only)

def check_table8():
    for name, thunk, expected in nv.table8_cases():
        got = thunk()
        check(f'table8 {name}', got == expected, f'got {got}, expected {expected}')

def check_directed():
    m24 = nv.make_t_fdpa(fp.FP16, fp.FP16, fp.FP32, 24, nv.RZ_FP32)

    check('special NaN input', fp.isnan(m24([fp.nan()], [1.0], 0.0)))
    check('special inf * 0', fp.isnan(m24([float('inf')], [0.0], 0.0)))
    check('special inf - inf', fp.isnan(m24([float('inf'), 1.0], [1.0, float('-inf')], 0.0)))
    check('special c = inf', float(m24([1.0], [1.0], float('inf'))) == float('inf'))

    d = m24([-0.0, 0.0], [1.0, -1.0], -0.0)
    check('signed zero: all -0 sums to -0', d == 0 and fp.signbit(d))
    d = m24([0.0, 0.0], [1.0, -1.0], -0.0)
    check('signed zero: mixed 0 sums to +0', d == 0 and not fp.signbit(d))

    # e_zero raises the alignment maximum: with a zero product and
    # c = 0, the 2^-48 term is truncated and the sum lands exactly on
    # an RNE-FP16 tie (0.0); with a lower e_zero it survives and
    # breaks the tie (2^-24). Verified against the MMA-Sim reference.
    A = [2.0**-14, 2.0**-24, 0.0]
    B = [2.0**-11, 2.0**-24, 1.0]
    m = nv.make_t_fdpa(fp.FP16, fp.FP16, fp.FP16, 24, nv.RNE_FP16, e_zero=-21)
    check('e_zero tie case (e_zero=-21)', float(m(A, B, 0.0)) == 0.0)
    m = nv.make_t_fdpa(fp.FP16, fp.FP16, fp.FP16, 24, nv.RNE_FP16, e_zero=-29)
    check('e_zero tie case (e_zero=-29)', float(m(A, B, 0.0)) == 2.0**-24)

def check_fma(trials):
    from fractions import Fraction

    def rand_f64():
        return struct.unpack('<d', random.randbytes(8))[0]

    model = nv.make_fma_dpa(fp.FP64)
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
    from mmasim.arithmetic import fdpa  # type: ignore[import-not-found]
    from mmasim.arithmetic.helper import truncate_e4m3_to_ue4m3  # type: ignore[import-not-found]
    import validate_common as vc

    torch.manual_seed(0)

    RHO = {'RZ-FP32': nv.RZ_FP32, 'RZ-E8M13': nv.RZ_E8M13,
           'RNE-FP32': nv.RNE_FP32, 'RNE-FP16': nv.RNE_FP16}
    CTX = {torch.float16: fp.FP16, torch.bfloat16: fp.BF16,
           torch.float32: fp.TF32,  # float32 inputs mean TF32 instructions
           torch.float8_e4m3fn: fp.MX_E4M3, torch.float8_e5m2: fp.MX_E5M2}
    C_CTX = {torch.float16: fp.FP16, torch.float32: fp.FP32}

    def run_t_fdpa(name, a_dtype, c_dtype, L, K, F, rho, e_zero):
        op = fdpa.MMA_T_FDPA(F, rho, L, e_zero)
        model = nv.make_t_fdpa_chain(L, CTX[a_dtype], CTX[a_dtype], C_CTX[c_dtype],
                                     F, RHO[rho], e_zero=e_zero)
        fails = 0
        for i in range(trials):
            gen = vc.GENS[i % len(vc.GENS)]
            a, b, c = gen(K, a_dtype), gen(K, a_dtype), gen(1, c_dtype)[0]
            ref = op.dpa(a.clone(), b.clone(), c.clone())
            if a_dtype == torch.float32:  # tf32: truncate inputs on the FPy side
                af, bf = vc.tf32_list(a), vc.tf32_list(b)
            else:
                af, bf = a.float().tolist(), b.float().tolist()
            mine = model(af, bf, float(c.float()))
            if not vc.same(ref, mine, c_dtype if rho.endswith('FP16') else torch.float32):
                fails += 1
        check(f'sweep {name}: {trials - fails}/{trials}', fails == 0)

    def run_st_fdpa(name, a_dtype, L, F, rho, e_zero):
        model = nv.make_st_fdpa(CTX[a_dtype], CTX[a_dtype], fp.MX_E8M0,
                                F, RHO[rho], e_zero=e_zero)
        fails = 0
        for i in range(trials):
            gen = vc.GENS[i % len(vc.GENS)]
            a, b = gen(L, a_dtype), gen(L, a_dtype)
            c = vc.rand_bits(1, torch.float32)[0]
            alpha = vc.rand_bits(1, torch.float8_e8m0fnu)
            beta = vc.rand_bits(1, torch.float8_e8m0fnu)
            ref = fdpa.st_fdpa(a.clone(), b.clone(), c.clone(),
                               alpha.clone(), beta.clone(), F, rho, e_zero)
            mine = model(a.float().tolist(), b.float().tolist(), float(c.float()),
                         float(alpha.float()[0]), float(beta.float()[0]))
            if not vc.same(ref, mine):
                fails += 1
        check(f'sweep {name}: {trials - fails}/{trials}', fails == 0)

    FP4_VALS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]

    def run_gst_fdpa(name, scale_dtype, K, K_block, G, F, rho, e_zero):
        n_scales = K // K_block
        op = fdpa.MMA_GST_FDPA(G, F, rho, K, e_zero)
        scale_ctx = fp.MX_E4M3 if scale_dtype == torch.float8_e4m3fn else fp.MX_E8M0
        model = nv.make_gst_fdpa(G, scale_ctx, F, RHO[rho], e_zero=e_zero)
        fails = 0
        for i in range(trials):
            zero_heavy = i % 3 == 2
            vals = [0.0, -0.0, 0.0, 0.5, -0.5] if zero_heavy else FP4_VALS
            a = [random.choice(vals) for _ in range(K)]
            b = [random.choice(vals) for _ in range(K)]
            alpha = vc.rand_bits(n_scales, scale_dtype)
            beta = vc.rand_bits(n_scales, scale_dtype)
            c = vc.rand_bits(1, torch.float32)[0]
            ref = op.dpa(torch.tensor(a, dtype=torch.float16),
                         torch.tensor(b, dtype=torch.float16),
                         c.clone(), alpha.clone(), beta.clone())
            # apply the ue4m3 truncation and per-group replication ourselves
            if scale_dtype == torch.float8_e4m3fn:
                alpha_f = truncate_e4m3_to_ue4m3(alpha).float().tolist()
                beta_f = truncate_e4m3_to_ue4m3(beta).float().tolist()
            else:
                alpha_f, beta_f = alpha.float().tolist(), beta.float().tolist()
            rep = K_block // G
            alphas = [x for x in alpha_f for _ in range(rep)]
            betas = [x for x in beta_f for _ in range(rep)]
            mine = model(a, b, float(c.float()), alphas, betas)
            if not vc.same(ref, mine):
                fails += 1
        check(f'sweep {name}: {trials - fails}/{trials}', fails == 0)

    # T-FDPA configs (arch, F, rho, e_zero per MMA-Sim sim.py)
    run_t_fdpa('volta.f16.f32   L4 K4  F23', torch.float16, torch.float32, 4, 4, 23, 'RZ-FP32', -131)
    run_t_fdpa('volta.f16.f16   L4 K8  F23', torch.float16, torch.float16, 4, 8, 23, 'RNE-FP16', -20)
    run_t_fdpa('ampere.f16.f32  L8 K16 F24', torch.float16, torch.float32, 8, 16, 24, 'RZ-FP32', -132)
    run_t_fdpa('ampere.f16.f16  L8 K16 F24', torch.float16, torch.float16, 8, 16, 24, 'RNE-FP16', -21)
    run_t_fdpa('ampere.bf16.f32 L8 K16 F24', torch.bfloat16, torch.float32, 8, 16, 24, 'RZ-FP32', -132)
    run_t_fdpa('ampere.tf32.f32 L4 K8  F24', torch.float32, torch.float32, 4, 8, 24, 'RZ-FP32', -132)
    run_t_fdpa('hopper.f16.f32  L16 K32 F25', torch.float16, torch.float32, 16, 32, 25, 'RZ-FP32', -133)
    run_t_fdpa('hopper.f16.f16  L16 K16 F25', torch.float16, torch.float16, 16, 16, 25, 'RNE-FP16', -133)
    run_t_fdpa('ada.e4m3.f32    L16 K32 F13', torch.float8_e4m3fn, torch.float32, 16, 32, 13, 'RZ-E8M13', -132)
    run_t_fdpa('ada.e5m2.f16    L16 K16 F13', torch.float8_e5m2, torch.float16, 16, 16, 13, 'RNE-FP16', -21)
    run_t_fdpa('hopper.e4m3.f32 L32 K32 F13', torch.float8_e4m3fn, torch.float32, 32, 32, 13, 'RZ-E8M13', -133)
    run_t_fdpa('blackwell.f16.f32 L16 K32 F25', torch.float16, torch.float32, 16, 32, 25, 'RZ-FP32', -133)

    # ST-FDPA (MXFP8)
    run_st_fdpa('mxfp8.e4m3      L32 F25', torch.float8_e4m3fn, 32, 25, 'RZ-FP32', -133)
    run_st_fdpa('mxfp8.e5m2      L32 F25', torch.float8_e5m2, 32, 25, 'RZ-FP32', -133)

    # GST-FDPA (NVFP4 / MXFP4)
    run_gst_fdpa('nvfp4 (ue4m3)   K64 G16', torch.float8_e4m3fn, 64, 16, 16, 35, 'RZ-FP32', -139)
    run_gst_fdpa('mxfp4 (e8m0)    K64 G16', torch.float8_e8m0fnu, 64, 32, 16, 35, 'RZ-FP32', -139)

###########################################################

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials', type=int, default=1000,
                        help='trials per configuration in the sweep (default 1000)')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    seed: int = args.seed
    num_trials: int = args.trials

    random.seed(seed)

    check_table8()
    check_directed()
    check_fma(num_trials)

    try:
        run_sweep(num_trials)
    except ImportError as e:
        print(f'---- differential sweep skipped ({e}); pip install -r requirements.txt')

    if FAILURES:
        print(f'{FAILURES} check(s) FAILED')
        sys.exit(1)
    print('all checks passed')

if __name__ == '__main__':
    main()
