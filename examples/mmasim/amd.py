"""
Bit-accurate models of AMD Matrix Core MMA arithmetic, from MMA-Sim:

    P. Xie, S. Xu, Y. Wang, F. Yang, M. Yang.
    "Bit-Accurate Modeling of GPU Matrix Multiply-Accumulate Units:
    Demystifying Numerical Discrepancy and Accuracy". arXiv:2511.10909.
    Reference implementation: https://github.com/microsoft/MMA-Sim

Like `nv.py`, the models compute one dot-product-accumulate
`d = c + sum_k a_k * b_k`; the accumulator is FP32 (FP64 for the FMA
models). Instruction-to-model mapping (Table 6):

    arch    inputs              model              parameters
    -----------------------------------------------------------------
    CDNA1   FP32                nv.make_fma_dpa    fp.FP32
    CDNA1   BF16 / FP16         make_e_fdpa        L = 2 / 4
    CDNA2   FP64, FP32          nv.make_fma_dpa    fp.FP64 / fp.FP32
    CDNA2   BF16 (w/o _1k)      make_ftz_addmul    P = 2
    CDNA2   BF16 (_1k), FP16    make_ftz_addmul    P = 4
    CDNA3   FP64, FP32          nv.make_fma_dpa    fp.FP64 / fp.FP32
    CDNA3   TF32 (xf32)         make_tr_fdpa       L = 4  (Table 7)
    CDNA3   BF16 / FP16         make_tr_fdpa       L = 8
    CDNA3   FP8                 make_gtr_fdpa      L = 16

TR-FDPA / GTR-FDPA parameters (Table 7): F = 24, F2 = 31,
rho = RNE_FP32 for every input type. CDNA3 FP8 means the "fnuz"
formats: `fp.S1E4M3` (fp8) and `fp.S1E5M2` (bf8). TF32 instructions
truncate their FP32 operands to TF32 (`nv.RZ_TF32`) first.

Notes:

- CDNA2's FTZ path flushes *input* subnormals to +0.0 (the sign is
  dropped) and subnormal *results* to +/-0.0 (Algorithm 1/2).
- TR-FDPA and GTR-FDPA round the product sum and the accumulator
  toward negative infinity (RD) -- the asymmetric rounding of
  Sec. 6.2.4 -- and TR-FDPA products overflow to infinity at 2^128
  (Sec. 4.2).
- The paper's Algorithm 10 (line 14) shows a single RD at F2 relative
  to E; the reference implementation rounds down at F2 + 1, adds the
  accumulator, renormalizes, and rounds down again at F2. The models
  here follow the reference implementation.
"""

import fpy2 as fp

from utils import (exponent, fused_sum, join, make_fma_dpa,
                   round_down_at, sum_special_values)

RNE_FP32 = fp.FP32

EMIN_FP32 = fp.FP32.emin

# exponent of a zero product: below any real product exponent, so
# unlike T-FDPA a zero never raises the alignment maximum
E_ZERO_TR = -999

###########################################################
# Helpers

@fp.fpy(ctx=fp.REAL)
def exponent0(x, emin):
    """
    `exponent`, as the reference implementation reads it via `frexp`
    (normalized so 1 <= |mantissa| < 2): a NaN or infinity reads
    exponent -1, independent of its payload.
    """
    if not fp.isfinite(x):
        return -1
    return exponent(x, emin)

@fp.fpy(ctx=fp.REAL)
def overflow_inf(p):
    """TR-FDPA products overflow to infinity at 2^128 (Sec. 4.2)."""
    if abs(p) >= 2 ** 128:
        return -fp.inf() if fp.signbit(p) else fp.inf()
    return p

@fp.fpy(ctx=fp.REAL)
def flush_subnormal(x, tiny):
    """Flushes `x` to +0.0 (the sign is dropped) if `|x| < tiny`."""
    return 0 if abs(x) < tiny else x

@fp.fpy(ctx=fp.REAL)
def flush_subnormal_signed(x, tiny):
    """Flushes `x` to +/-0.0 (the sign is kept) if `|x| < tiny`."""
    return x * 0 if abs(x) < tiny else x

###########################################################
# Model factories

def make_e_fdpa(l_max: int):
    """
    Builds a Phi_E-FDPA dot-product-accumulate (Algorithm 6), chained
    L = min(K, Lmax) elements at a time; CDNA1 BF16 (L = 2) and FP16
    (L = 4) MFMA instructions.

    Each block computes `c + sum a_k * b_k` exactly and rounds once
    with RNE-FP32.
    """
    @fp.fpy(ctx=fp.REAL)
    def e_fdpa_block(A, B, c):
        """Exact fused dot-product-add (Algorithm 6)."""
        prods = [a * b for a, b in zip(A, B)]
        if any([not fp.isfinite(p) for p in prods]) or not fp.isfinite(c):
            return sum_special_values(prods, c)

        s = sum(join(prods, [c]))
        with RNE_FP32:
            return fp.round(s)

    @fp.fpy(ctx=fp.REAL)
    def e_fdpa(A, B, c):
        """Chain of E-FDPAs (Algorithm 5)."""
        k = len(A)
        l = min(k, l_max)

        d = c
        for i in range(0, k, l):
            hi = min(i + l, k)
            d = e_fdpa_block(A[i:hi], B[i:hi], d)
        return d
    return e_fdpa

def make_ftz_addmul(a_ctx: fp.EFloatContext, P: int):
    """
    Builds a Phi_FTZ-AddMul dot-product-accumulate (Algorithm 2);
    CDNA2 BF16 (P = 2, or 4 with the `_1k` suffix) and FP16 (P = 4)
    MFMA instructions. Assumes P divides K.

    Every P consecutive products are summed pairwise with FTZ-Add and
    accumulated sequentially. Input subnormals (including the initial
    accumulator) flush to +0.0; subnormal FP32 results flush to
    +/-0.0.
    """
    tiny_in = 2.0 ** a_ctx.emin    # smallest normal of the input format
    tiny_out = 2.0 ** EMIN_FP32    # smallest normal FP32

    @fp.fpy(ctx=fp.REAL)
    def ftz_mul(x, y):
        """FTZ-Mul (Algorithm 1)."""
        x = flush_subnormal(x, tiny_in)
        y = flush_subnormal(y, tiny_in)
        with RNE_FP32:
            z = fp.round(x * y)
        return flush_subnormal_signed(z, tiny_out)

    @fp.fpy(ctx=fp.REAL)
    def ftz_add(x, y):
        """FTZ-Add (Algorithm 1)."""
        with RNE_FP32:
            z = fp.round(x + y)
        return flush_subnormal_signed(z, tiny_out)

    @fp.fpy(ctx=fp.REAL)
    def ftz_block(A, B, c):
        """One P-wide pairwise summation step of Algorithm 2."""
        s = ftz_add(ftz_mul(A[0], B[0]), ftz_mul(A[1], B[1]))
        if P == 4:
            s2 = ftz_add(ftz_mul(A[2], B[2]), ftz_mul(A[3], B[3]))
            s = ftz_add(s, s2)
        return ftz_add(c, s)

    @fp.fpy(ctx=fp.REAL)
    def ftz_addmul(A, B, c):
        """Pairwise summation and accumulation (Algorithm 2)."""
        # only the *input* accumulator is flushed sign-dropping;
        # intermediate accumulators are FTZ-Add outputs
        d = flush_subnormal(c, tiny_out)
        for i in range(0, len(A), P):
            d = ftz_block(A[i:i + P], B[i:i + P], d)
        return d

    return ftz_addmul

def make_tr_fdpa(l_max: int, a_ctx: fp.EFloatContext, b_ctx: fp.EFloatContext,
                 F: int = 24, F2: int = 31, rho: fp.Context = RNE_FP32):
    """
    Builds a Phi_TR-FDPA dot-product-accumulate (Algorithm 10),
    chained L = min(K, Lmax) elements at a time; CDNA3 TF32 (L = 4)
    and BF16/FP16 (L = 8) MFMA instructions.

    Each block computes the products exactly (overflowing to infinity
    at 2^128), truncates them toward zero at F fractional bits below
    their maximum exponent, and sums exactly. The product sum and the
    accumulator are then combined with round-down (RD) rounding: the
    sum at F2 + 1 fractional bits, the accumulator at F, and their
    total -- renormalized -- at F2, before the RNE-FP32 output
    conversion.
    """
    emin_a = a_ctx.emin
    emin_b = b_ctx.emin

    @fp.fpy(ctx=fp.REAL)
    def tr_fdpa_block(A, B, c):
        """Truncated rounded fused dot-product-add (Algorithm 10)."""
        # Step 1: exact products; |p| >= 2^128 overflows to infinity
        prods = [overflow_inf(a * b) for a, b in zip(A, B)]
        if any([not fp.isfinite(p) for p in prods]) or not fp.isfinite(c):
            return sum_special_values(prods, c)

        # Step 2: truncated fused sum of the L products (without c)
        es = [E_ZERO_TR if a * b == 0 else exponent(a, emin_a) + exponent(b, emin_b)
              for a, b in zip(A, B)]
        e_dot = max(es)
        t = fused_sum(prods, e_dot - F - 1, fp.RM.RTZ)

        # Step 3: rounded (round-down) sum of the product sum and c
        e_c = exponent(c, EMIN_FP32)
        e = max(e_dot, e_c)
        t = round_down_at(t, e - F2 - 2)   # F2 + 1 fractional bits
        cr = round_down_at(c, e - F - 1)   # F fractional bits
        s = t + cr

        # renormalize and round down to F2 fractional bits
        s = round_down_at(s, exponent(s, EMIN_FP32) - F2 - 1)

        # Step 4: convert to the output format
        with rho:
            return fp.round(s)

    @fp.fpy(ctx=fp.REAL)
    def tr_fdpa(A, B, c):
        """Chain of TR-FDPAs (Algorithm 5)."""
        k = len(A)
        l = min(k, l_max)

        d = c
        for i in range(0, k, l):
            hi = min(i + l, k)
            d = tr_fdpa_block(A[i:hi], B[i:hi], d)
        return d
    return tr_fdpa

def make_gtr_fdpa(l_max: int, a_ctx: fp.EFloatContext, b_ctx: fp.EFloatContext,
                  F: int = 24, F2: int = 31, rho: fp.Context = RNE_FP32):
    """
    Builds a Phi_GTR-FDPA dot-product-accumulate (Algorithm 11),
    chained L = min(K, Lmax) elements at a time; CDNA3 FP8 (L = 16,
    `fp.S1E4M3`/`fp.S1E5M2`) MFMA instructions.

    Like TR-FDPA, but the truncated fused sum is computed separately
    over the even-index and odd-index products; the two group sums are
    combined with round-down rounding at F fractional bits, and the
    accumulator is forced to zero when `e_c < E - F - 1` (the paper's
    "special truncation").
    """
    emin_a = a_ctx.emin
    emin_b = b_ctx.emin

    @fp.fpy(ctx=fp.REAL)
    def gtr_fdpa_block(A, B, c):
        """Group-truncated rounded fused dot-product-add (Algorithm 11)."""
        # Step 1: exact products and their exponent-field sums
        # (a NaN or infinite factor reads exponent 0)
        prods = [a * b for a, b in zip(A, B)]
        es = [E_ZERO_TR if a * b == 0 else exponent0(a, emin_a) + exponent0(b, emin_b)
              for a, b in zip(A, B)]
        l = len(A)
        e_even = max([es[i] for i in range(0, l, 2)])
        e_odd = max([es[i] for i in range(1, l, 2)])

        # a NaN or infinite accumulator also reads exponent -1, so the
        # special truncation of Step 4 drops it whenever the product
        # sum's exponent exceeds F
        if not fp.isfinite(c) and max(e_even, e_odd) > F:
            c = 0

        if any([not fp.isfinite(p) for p in prods]) or not fp.isfinite(c):
            return sum_special_values(prods, c)

        # Step 2: truncated fused sums of the even- and odd-index products
        t_even = fused_sum([prods[i] for i in range(0, l, 2)], e_even - F - 1, fp.RM.RTZ)
        t_odd = fused_sum([prods[i] for i in range(1, l, 2)], e_odd - F - 1, fp.RM.RTZ)

        # Step 3: rounded (round-down) sum of the two group sums
        e_dot = max(e_even, e_odd)
        t = round_down_at(t_even, e_dot - F - 1) + round_down_at(t_odd, e_dot - F - 1)

        # Step 4: final rounded sum with c ("special truncation" forces
        # a far-below accumulator to zero)
        e_c = exponent(c, EMIN_FP32)
        e = max(e_dot, e_c)
        t = round_down_at(t, e - F2 - 2)   # F2 + 1 fractional bits
        cr = 0 if e_c < e - F - 1 else round_down_at(c, e - F - 1)
        s = t + cr

        # renormalize and round down to F2 fractional bits
        s = round_down_at(s, exponent(s, EMIN_FP32) - F2 - 1)

        # Step 5: convert to the output format
        with rho:
            return fp.round(s)

    @fp.fpy(ctx=fp.REAL)
    def gtr_fdpa(A, B, c):
        """Chain of GTR-FDPAs (Algorithm 5)."""
        k = len(A)
        l = min(k, l_max)

        d = c
        for i in range(0, k, l):
            hi = min(i + l, k)
            d = gtr_fdpa_block(A[i:hi], B[i:hi], d)
        return d
    return gtr_fdpa


if __name__ == "__main__":
    # Run the paper's Eq. 10 example through every model, reproducing
    # the AMD rows of Table 8. The exact answer is -0.875.
    def pad(xs, k):
        return xs + [0.0] * (k - len(xs))

    a = [-2.0**13, -0.5, -0.25, -0.125]
    b = [2.0**10, 1.0, 1.0, 1.0]
    c = 2.0**23

    models = [
        # name                     model                                  K     expected
        ('fp64 (fma)',             make_fma_dpa(fp.FP64),                 4),   # -0.875
        ('cdna1.bf16 (e-fdpa)',    make_e_fdpa(2),                        4),   # -0.875
        ('cdna1.f16 (e-fdpa)',     make_e_fdpa(4),                        4),   # -0.875
        ('cdna2.bf16 (ftz, P=2)',  make_ftz_addmul(fp.BF16, 2),           4),   # -0.375
        ('cdna2.bf16_1k (ftz, P=4)', make_ftz_addmul(fp.BF16, 4),         4),   # 0.0
        ('cdna2.f16 (ftz, P=4)',   make_ftz_addmul(fp.FP16, 4),           4),   # 0.0
        ('cdna3.f16 (tr-fdpa)',    make_tr_fdpa(8, fp.FP16, fp.FP16),     8),   # -0.5
        ('cdna3.bf16 (tr-fdpa)',   make_tr_fdpa(8, fp.BF16, fp.BF16),     8),   # -0.5
        ('cdna3.bf8 (gtr-fdpa)',   make_gtr_fdpa(16, fp.S1E5M2, fp.S1E5M2), 16), # -1.0
    ]
    for name, model, k in models:
        print(f'{name:28s} {float(model(pad(a, k), pad(b, k), c)):+.3f}')
