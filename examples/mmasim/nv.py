"""
Bit-accurate models of NVIDIA Tensor Core MMA arithmetic, from MMA-Sim:

    P. Xie, S. Xu, Y. Wang, F. Yang, M. Yang.
    "Bit-Accurate Modeling of GPU Matrix Multiply-Accumulate Units:
    Demystifying Numerical Discrepancy and Accuracy". arXiv:2511.10909.
    Reference implementation: https://github.com/microsoft/MMA-Sim

Every output element of an MMA `D = A x B + C` is computed independently
as a dot-product-accumulate `d = c + sum_k a_k * b_k` (paper, Sec. 3.1.1),
so the models here are written at that level.

Instruction-to-model mapping on NVIDIA Tensor Cores (Table 3):

    FP64            -> `fma_dpa`   (chain of standard FMAs)
    TF32/BF16/FP16  -> `t_fdpa`    (truncated FDPA)
    FP8, FP6/FP4    -> `t_fdpa`
    MXFP8/6/4       -> `st_fdpa`   (scaled truncated FDPA)
    MXFP4/NVFP4     -> `gst_fdpa`  (group-scaled truncated FDPA)

When K exceeds the FDPA arity `Lmax`, FDPAs are chained (`t_fdpa_chain`).
TF32 instructions truncate their FP32 operands to TF32 (`RZ_TF32`)
before the dot product.

T-FDPA / ST-FDPA parameters by architecture (paper Table 4; `e_zero`
from the MMA-Sim reference implementation, keyed by the accumulator
type: FP32 / FP16):

    architecture    inputs      output  Lmax  F   rho       e_zero
    ----------------------------------------------------------------
    Volta           FP16        FP32    4     23  RZ_FP32   -131
    Volta           FP16        FP16    4     23  RNE_FP16  -20
    Turing          FP16        FP32    8     24  RZ_FP32   -132
    Turing          FP16        FP16    8     24  RNE_FP16  -21
    Ampere/Ada      TF32        FP32    4     24  RZ_FP32   -132
    Ampere/Ada      BF16/FP16   FP32    8     24  RZ_FP32   -132
    Ampere/Ada      FP16        FP16    8     24  RNE_FP16  -21
    Ada             FP8         FP32    16    13  RZ_E8M13  -132
    Ada             FP8         FP16    16    13  RNE_FP16  -21
    Hopper          TF32        FP32    8     25  RZ_FP32   -133
    Hopper          BF16/FP16   FP32    16    25  RZ_FP32   -133
    Hopper          FP16        FP16    16    25  RNE_FP16  -133 (wgmma) / -22 (mma)
    Hopper          FP8         FP32    32    13  RZ_E8M13  -133
    Hopper          FP8         FP16    32    13  RNE_FP16  -133
    Blackwell/RTX   TF32        FP32    8     25  RZ_FP32   -133
    Blackwell/RTX   BF16/FP16   FP32    16    25  RZ_FP32   -133
    Blackwell/RTX   FP16        FP16    16    25  RNE_FP16  -133 (tcgen05) / -22 (mma)
    Blackwell/RTX   FP8/6/4     FP32    32    25  RZ_FP32   -133
    Blackwell/RTX   FP8/6/4     FP16    32    25  RNE_FP16  -133 (tcgen05) / -22 (mma)
    Blackwell/RTX   MXFP8/6/4   FP32    32    25  RZ_FP32   -133  (ST-FDPA)

GST-FDPA parameters (Table 5):

    Blackwell/RTX, MXFP4/NVFP4: L = 64, G = 16, F = 35,
    rho = RZ_FP32, e_zero = -139
"""

import fpy2 as fp

# conversion functions `rho` (Table 2); on overflow the hardware
# produces infinity even in RZ mode, see `fdpa_round`
RZ_FP32 = fp.IEEEContext(8, 32, fp.RM.RTZ)
RZ_E8M13 = fp.IEEEContext(8, 22, fp.RM.RTZ)
RNE_FP32 = fp.FP32
RNE_FP16 = fp.FP16

# input conversion applied by TF32 instructions to their FP32 operands
RZ_TF32 = fp.IEEEContext(8, 19, fp.RM.RTZ)

# emin of the input formats, for `exponent`
EMIN_FP32 = fp.FP32.emin
EMIN_TF32 = fp.TF32.emin
EMIN_BF16 = fp.BF16.emin
EMIN_FP16 = fp.FP16.emin
EMIN_E4M3 = fp.MX_E4M3.emin
EMIN_E5M2 = fp.MX_E5M2.emin
EMIN_E3M2 = fp.MX_E3M2.emin
EMIN_E2M3 = fp.MX_E2M3.emin
EMIN_E2M1 = fp.MX_E2M1.emin
EMIN_UE4M3 = fp.MX_E4M3.emin  # NVFP4 scale format (E4M3 minus the sign)
EMIN_E8M0 = fp.MX_E8M0.emin   # MXFP scale format (never subnormal)


@fp.fpy(ctx=fp.REAL)
def join(xs, ys):
    """
    Concatenates two lists.

    FPy has no list concatenation, so preallocate and copy.
    """
    n = len(xs)
    m = len(ys)
    zs = fp.empty(n + m)
    for i in range(n):
        zs[i] = xs[i]
    for i in range(m):
        zs[n + i] = ys[i]
    return zs

@fp.fpy(ctx=fp.REAL)
def vec_sum(xs):
    """
    Exact sum of a non-empty list.

    Folds from the first element rather than from 0 so that IEEE
    signed-zero rules apply: a sum of all `-0.0` terms is `-0.0`.
    """
    acc = xs[0]
    for x in xs[1:]:
        acc += x
    return acc

@fp.fpy(ctx=fp.REAL)
def fused_sum(xs, n, rm):
    """
    Fused summation of a non-empty list.

    Each summand is rounded with mode `rm` onto the fixed-point grid
    whose first unrepresentable digit is `n` (so the least significant
    representable digit has exponent `exp = n + 1`), then the rounded
    summands are added exactly.
    """
    ctx = fp.MPFixedContext(n, rm)
    with ctx:
        acc = fp.round(xs[0])
    for x in xs[1:]:
        with ctx:
            t = fp.round(x)
        acc += t
    return acc

@fp.fpy(ctx=fp.REAL)
def exponent(x, emin):
    """
    The paper's `Exp(x)`: the exponent field of `x` in a floating-point
    format with minimum normalized exponent `emin`. Subnormal values
    (including zero) read as `emin`.
    """
    return max(fp.logb(x), emin)

@fp.fpy(ctx=fp.REAL)
def fdpa_round(s, rho):
    """
    Converts an exact FDPA sum `s` to the output format with `rho`.

    Unlike IEEE round-toward-zero, the hardware conversions overflow
    to infinity, never to the maximum finite value.
    """
    if abs(s) >= 2 ** 128:
        return -fp.inf() if fp.signbit(s) else fp.inf()
    with rho:
        return fp.round(s)

@fp.fpy(ctx=fp.REAL)
def dot_prod_special_values(A, B):
    """
    Result of a dot product whose inputs contain a NaN or an infinity:
    NaN if any input is NaN, any product is `inf * 0`, or two infinite
    products disagree in sign; otherwise the common infinity.
    """
    # any NaN => produce NaN
    if any([fp.isnan(a) for a in A]):
        return fp.nan()
    if any([fp.isnan(b) for b in B]):
        return fp.nan()

    # any infinity => produce either infinity or NaN
    has_inf = False
    inf_sgn = False
    for a, b in zip(A, B):
        if fp.isinf(a) or fp.isinf(b):
            # actually compute the product
            t = a * b
            if fp.isnan(t):
                return t

            t_sgn = fp.signbit(t)
            if has_inf:
                # if we have already seen an infinity, check that the sign is consistent
                if inf_sgn != t_sgn:
                    return fp.nan()
            else:
                # if this is the first infinity, record its sign
                has_inf = True
                inf_sgn = t_sgn

    assert has_inf, "expected either NaN or infinity in the input"
    return -fp.inf() if inf_sgn else fp.inf()

@fp.fpy(ctx=fp.REAL)
def dpa_special_values(A, B, c):
    """
    Result of a dot-product-accumulate whose inputs contain a NaN or an
    infinity (Sec. 4.2): `NaN + x = NaN`, `+/-inf + y = +/-inf`,
    `inf - inf = NaN`, and `inf * 0 = NaN`.
    """
    if fp.isnan(c):
        return fp.nan()

    if any([not fp.isfinite(a) for a in A]) or any([not fp.isfinite(b) for b in B]):
        s = dot_prod_special_values(A, B)
        if fp.isnan(s):
            return s
        # s is +/-inf; a conflicting infinite accumulator produces NaN
        if fp.isinf(c) and fp.signbit(c) != fp.signbit(s):
            return fp.nan()
        return s

    # products are all finite, c is +/-inf
    return c

@fp.fpy(ctx=fp.REAL)
def fma_dpa(A, B, c, ctx):
    """
    Phi_FMA (Algorithm 4): a chain of standard FMAs under `ctx`
    (FP64 or FP32, round-to-nearest-even); all FP64 MMA instructions.
    """
    d = c
    for a, b in zip(A, B):
        with ctx:
            d = fp.fma(a, b, d)
    return d

@fp.fpy(ctx=fp.REAL)
def t_fdpa(A, B, c, emin_a, emin_b, emin_c, e_zero, F, rho):
    """
    T-FDPA (Algorithm 7): truncated fused dot-product-add.

    Computes the L products exactly and aligns them and `c` at the
    maximum exponent-field sum `e_max`. Each term is truncated toward
    zero at 2^(e_max - F), i.e. to F fractional bits, summed exactly,
    and converted to the output format with `rho`.

    A zero product (either factor zero) or a zero accumulator reads
    as exponent `e_zero` rather than its format exponent; `e_zero` is
    instruction-specific (see the module docstring).
    """
    if any([not fp.isfinite(a) for a in A]) or any([not fp.isfinite(b) for b in B]) \
            or not fp.isfinite(c):
        return dpa_special_values(A, B, c)

    # Step 1: exact products and their exponent-field sums
    prods = [a * b for a, b in zip(A, B)]
    es = [e_zero if a * b == 0 else exponent(a, emin_a) + exponent(b, emin_b)
          for a, b in zip(A, B)]
    e_c = e_zero if c == 0 else exponent(c, emin_c)

    # Step 2: truncated fused sum of L + 1 terms
    e_max = max(max(es), e_c)
    s = fused_sum(join(prods, [c]), e_max - F - 1, fp.RM.RTZ)

    # Step 3: convert to the output format
    return fdpa_round(s, rho)

@fp.fpy(ctx=fp.REAL)
def t_fdpa_chain(A, B, c, l_max, emin_a, emin_b, emin_c, e_zero, F, rho):
    """
    Phi_FDPA (Algorithm 5): chains T-FDPAs over vectors of length K,
    L = min(K, Lmax) elements at a time.

    Assumes `rho` outputs the accumulator format, so `emin_c` also
    applies to the running accumulator.
    """
    k = len(A)
    l = min(k, l_max)

    d = c
    for i in range(0, k, l):
        hi = min(i + l, k)
        d = t_fdpa(A[i:hi], B[i:hi], d, emin_a, emin_b, emin_c, e_zero, F, rho)
    return d

@fp.fpy(ctx=fp.REAL)
def st_fdpa(A, B, c, alpha, beta, emin_a, emin_b, emin_s, e_zero, F, rho):
    """
    ST-FDPA (Algorithm 8): T-FDPA with block scale factors `alpha` and
    `beta` (E8M0) applied to the products; MXFP8/6/4 MMA instructions.
    The accumulator `c` is FP32.

    Scaling is exact: the scale exponents shift each product's
    alignment exponent, and E8M0 significands are always 1.
    """
    if fp.isnan(alpha) or fp.isnan(beta):
        return fp.nan()
    if any([not fp.isfinite(a) for a in A]) or any([not fp.isfinite(b) for b in B]) \
            or not fp.isfinite(c):
        return dpa_special_values(A, B, c)

    # Step 1: exact scaled products and their exponent-field sums
    prods = [a * b * alpha * beta for a, b in zip(A, B)]
    es = [e_zero if a * b * alpha * beta == 0 else
          exponent(a, emin_a) + exponent(b, emin_b)
          + exponent(alpha, emin_s) + exponent(beta, emin_s)
          for a, b in zip(A, B)]
    e_c = e_zero if c == 0 else exponent(c, EMIN_FP32)

    # Step 2: truncated fused sum of L + 1 terms
    e_max = max(max(es), e_c)
    s = fused_sum(join(prods, [c]), e_max - F - 1, fp.RM.RTZ)

    # Step 3: convert to the output format
    return fdpa_round(s, rho)

@fp.fpy(ctx=fp.REAL)
def gst_fdpa(A, B, c, alphas, betas, G, emin_s, e_zero, F, rho):
    """
    GST-FDPA (Algorithm 9): group-scaled truncated FDPA;
    MXFP4/NVFP4 MMA instructions. The accumulator `c` is FP32.

    The vectors are split into groups of size G. Each group's dot
    product is computed exactly, scaled by its scale factors
    (`alphas[g]`, `betas[g]`, one per group: replicate them if the
    scale block size exceeds G), and aligned using only the scale
    exponents. A group reads as exponent `e_zero` only if all of its
    products are zero or its scale product is zero -- a group that
    merely sums to zero by cancellation keeps its scale exponent.
    `emin_s` is the emin of the scale format (UE4M3 for NVFP4, E8M0
    for MXFP4).
    """
    if any([fp.isnan(s1) for s1 in alphas]) or any([fp.isnan(s2) for s2 in betas]):
        return fp.nan()
    if any([not fp.isfinite(a) for a in A]) or any([not fp.isfinite(b) for b in B]) \
            or not fp.isfinite(c):
        return dpa_special_values(A, B, c)

    # Step 1: exact dot product per group, scaled
    n_groups = len(alphas)
    ts = fp.empty(n_groups)
    es = fp.empty(n_groups)
    for g in range(n_groups):
        prods = [a * b for a, b in zip(A[g * G:(g + 1) * G], B[g * G:(g + 1) * G])]
        scale = alphas[g] * betas[g]
        ts[g] = vec_sum(prods) * scale
        if all([p == 0 for p in prods]) or scale == 0:
            es[g] = e_zero
        else:
            es[g] = exponent(alphas[g], emin_s) + exponent(betas[g], emin_s)
    e_c = e_zero if c == 0 else exponent(c, EMIN_FP32)

    # Step 2: truncated fused sum of L/G + 1 terms
    e_max = max(max(es), e_c)
    s = fused_sum(join(ts, [c]), e_max - F - 1, fp.RM.RTZ)

    # Step 3: convert to the output format
    return fdpa_round(s, rho)
