"""
Bit-accurate models of NVIDIA Tensor Core MMA arithmetic, from MMA-Sim:

    P. Xie, S. Xu, Y. Wang, F. Yang, M. Yang.
    "Bit-Accurate Modeling of GPU Matrix Multiply-Accumulate Units:
    Demystifying Numerical Discrepancy and Accuracy". arXiv:2511.10909.
    Reference implementation: https://github.com/microsoft/MMA-Sim

Every output element of an MMA `D = A x B + C` is computed independently
as a dot-product-accumulate `d = c + sum_k a_k * b_k` (paper, Sec. 3.1.1),
so the models here are written at that level.

Each model is built by calling a `make_XXX` factory with the
per-instruction parameters; input and accumulator formats are given as
contexts (`EFloatContext`, or `ExpContext` for E8M0 scales; only their
`emin` is read). For example, Hopper's FP16 x FP16 + FP32 wgmma
instruction with K = 32 is:

    dpa = make_t_fdpa_chain(16, fp.FP16, fp.FP16, fp.FP32, 25, RZ_FP32)
    d = dpa(a_row, b_col, c)

Instruction-to-model mapping on NVIDIA Tensor Cores (Table 3):

    FP64            -> `make_fma_dpa`    (chain of standard FMAs)
    TF32/BF16/FP16  -> `make_t_fdpa`     (truncated FDPA)
    FP8, FP6/FP4    -> `make_t_fdpa`
    MXFP8/6/4       -> `make_st_fdpa`    (scaled truncated FDPA)
    MXFP4/NVFP4     -> `make_gst_fdpa`   (group-scaled truncated FDPA)

When K exceeds the FDPA arity `Lmax`, FDPAs are chained
(`make_t_fdpa_chain`). TF32 instructions truncate their FP32 operands
to TF32 (`RZ_TF32`) before the dot product.

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

`e_zero` is the exponent an exact-zero term reads out of the internal
alignment datapath. When omitted, the factories derive it as `3 - F`
for FP16-accumulate `mma`-class instructions and `-108 - F` otherwise
(`is_mma=False` covers wgmma/tcgen05, whose FP16-accumulate datapath
is FP32-wide). The derivation tracks each generation's base F, so the
FP8 instructions (F = 13) must pass the architecture's `e_zero`
explicitly, per the table above; GST-FDPA defaults to -139.
"""

import fpy2 as fp

from utils import RZ_TF32, dpa_special_values, exponent, fused_sum, join, make_fma_dpa

###########################################################
# Rounding contexts

# conversion functions `rho` (Table 2); on overflow the hardware
# produces infinity even in RZ mode, see `fdpa_round`
RZ_FP32 = fp.IEEEContext(8, 32, fp.RM.RTZ)
RZ_E8M13 = fp.IEEEContext(8, 22, fp.RM.RTZ)
RNE_FP32 = fp.FP32
RNE_FP16 = fp.FP16

###########################################################
# Helpers

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

###########################################################
# Model factories

def _default_e_zero(c_ctx: fp.EFloatContext, F: int, is_mma: bool) -> int:
    """The `e_zero` of the standard (non-FP8, non-GST) instructions."""
    assert c_ctx.emin in (fp.FP16.emin, fp.FP32.emin), \
        'no known e_zero: the accumulator must be FP16 or FP32'
    if is_mma and c_ctx.emin == fp.FP16.emin:
        return 3 - F    # mma-class FP16-accumulate datapath
    return -108 - F     # FP32-wide datapath

def make_t_fdpa(a_ctx: fp.EFloatContext, b_ctx: fp.EFloatContext, c_ctx: fp.EFloatContext,
                F: int, rho: fp.Context,
                *, e_zero: int | None = None, is_mma: bool = True):
    """
    Builds a T-FDPA (Algorithm 7): truncated fused dot-product-add.

    `a_ctx`, `b_ctx`, `c_ctx` describe the input and accumulator
    formats (only their `emin` is read); `F` and `rho` are
    per-instruction parameters (see the module docstring).

    If `e_zero` is omitted, it is derived from the accumulator format
    and F (`is_mma=False` selects the wgmma/tcgen05 datapath); the
    FP8 instructions (F = 13) must pass it explicitly.
    """
    emin_a = a_ctx.emin
    emin_b = b_ctx.emin
    emin_c = c_ctx.emin
    if e_zero is None:
        e_zero = _default_e_zero(c_ctx, F, is_mma)

    @fp.fpy(ctx=fp.REAL)
    def t_fdpa(A, B, c):
        """
        Truncated fused dot-product-add (Algorithm 7).

        Computes the L products exactly and aligns them and `c` at the
        maximum exponent-field sum `e_max`. Each term is truncated
        toward zero at 2^(e_max - F), i.e. to F fractional bits, summed
        exactly, and converted to the output format with `rho`.

        A zero product (either factor zero) or a zero accumulator
        reads as exponent `e_zero` rather than its format exponent.
        """
        # Step 1: exact products and their exponent-field sums
        # (an exact product is nonfinite iff one of its factors is)
        prods = [a * b for a, b in zip(A, B)]
        if any([not fp.isfinite(p) for p in prods]) or not fp.isfinite(c):
            return dpa_special_values(A, B, c)

        es = [e_zero if p == 0 else exponent(a, emin_a) + exponent(b, emin_b)
              for p, a, b in zip(prods, A, B)]
        e_c = e_zero if c == 0 else exponent(c, emin_c)

        # Step 2: truncated fused sum of L + 1 terms
        e_max = max(max(es), e_c)
        s = fused_sum(join(prods, [c]), e_max - F - 1, fp.RM.RTZ)

        # Step 3: convert to the output format
        return fdpa_round(s, rho)
    return t_fdpa

def make_t_fdpa_chain(l_max: int,
                      a_ctx: fp.EFloatContext, b_ctx: fp.EFloatContext, c_ctx: fp.EFloatContext,
                      F: int, rho: fp.Context,
                      *, e_zero: int | None = None, is_mma: bool = True):
    """
    Builds a Phi_FDPA dot-product-accumulate (Algorithm 5): chains
    T-FDPAs over vectors of length K, L = min(K, Lmax) elements at
    a time.

    Assumes `rho` outputs the accumulator format, so `c_ctx` also
    describes the running accumulator.
    """
    fdpa_op = make_t_fdpa(a_ctx, b_ctx, c_ctx, F, rho, e_zero=e_zero, is_mma=is_mma)

    @fp.fpy(ctx=fp.REAL)
    def t_fdpa_chain(A, B, c):
        """Chain of T-FDPAs (Algorithm 5)."""
        k = len(A)
        L = min(k, l_max)

        d = c
        for i in range(0, k, L):
            hi = min(i + L, k)
            d = fdpa_op(A[i:hi], B[i:hi], d)
        return d
    return t_fdpa_chain

def make_st_fdpa(a_ctx: fp.EFloatContext, b_ctx: fp.EFloatContext,
                 scale_ctx: fp.EFloatContext | fp.ExpContext,
                 F: int, rho: fp.Context, *, e_zero: int | None = None):
    """
    Builds an ST-FDPA (Algorithm 8): a single T-FDPA block (K = Lmax)
    with scale factors `alpha` and `beta` (E8M0) applied to the
    products; MXFP8/6/4 MMA instructions. The accumulator is FP32.
    """
    emin_a = a_ctx.emin
    emin_b = b_ctx.emin
    emin_s = scale_ctx.emin
    emin_c = fp.FP32.emin
    if e_zero is None:
        e_zero = _default_e_zero(fp.FP32, F, False)

    @fp.fpy(ctx=fp.REAL)
    def st_fdpa(A, B, c, alpha, beta):
        """
        Scaled truncated fused dot-product-add (Algorithm 8).

        Scaling is exact: the scale exponents shift each product's
        alignment exponent, and E8M0 significands are always 1.
        """
        # the scale formats (E8M0/UE4M3) encode NaN but no infinity,
        # and are positive, so they cannot otherwise affect the
        # special-value semantics
        if fp.isnan(alpha) or fp.isnan(beta):
            return fp.nan()

        # Step 1: exact scaled products and their exponent-field sums
        # (an exact product is nonfinite iff one of its factors is)
        prods = [a * b * alpha * beta for a, b in zip(A, B)]
        if any([not fp.isfinite(p) for p in prods]) or not fp.isfinite(c):
            return dpa_special_values(A, B, c)

        es = [e_zero if p == 0 else
              exponent(a, emin_a) + exponent(b, emin_b)
              + exponent(alpha, emin_s) + exponent(beta, emin_s)
              for p, a, b in zip(prods, A, B)]
        e_c = e_zero if c == 0 else exponent(c, emin_c)

        # Step 2: truncated fused sum of L + 1 terms
        e_max = max(max(es), e_c)
        s = fused_sum(join(prods, [c]), e_max - F - 1, fp.RM.RTZ)

        # Step 3: convert to the output format
        return fdpa_round(s, rho)

    return st_fdpa

def make_gst_fdpa(G: int, scale_ctx: fp.EFloatContext | fp.ExpContext,
                  F: int, rho: fp.Context, *, e_zero: int = -139):
    """
    Builds a GST-FDPA (Algorithm 9): group-scaled truncated FDPA;
    MXFP4/NVFP4 MMA instructions. The accumulator is FP32.

    The element formats are irrelevant: group dot products are
    computed exactly, so only the scale format (`scale_ctx`: UE4M3
    for NVFP4, E8M0 for MXFP4) enters the alignment. `e_zero` is not
    derivable from any format; -139 is the only known configuration.
    """
    emin_s = scale_ctx.emin
    emin_c = fp.FP32.emin

    @fp.fpy(ctx=fp.REAL)
    def gst_fdpa(A, B, c, alphas, betas):
        """
        Group-scaled truncated fused dot-product-add (Algorithm 9).

        The vectors are split into groups of size G. Each group's dot
        product is computed exactly, scaled by its scale factors
        (`alphas[g]`, `betas[g]`, one per group: replicate them if the
        scale block size exceeds G), and aligned using only the scale
        exponents. A group reads as exponent `e_zero` only if all of
        its products are zero or its scale product is zero -- a group
        that merely sums to zero by cancellation keeps its scale
        exponent.
        """
        n_groups = len(alphas)
        assert len(betas) == n_groups, "expected one scale per group on each side"
        assert len(A) == n_groups * G, "expected G elements per scale group"

        # the scale formats (E8M0/UE4M3) encode NaN but no infinity,
        # and are positive, so they cannot otherwise affect the
        # special-value semantics
        if any([fp.isnan(s1) for s1 in alphas]) or any([fp.isnan(s2) for s2 in betas]):
            return fp.nan()

        # (an exact product is nonfinite iff one of its factors is)
        prods = [a * b for a, b in zip(A, B)]
        if any([not fp.isfinite(p) for p in prods]) or not fp.isfinite(c):
            return dpa_special_values(A, B, c)

        # Step 1: exact dot product per group, scaled
        ts = fp.empty(n_groups)
        es = fp.empty(n_groups)
        for g in range(n_groups):
            group = prods[g * G:(g + 1) * G]
            scale = alphas[g] * betas[g]
            ts[g] = sum(group) * scale
            if all([p == 0 for p in group]) or scale == 0:
                es[g] = e_zero
            else:
                es[g] = exponent(alphas[g], emin_s) + exponent(betas[g], emin_s)
        e_c = e_zero if c == 0 else exponent(c, emin_c)

        # Step 2: truncated fused sum of L/G + 1 terms
        e_max = max(max(es), e_c)
        s = fused_sum(join(ts, [c]), e_max - F - 1, fp.RM.RTZ)

        # Step 3: convert to the output format
        return fdpa_round(s, rho)

    return gst_fdpa


def table8_cases():
    """
    The paper's Eq. 10 example on every NVIDIA model, reproducing the
    divergent per-architecture results of Table 8:

        d = 2^23 + (-2^23) + (-0.5) + (-0.25) + (-0.125)

    The exact answer is -0.875; each architecture truncates the small
    products differently. FP8 rows use E5M2 so the inputs are
    representable. Returns (name, result thunk, expected) triples.
    """
    def pad(xs, k):
        return xs + [0.0] * (k - len(xs))

    a = [-2.0**13, -0.5, -0.25, -0.125]
    b = [2.0**10, 1.0, 1.0, 1.0]
    c = 2.0**23

    def dpa(model, k):
        return lambda: float(model(pad(a, k), pad(b, k), c))

    # MXFP8 (ST-FDPA, Blackwell): unit E8M0 scales
    st = make_st_fdpa(fp.MX_E5M2, fp.MX_E5M2, fp.MX_E8M0, 25, RZ_FP32)

    # NVFP4 (GST-FDPA, Blackwell): FP4 elements, per-group UE4M3
    # scales. Group 0 encodes 2^23 * -1 via scales; group 1 encodes
    # -0.875 exactly, since group dot products are exact and F = 35
    # keeps everything here.
    a4 = pad([-4.0], 16) + pad([-1.0, -1.0, -0.5], 16) + [0.0] * 32
    b4 = pad([4.0], 16) + pad([1.0, 0.5, 0.5], 16) + [0.0] * 32
    gst = make_gst_fdpa(16, fp.MX_E8M0, 35, RZ_FP32)

    return [
        ('fp64 (fma)',
         dpa(make_fma_dpa(fp.FP64), 4), -0.875),
        ('volta.f16.f32',
         dpa(make_t_fdpa_chain(4, fp.FP16, fp.FP16, fp.FP32, 23, RZ_FP32), 8), 0.0),
        ('turing/ampere/ada.f16.f32',
         dpa(make_t_fdpa_chain(8, fp.FP16, fp.FP16, fp.FP32, 24, RZ_FP32), 16), -0.5),
        ('ampere.tf32.f32',
         dpa(make_t_fdpa_chain(4, fp.TF32, fp.TF32, fp.FP32, 24, RZ_FP32), 8), -0.5),
        ('ampere.bf16.f32',
         dpa(make_t_fdpa_chain(8, fp.BF16, fp.BF16, fp.FP32, 24, RZ_FP32), 16), -0.5),
        ('ada.e5m2.f32',
         dpa(make_t_fdpa_chain(16, fp.MX_E5M2, fp.MX_E5M2, fp.FP32, 13, RZ_E8M13, e_zero=-132), 16), 0.0),
        ('hopper.f16.f32 (wgmma)',
         dpa(make_t_fdpa_chain(16, fp.FP16, fp.FP16, fp.FP32, 25, RZ_FP32, is_mma=False), 32), -0.75),
        ('hopper.e5m2.f32 (wgmma)',
         dpa(make_t_fdpa_chain(32, fp.MX_E5M2, fp.MX_E5M2, fp.FP32, 13, RZ_E8M13, e_zero=-133), 32), 0.0),
        ('blackwell.f16.f32 (tcgen05)',
         dpa(make_t_fdpa_chain(16, fp.FP16, fp.FP16, fp.FP32, 25, RZ_FP32, is_mma=False), 32), -0.75),
        ('blackwell.e5m2.f32 (tcgen05)',
         dpa(make_t_fdpa_chain(32, fp.MX_E5M2, fp.MX_E5M2, fp.FP32, 25, RZ_FP32, is_mma=False), 32), -0.75),
        ('blackwell.mxfp8 (st-fdpa)',
         lambda: float(st(pad(a, 32), pad(b, 32), c, 1.0, 1.0)), -0.75),
        ('blackwell.nvfp4 (gst-fdpa)',
         lambda: float(gst(a4, b4, c, [2.0**11, 1.0, 1.0, 1.0], [2.0**8, 2.0**-1, 1.0, 1.0])), -0.875),
    ]


if __name__ == "__main__":
    for name, thunk, expected in table8_cases():
        print(f'{name:32s} {thunk():+.3f}')
