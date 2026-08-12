"""
Shared helpers for the MMA-Sim models (`nv.py`, `amd.py`).
"""

import fpy2 as fp

# input conversion applied by TF32 (NVIDIA) and XF32 (AMD CDNA3)
# instructions to their FP32 operands
RZ_TF32 = fp.IEEEContext(8, 19, fp.RM.RTZ)

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
def fused_sum(xs, n, rm):
    """
    Fused summation of a non-empty list.

    Each summand is rounded with mode `rm` onto the fixed-point grid
    whose first unrepresentable digit is `n` (so the least significant
    representable digit has exponent `exp = n + 1`), then the rounded
    summands are added exactly. `sum` folds from the first element
    with no `+0` identity, so IEEE signed-zero rules apply: a sum of
    all `-0.0` terms is `-0.0`.
    """
    with fp.MPFixedContext(n, rm):
        ts = [fp.round(x) for x in xs]
    return sum(ts)

@fp.fpy(ctx=fp.REAL)
def exponent(x, emin):
    """
    The paper's `Exp(x)`: the exponent field of `x` in a floating-point
    format with minimum normalized exponent `emin`. Subnormal values
    (including zero) read as `emin`.
    """
    return max(fp.logb(x), emin)

@fp.fpy(ctx=fp.REAL)
def round_down_at(x, n):
    """
    Rounds `x` toward negative infinity on the fixed-point grid whose
    first unrepresentable digit is `n` (so the least significant
    representable digit has exponent `exp = n + 1`).
    """
    with fp.MPFixedContext(n, fp.RM.RTN):
        return fp.round(x)

@fp.fpy(ctx=fp.REAL)
def sum_special_values(ts, c):
    """
    Result of summing precomputed terms `ts` and accumulator `c` when
    they contain a NaN or an infinity (Sec. 4.2): `NaN + x = NaN`,
    `+/-inf + y = +/-inf`, and `inf - inf = NaN`.

    The accumulator is just another summand: it seeds the scan over
    the infinite terms.
    """
    if any([fp.isnan(t) for t in ts]) or fp.isnan(c):
        return fp.nan()

    has_inf = fp.isinf(c)
    inf_sgn = fp.signbit(c)
    for t in ts:
        if fp.isinf(t):
            t_sgn = fp.signbit(t)
            if has_inf:
                # check that the sign is consistent with earlier infinities
                if inf_sgn != t_sgn:
                    return fp.nan()
            else:
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

    Only the products of infinite factors are computed; the
    accumulator is just another summand and seeds the scan.
    """
    # any NaN summand => produce NaN
    if any([fp.isnan(a) for a in A]) or any([fp.isnan(b) for b in B]) or fp.isnan(c):
        return fp.nan()

    # any infinity => produce either infinity or NaN
    has_inf = fp.isinf(c)
    inf_sgn = fp.signbit(c)
    for a, b in zip(A, B):
        if fp.isinf(a) or fp.isinf(b):
            t = a * b
            if fp.isnan(t):
                return t

            t_sgn = fp.signbit(t)
            if has_inf:
                # check that the sign is consistent with earlier infinities
                if inf_sgn != t_sgn:
                    return fp.nan()
            else:
                has_inf = True
                inf_sgn = t_sgn

    assert has_inf, "expected either NaN or infinity in the input"
    return -fp.inf() if inf_sgn else fp.inf()

def make_fma_dpa(ctx: fp.Context):
    """
    Builds a Phi_FMA dot-product-accumulate (Algorithm 4): a chain of
    standard FMAs under `ctx` (FP64 or FP32, round-to-nearest-even).
    """
    @fp.fpy(ctx=fp.REAL)
    def fma_dpa(A, B, c):
        """Chain of standard FMAs (Algorithm 4)."""
        d = c
        for a, b in zip(A, B):
            with ctx:
                d = fp.fma(a, b, d)
        return d
    return fma_dpa
