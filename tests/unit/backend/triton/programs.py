"""Kernels shared by the Triton expect and launch tests."""

import fpy2 as fp

K = 8


@fp.fpy(ctx=fp.REAL)
def batched_dot(xss: list[list[fp.Real]], yss: list[list[fp.Real]],
                out: list[fp.Real], BLOCK: fp.Real):
    """FP16 in, exact products, FP32 accumulation."""
    for r in range(len(xss)):
        acc = fp.round(0)
        for k in range(K):
            with fp.FP32:
                acc = acc + xss[r][k] * yss[r][k]
        out[r] = acc
    return out


@fp.fpy(ctx=fp.REAL)
def row_bound(xss: list[list[fp.Real]], yss: list[list[fp.Real]],
              out: list[fp.Real], BLOCK: fp.Real):
    """`batched_dot` with its rows bound to names, as inlining a call does."""
    for r in range(len(out)):
        xs = xss[r]
        ys = yss[r]
        acc = fp.round(0)
        for k in range(K):
            with fp.FP32:
                acc = acc + xs[k] * ys[k]
        out[r] = acc
    return out


@fp.fpy(ctx=fp.FP32)
def any_all(xss: list[list[fp.Real]], out: list[fp.Real], BLOCK: fp.Real):
    for r in range(len(out)):
        flags = [xss[r][k] > 0.0 for k in range(3)]
        out[r] = 2.0 if all(flags) else (1.0 if any(flags) else 0.0)
    return out


@fp.fpy(ctx=fp.FP32)
def signbit(xs: list[fp.Real], out: list[fp.Real], BLOCK: fp.Real):
    for i in range(len(out)):
        out[i] = 1.0 if fp.signbit(xs[i]) else 0.0
    return out


@fp.fpy(ctx=fp.FP32)
def nan_inf(xs: list[fp.Real], out: list[fp.Real], BLOCK: fp.Real):
    for i in range(len(out)):
        v = fp.nan() if xs[i] > 1.0 else (
            fp.inf() if xs[i] > 0.0 else -fp.inf())
        out[i] = v
    return out


@fp.fpy(ctx=fp.FP32)
def logb(xs: list[fp.Real], out: list[fp.Real], BLOCK: fp.Real):
    for i in range(len(out)):
        out[i] = fp.logb(xs[i])
    return out


@fp.fpy(ctx=fp.REAL)
def scaled(xs: list[fp.Real], ns: list[fp.Real], out: list[fp.Real],
           BLOCK: fp.Real):
    """`2 ** n * x` with a per-lane `n`, as `RescaleFixed` emits."""
    for i in range(len(out)):
        with fp.REAL:
            t = (2 ** ns[i]) * xs[i]
        out[i] = t
    return out


def round_to_int(rm: fp.RoundingMode) -> fp.Function:
    """A kernel rounding to the integers under *rm*."""
    ctx = fp.MPFixedContext(-1, rm)

    @fp.fpy(ctx=fp.REAL)
    def k(xs: list[fp.Real], out: list[fp.Real], BLOCK: fp.Real):
        for i in range(len(out)):
            with ctx:
                t = fp.round(xs[i])
            out[i] = t
        return out

    return k
