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


def aligned_sum(rm: fp.RM) -> fp.Function:
    """`fused_sum`'s shape: each row's terms rounded at a grid its largest
    exponent sets, which `RescaleFixed` scales in, then summed."""
    @fp.fpy(ctx=fp.REAL)
    def f(xss: list[list[fp.Real]], out: list[fp.Real], BLOCK: fp.Real):
        for r in range(len(out)):
            xs = xss[r]
            e = max([max(fp.logb(x), -126) for x in xs])
            with fp.MPFixedContext(e - 24, rm):
                ts = [fp.round(x) for x in xs]
            out[r] = sum(ts)
        return out
    return f


def logb_clamped(c: int) -> fp.Function:
    """`max(logb(x), c)`, with nothing known of `x`."""
    @fp.fpy(ctx=fp.REAL)
    def f(xs: list[fp.Real], out: list[fp.Real], BLOCK: fp.Real):
        for i in range(len(xs)):
            out[i] = max(fp.logb(xs[i]), c)
        return out
    return f


def logb_guarded(c: int) -> fp.Function:
    """`max(logb(x), c)` where `x` is proven finite."""
    @fp.fpy(ctx=fp.REAL)
    def f(xs: list[fp.Real], out: list[fp.Real], BLOCK: fp.Real):
        for i in range(len(xs)):
            x = xs[i]
            if fp.isfinite(x):
                out[i] = max(fp.logb(x), c)
            else:
                out[i] = 0
        return out
    return f


@fp.fpy(ctx=fp.REAL)
def logb_finite(xs: list[fp.Real], out: list[fp.Real], BLOCK: fp.Real):
    """`logb(x)` where `x` is proven finite."""
    for i in range(len(xs)):
        x = xs[i]
        if fp.isfinite(x):
            out[i] = fp.logb(x)
        else:
            out[i] = 0
    return out


@fp.fpy(ctx=fp.FP32)
def rare_arm(xss: list[list[fp.Real]], out: list[fp.Real], BLOCK: fp.Real):
    """A row with a non-finite element takes a long arm; the rest a short one."""
    for r in range(len(out)):
        xs = xss[r]
        if any([not fp.isfinite(x) for x in xs]):  # noqa: C419
            a = xs[0] * xs[1] + xs[2]
            b = a * xs[3] - xs[0] * xs[2]
            c = b * b + a * xs[1]
            d = c - b * xs[3] + a
            e = d * xs[0] + c * xs[1] - b * xs[2]
            s = e + d * a - c
        else:
            s = xs[0] + xs[1]
        out[r] = s
    return out


@fp.fpy(ctx=fp.FP32)
def short_arm(xss: list[list[fp.Real]], out: list[fp.Real], BLOCK: fp.Real):
    """The same with a short arm, which is not worth skipping."""
    for r in range(len(out)):
        xs = xss[r]
        if any([not fp.isfinite(x) for x in xs]):  # noqa: C419
            s = xs[0] * xs[1]
        else:
            s = xs[0] + xs[1]
        out[r] = s
    return out


@fp.fpy(ctx=fp.REAL)
def interleaved(xss: list[list[fp.Real]], out: list[list[fp.Real]], BLOCK: fp.Real):
    """The larger of each even and odd pair."""
    for j in range(len(out)):
        xs = xss[j]
        row = out[j]
        ys = [x * 2 for x in xs]
        es = [ys[i] for i in range(0, 8, 2)]
        os = [ys[i] for i in range(1, 8, 2)]
        for k in range(4):
            row[k] = max(es[k], os[k])
    return out


@fp.fpy(ctx=fp.REAL)
def reversed_row(xss: list[list[fp.Real]], out: list[list[fp.Real]], BLOCK: fp.Real):
    for j in range(len(out)):
        xs = xss[j]
        row = out[j]
        ys = [x * 2 for x in xs]
        for k in range(4):
            row[k] = ys[3 - k]
    return out
