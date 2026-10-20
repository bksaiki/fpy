"""
The digit-bound store against runs: :func:`.digit_bound_oracle.check` over
shapes the store's rules cover, regressions, and generated programs.
"""

# FPy programs: an unused read is what the store is tested on, and `all`
# takes a list
# ruff: noqa: F841, B007, C419

import os
from pathlib import Path

import pytest

import fpy2 as fp
from fpy2.analysis import FormatInfer
from fpy2.analysis.digit_bound import DigitBoundInfer
from fpy2.ast.fpyast import ReturnStmt, Round
from fpy2.strategies import monomorphize
from fpy2.types import ListType, RealType

from .digit_bound_oracle import L4, check, generate, load

# -- the shapes ----------------------------------------------------------


def _fp32s(n: int) -> list[ListType]:
    """One argument, a list of *n* FP32 values."""
    return [ListType(RealType(fp.FP32), n)]


@fp.fpy(ctx=fp.REAL)
def _paired_sums(xs):
    ys = [2 ** (-fp.logb(x)) for x in xs]
    s1 = sum(xs)
    s2 = sum(ys)
    p = s1 * s2
    e = max([fp.logb(x) for x in xs])
    with fp.MPFixedContext(e - 10, fp.RM.RTZ):
        r = fp.round(p)
    return p + p + r


@fp.fpy(ctx=fp.REAL)
def _carried(xs):
    # nonzero: a zero's `msb` is unconstrained, which would hide the join
    acc = fp.digits(1, -200, 2)
    out = fp.empty(len(xs))
    for j in range(len(xs)):
        prev = acc
        y = xs[j] * 2
        e = fp.logb(y)
        with fp.MPFixedContext(e - 10, fp.RM.RTZ):
            out[j] = fp.round(prev)
        acc = y
    return out


@fp.fpy(ctx=fp.REAL)
def _reversed_write(xs):
    n = len(xs)
    out = fp.empty(n)
    for j in range(n):
        with fp.INTEGER:
            k = n - 1 - j
        out[k] = xs[j] * 2
    rs = fp.empty(n)
    for i in range(n):
        rs[i] = out[i] + xs[i]
    return rs


@fp.fpy(ctx=fp.REAL)
def _inner_write(xs):
    out = fp.empty(4)
    for i in range(4):
        for j in range(4):
            out[i] = xs[j]
    rs = fp.empty(4)
    for i in range(4):
        rs[i] = out[i] + xs[i]
    return rs


@fp.fpy(ctx=fp.REAL)
def _product(xs, ys):
    zs = [x * y for x in xs for y in ys]
    a = zs[1]
    b = xs[1]
    return a + b


@fp.fpy(ctx=fp.REAL)
def _outer_max(xs):
    out = fp.empty(2)
    for j in range(2):
        es = [fp.logb(x) for x in xs]
        t = xs[0]
        e = max(es)
        with fp.MPFixedContext(e - 12, fp.RM.RTZ):
            out[j] = fp.round(t)
    return out


@fp.fpy(ctx=fp.REAL)
def _fused(xs, ys):
    prods = [x * y for x, y in zip(xs, ys)]
    es = [fp.logb(p) for p in prods]
    e = max(es)
    with fp.MPFixedContext(e - 10, fp.RM.RTZ):
        ts = [fp.round(p) for p in prods]
    return sum(ts)


@fp.fpy(ctx=fp.REAL)
def _affine(xs):
    out = fp.empty(3)
    for i in range(3):
        out[i] = xs[i + 1] - xs[i]
    return out


@fp.fpy(ctx=fp.REAL)
def _covering(xs):
    ys = fp.empty(4)
    for i in range(4):
        ys[i] = xs[i] * xs[i]
    zs = fp.empty(4)
    for i in range(4):
        zs[i] = ys[i] + xs[i]
    return zs


@fp.fpy(ctx=fp.REAL)
def _nested_inner_index(xs):
    out = fp.empty(4)
    for _ in range(2):
        for j in range(4):
            out[j] = xs[j] * 2
    return out


@fp.fpy(ctx=fp.REAL)
def _carried_list(xs):
    ys = [x for x in xs]
    for _ in range(3):
        ys = [y * 2 for y in ys]
    zs = fp.empty(4)
    for i in range(4):
        zs[i] = ys[i] + xs[i]
    return zs


@fp.fpy(ctx=fp.REAL)
def _branch(xs):
    out = fp.empty(4)
    for i in range(4):
        x = xs[i]
        if x > 0:
            y = x * 4
        else:
            y = x
        out[i] = y + x
    return out


@fp.fpy(ctx=fp.REAL)
def _inner_write_paired(xs, ys):
    out = fp.empty(4)
    for i in range(4):
        for j in range(4):
            out[i] = xs[j] * ys[i]
    rs = fp.empty(4)
    for i in range(4):
        rs[i] = out[i] + ys[i] + xs[i]
    return rs


@fp.fpy(ctx=fp.REAL)
def _gather_counter(xs):
    out = fp.empty(3)
    k = 0
    for t in range(1, 4):
        out[k] = xs[t]
        with fp.INTEGER:
            k = k + 1
    rs = fp.empty(3)
    for i in range(3):
        rs[i] = out[i] + xs[i + 1]
    return rs


@fp.fpy(ctx=fp.REAL)
def _gather_counter_copied(xs):
    out = fp.empty(3)
    k = 0
    for t in range(1, 4):
        out[k] = xs[t]
        with fp.INTEGER:
            k1 = k + 1
        k = k1                  # the step through a name, as the C++ pipeline leaves it
    rs = fp.empty(3)
    for i in range(3):
        rs[i] = out[i] + xs[i + 1]
    return rs


@fp.fpy(ctx=fp.REAL)
def _gather_one_index(xs):
    out = fp.empty(3)
    for t in range(1, 4):
        out[0] = xs[t]          # out[0] is xs[3]; the rest never written
    rs = fp.empty(3)
    for i in range(3):
        rs[i] = out[0] + xs[i + 1]
    return rs


@fp.fpy(ctx=fp.REAL)
def _gather_inner(xs):
    out = fp.empty(3)
    k = 0
    for t in range(1, 4):
        for _ in range(2):
            out[k] = xs[t]
        with fp.INTEGER:
            k = k + 1
    rs = fp.empty(3)
    for i in range(3):
        rs[i] = out[i] + xs[i + 1]
    return rs


@fp.fpy(ctx=fp.REAL)
def _after_loop(xs):
    acc = xs[0]
    for i in range(4):
        acc = xs[i] * 2         # past the loop, 2 * xs[3]
    rs = fp.empty(4)
    for i in range(4):
        rs[i] = acc + xs[i]
    return rs


@fp.fpy(ctx=fp.REAL)
def _g(zs, y):
    t = y * 2
    e0 = fp.logb(zs[0])
    with fp.MPFixedContext(e0 - 10, fp.RM.RTZ):
        r = fp.round(t)
    return r


@fp.fpy(ctx=fp.REAL)
def _callee_in_loop(xs):
    out = fp.empty(4)
    for i in range(4):
        y = xs[i]
        r = _g(xs, y)
        out[i] = r + y
    return out


@fp.fpy(ctx=fp.REAL)
def _nested_cover(xs):
    out = fp.empty(4)
    for j in range(4):
        e = fp.logb(xs[j])
        a = xs[j]
        for i in range(4):
            with fp.MPFixedContext(e - 4, fp.RM.RTZ):
                a = fp.round(xs[i])
        out[j] = a
    return out


@fp.fpy(ctx=fp.REAL)
def _product_of_one(xs):
    zs = [x * y for x in xs for y in xs]
    return sum(zs)


@fp.fpy(ctx=fp.REAL)
def _comp_in_loop(xs):
    rs = fp.empty(4)
    for j in range(4):
        e = fp.logb(xs[j])
        ys = [x * 2 for x in xs]
        for k in range(4):
            with fp.MPFixedContext(e - 5, fp.RM.RTZ):
                rs[k] = fp.round(ys[k])
    return rs


@fp.fpy(ctx=fp.REAL)
def _parts_paired(xs):
    ys = [2 ** (-fp.logb(x)) for x in xs]
    m = max([xs[t] for t in range(1, 3)])
    n = max([ys[t] for t in range(1, 3)])
    return m * n


@fp.fpy(ctx=fp.REAL)
def _shared_part(xs):
    s = fp.digits(1, 300, 2)
    for t in range(1, 4):
        s = 2 ** fp.logb(xs[t])
    ys = [xs[u] * 2 for u in range(1, 4)]
    out = fp.empty(3)
    for i in range(3):
        out[i] = ys[i] + s
    return out


@fp.fpy(ctx=fp.REAL)
def _list_scalar_exit(xs):
    ys = [fp.digits(1, -300, 2) for x in xs]
    s = fp.digits(1, 300, 2)
    for i in range(4):
        ys = [x for x in xs]
        s = 2 ** fp.logb(ys[i])
    out = fp.empty(4)
    for k in range(4):
        out[k] = ys[k] + s
    return out


@fp.fpy(ctx=fp.REAL)
def _partial_then_cover(xs):
    out = fp.empty(4)
    for i in range(4):
        with fp.INTEGER:
            k = 3 - i
        out[k] = xs[i] * 1024
        out[i] = xs[i]
    rs = fp.empty(4)
    for i in range(4):
        rs[i] = out[i] + xs[i]
    return rs


@fp.fpy(ctx=fp.REAL)
def _outer_rewrite(xs, ys):
    out = fp.empty(4)
    for r in range(4):
        for i in range(4):
            out[i] = xs[i] * ys[r]
    rs = fp.empty(4)
    for j in range(4):
        rs[j] = out[j] + ys[j]
    return rs


@fp.fpy(ctx=fp.REAL)
def _outer_gather(xs, ys):
    out = fp.empty(3)
    for r in range(4):
        k = 0
        for t in range(1, 4):
            out[k] = xs[t] * ys[r]
            with fp.INTEGER:
                k = k + 1
    rs = fp.empty(4)
    for j in range(4):
        rs[j] = out[0] + ys[j]
    return rs


@fp.fpy(ctx=fp.REAL)
def _tuple_outer(xs, ys):
    out = [(x, x) for x in xs]
    for r in range(4):
        for i in range(4):
            out[i] = (xs[i] * ys[r], xs[i])
    rs = fp.empty(4)
    for j in range(4):
        a, _ = out[j]
        rs[j] = a + ys[j]
    return rs


@fp.fpy(ctx=fp.REAL)
def _read_then_cover(xs):
    out = [x * fp.digits(1, -20, 2) for x in xs]
    es = fp.empty(4)
    for i in range(4):
        es[i] = fp.logb(out[i])
        out[i] = xs[i]
    zs = fp.empty(4)
    for k in range(4):
        e = es[k]
        with fp.MPFixedContext(e - 1, fp.RM.RTZ):
            zs[k] = fp.round(out[k])
    return zs


@fp.fpy(ctx=fp.REAL)
def _unknown_prior(xs):
    out = fp.empty(4)
    for i in range(4):
        out[i] = xs[i] * fp.digits(1, -20, 2)
    es = fp.empty(4)
    for i in range(4):
        es[i] = fp.logb(out[i])
        with fp.INTEGER:
            k = 3 - i
        out[k] = xs[i]
    zs = fp.empty(4)
    for k in range(4):
        e = es[k]
        with fp.MPFixedContext(e - 1, fp.RM.RTZ):
            zs[k] = fp.round(out[k])
    return zs


@fp.fpy(ctx=fp.REAL)
def _partial_leak(xs):
    n = len(xs)
    out = fp.empty(n)
    for j in range(n):
        with fp.INTEGER:
            k = n - 1 - j
        out[k] = xs[j] * 2
    for _ in range(1):
        out[0] = fp.digits(1, -100, 2)
    rs = fp.empty(n)
    for i in range(n):
        e = fp.logb(xs[i])
        with fp.MPFixedContext(e - 300, fp.RM.RTZ):
            z = fp.round(xs[i])
        rs[i] = out[i] + z
    return rs


@fp.fpy(ctx=fp.REAL)
def _zip_sibling(xs, ys):
    z = zip(xs, ys)
    out = fp.empty(4)
    for j in range(4):
        p, _ = z[j]
        with fp.INTEGER:
            e = max(fp.logb(xs[0]), -160)
        with fp.MPFixedContext(e - 10, fp.RM.RTZ):
            out[j] = fp.round(p)
    return out


@fp.fpy(ctx=fp.REAL)
def _zip_copy(xs, ys):
    z = zip(xs, ys)
    h = fp.empty(4)
    for j in range(4):
        p, _ = z[j]
        h[j] = p
    with fp.INTEGER:
        e = max(fp.logb(h[0]), -160)
    with fp.MPFixedContext(e - 10, fp.RM.RTZ):
        r = fp.round(h[3])
    return r


@fp.fpy(ctx=fp.REAL)
def _zip_nested(xs, ys):
    z = zip(xs, ys)
    out = fp.empty(4)
    for i in range(4):
        with fp.INTEGER:
            e = max(fp.logb(xs[i]), -160)
        for j in range(4):
            p, _ = z[j]
            with fp.MPFixedContext(e - 10, fp.RM.RTZ):
                out[j] = fp.round(p)
    return out


@fp.fpy(ctx=fp.REAL)
def _inner_partial(xs, ys):
    out = fp.empty(4)
    rs = fp.empty(4)
    for r in range(4):
        s = ys[r] * 2
        with fp.INTEGER:
            k = 3 - r
        for _ in range(1):
            out[k] = s          # out[3] is ys[0] * 2, from the first iteration
        e = fp.logb(s)
        with fp.MPFixedContext(e - 10, fp.RM.RTZ):
            rs[r] = fp.round(out[3])
    return rs


# -- rounding under each context class ----------------------------------

_F32 = [RealType(fp.FP32)]


@fp.fpy(ctx=fp.REAL)
def _logb_rounded(x):
    with fp.MPFloatContext(2):
        e = fp.logb(x)
    with fp.MPFixedContext(e - 3, fp.RM.RTZ):
        y = fp.round(x)
    return y


@fp.fpy(ctx=fp.REAL)
def _int_sub_rounded(x):
    e = fp.logb(x)
    with fp.MPFloatContext(2, fp.RM.RTZ):
        n = e - 3
    with fp.MPFixedContext(n, fp.RM.RTZ):
        y = fp.round(x)
    return y


@fp.fpy(ctx=fp.REAL)
def _sum_fixed_raz(xs):
    ys = [fp.digits(1, -10, 2) for x in xs]
    with fp.MPFixedContext(-1, fp.RM.RAZ):
        s = sum(ys)
    return s


@fp.fpy(ctx=fp.REAL)
def _sum_float_raz(xs):
    ys = [fp.digits(1, -10, 2) for x in xs]
    ys[0] = 1
    with fp.MPFloatContext(1, fp.RM.RAZ):
        s = sum(ys)
    return s


@fp.fpy(ctx=fp.REAL)
def _sum_of_one_fixed(xs):
    with fp.MPFixedContext(-1, fp.RM.RTZ):
        s = sum(xs)
    return s


@fp.fpy(ctx=fp.REAL)
def _sum_of_one_float(xs):
    with fp.MPFloatContext(2):
        s = sum(xs)
    return s


@fp.fpy(ctx=fp.REAL)
def _bounded_fixed(x):
    c = fp.digits(31, -4, 2)
    with fp.FixedContext(True, 0, 16):
        y = c * c
    return y


@fp.fpy(ctx=fp.REAL)
def _symbolic_float(xs):
    c = fp.digits(31, -4, 2)
    p = len(xs)
    with fp.MPFloatContext(p):
        y = c * c
    return y


_STOCHASTIC = fp.MPFloatContext(2, fp.RM.RTZ, 8)


@fp.fpy(ctx=fp.REAL)
def _stochastic(x):
    c = fp.digits(31, -4, 2)
    with _STOCHASTIC:
        y = abs(c)
    return y


@fp.fpy(ctx=fp.REAL)
def _underflow_mul(x):
    c = fp.digits(1, -100, 2)
    with fp.IEEEContext(8, 32, fp.RM.RTP):
        y = c * c
    return y


@fp.fpy(ctx=fp.REAL)
def _underflow_round(x):
    c = fp.digits(1, -200, 2)
    with fp.IEEEContext(8, 32, fp.RM.RAZ):
        y = fp.round(c)
    return y


@fp.fpy(ctx=fp.REAL)
def _underflow_exp2(x):
    with fp.IEEEContext(8, 32, fp.RM.RTP):
        y = fp.exp2(-200)
    return y


@fp.fpy(ctx=fp.REAL)
def _unsigned_wrap(x):
    a = fp.digits(1, 0, 2)
    b = fp.digits(2, 0, 2)
    with fp.FixedContext(False, 0, 8):
        y = a - b
    return y


@fp.fpy(ctx=fp.REAL)
def _saturate(x):
    c = fp.digits(16, 0, 2)
    with fp.FixedContext(True, 0, 8, fp.RM.RNE, fp.OverflowMode.SATURATE):
        y = c * c
    return y


@fp.fpy(ctx=fp.REAL)
def _all_of_or(xs):
    b = [(1 if x == 0 else 0) for x in xs]
    p = sum(xs) * sum(b)
    c = all([x == 0 or y == 0 for x, y in zip(xs, b)])
    n = -200 if c else fp.logb(p)
    with fp.MPFixedContext(n - 10, fp.RM.RTZ):
        r = fp.round(xs[0])
    return r + p

@fp.fpy(ctx=fp.REAL)
def _all_of_empty(xs):
    y = xs[0]
    p = y * 2
    with fp.INTEGER:
        k = len(xs) - 4
    c = all([y == 0 for t in range(k)])
    n = -200 if c else fp.logb(p)
    with fp.MPFixedContext(n - 10, fp.RM.RTZ):
        r = fp.round(xs[1])
    return r + p


@fp.fpy(ctx=fp.REAL)
def _first_big(xs):
    for x in xs:
        if x > 1:
            return x + 1
    return fp.inf()

@fp.fpy(ctx=fp.REAL)
def _return_in_loop(xs):
    r = _first_big(xs)
    ys = fp.empty(4)
    for i in range(4):
        e = max(fp.logb(xs[i]), -20)
        with fp.MPFixedContext(e - 10, fp.RM.RTZ):
            ys[i] = fp.round(r)
    return ys

@fp.fpy(ctx=fp.REAL)
def _first_big_or_first(xs):
    for z in xs:
        if z > 1:
            return z * 2
    return fp.digits(1, -200, 2)

@fp.fpy(ctx=fp.REAL)
def _returns_merged(xs):
    r = _first_big_or_first(xs)
    e = fp.logb(xs[0])
    with fp.MPFixedContext(e - 10, fp.RM.RTZ):
        y = fp.round(r)
    return y


@fp.fpy(ctx=fp.REAL)
def _alias_write(xs):
    ys = [x for x in xs]
    zs = ys
    zs[0] = xs[1] * 1024
    a = ys[0]
    e = fp.logb(xs[0])
    with fp.MPFixedContext(e - 10, fp.RM.RTZ):
        r = fp.round(a)
    return r

@fp.fpy(ctx=fp.REAL)
def _alias_after_writes(xs):
    ys = fp.empty(4)
    for i in range(4):
        ys[i] = xs[i] * 1024
    zs = ys
    e = fp.logb(xs[0])
    with fp.MPFixedContext(e - 10, fp.RM.RTZ):
        r = fp.round(zs[0])
    return r

@fp.fpy(ctx=fp.REAL)
def _scale(zs):
    for i in range(4):
        zs[i] = zs[i] * 1024
    return 0

@fp.fpy(ctx=fp.REAL)
def _callee_writes(xs):
    ys = [x * 1 for x in xs]
    t = _scale(ys)
    rs = fp.empty(4)
    for i in range(4):
        rs[i] = ys[i] + xs[i]
    return rs

@fp.fpy(ctx=fp.REAL)
def _put(zs, y):
    zs[0] = y
    return y

@fp.fpy(ctx=fp.REAL)
def _callee_writes_one(xs, ys):
    ws = [x * 2 for x in xs]
    r = _put(ws, ys[1] * 1024)
    return [w + x for w, x in zip(ws, xs)]

@fp.fpy(ctx=fp.REAL)
def _iterate_written(xs):
    rs = fp.empty(4)
    k = 0
    for x in xs:
        xs[3] = x * 1024
        rs[k] = x * 2
        with fp.INTEGER:
            k = k + 1
    return rs

@fp.fpy(ctx=fp.REAL)
def _iterate_written_ahead(xs):
    out = [x for x in xs]
    ys = fp.empty(4)
    k = 0
    for x in out:
        ys[k] = x
        if k < 3:
            out[k + 1] = x * 1024
        k = k + 1
    return ys


@fp.fpy(ctx=fp.REAL)
def _universal_across_iterations(xs, ys):
    acc = fp.digits(1, -200, 2)
    out = fp.empty(4)
    for r in range(4):
        prev = acc
        with fp.INTEGER:
            k = 3 - r
        w = ys[k]
        with fp.FP32:
            zs = [x * w for x in xs]
        y = fp.digits(1, -200, 2)
        for i in range(4):
            z = zs[i]
            if not fp.isfinite(z):
                y = w * fp.digits(1, 200, 2)
            else:
                y = z
        acc = y
        m = fp.empty(4)
        for j in range(4):
            t = zs[j]
            m[j] = fp.isfinite(t)
        if all(m):
            out[r] = prev
        else:
            out[r] = fp.digits(1, -200, 2)
    return out

@fp.fpy(ctx=fp.REAL)
def _arm_seed_leaks(xs):
    x = xs[0]
    t = xs[1]
    if x >= 1:
        with fp.INTEGER:
            m = min(fp.logb(x), 5)
        with fp.MPFixedContext(m - 3, fp.RM.RTZ):
            t = fp.round(xs[1])
    e = fp.logb(x)
    with fp.MPFixedContext(e - 10, fp.RM.RTZ):
        r = fp.round(xs[2])
    return r + t

@fp.fpy(ctx=fp.REAL)
def _maybe_inf(x, c):
    y = x * 2
    if c > 0:
        return fp.inf()
    return y

@fp.fpy(ctx=fp.REAL)
def _dropped_inf_return(x, c):
    r = _maybe_inf(x, c)
    m = min(fp.logb(r), 3)
    with fp.MPFixedContext(m - 10, fp.RM.RTZ):
        z = fp.round(x)
    return z + m

@fp.fpy(ctx=fp.REAL)
def _through_inf(x, y):
    with fp.FP16:
        v = x * x
    with fp.FP64:
        r = fp.fmod(y, v)
    return r * 2


@fp.fpy(ctx=fp.REAL)
def _read_after_cover(xs):
    out = [x for x in xs]
    rs = fp.empty(4)
    for i in range(4):
        out[i] = xs[i] * 1024
        rs[i] = out[3] + xs[i]
    return rs

@fp.fpy(ctx=fp.REAL)
def _inner_read_after_cover(xs):
    out = [x for x in xs]
    rs = fp.empty(4)
    for i in range(4):
        out[i] = xs[i] * 1024
        s = xs[i]
        for k in range(4):
            s = out[k] + s
        rs[i] = s
    return rs

@fp.fpy(ctx=fp.REAL)
def _double_cover(xs):
    out = [x for x in xs]
    for i in range(4):
        y = out[i]
        out[i] = xs[i] * 0.0009765625
        out[i] = xs[i] * 0.0009765625
    return out

@fp.fpy(ctx=fp.REAL)
def _cover_then_partial(xs):
    out = [x for x in xs]
    for i in range(4):
        y = out[i]
        out[i] = xs[i] * 0.0009765625
        out[0] = xs[i] * 0.0009765625
    return out

@fp.fpy(ctx=fp.REAL)
def _literal_in_loop(xs):
    r = xs[0]
    for i in range(4):
        x = xs[i]
        zs = [x, x, x, x]
        b = zs[i]
        c = zs[0]
        e = fp.logb(xs[0])
        with fp.MPFixedContext(e - 10, fp.RM.RTZ):
            r = fp.round(c)
    return r

@fp.fpy(ctx=fp.REAL)
def _zip_literal_in_loop(xs):
    r = xs[0]
    for j in range(4):
        x = xs[j]
        zs = [x, x, x, x]
        z = zip(zs, xs)
        for k in range(4):
            p, q = z[k]
            e = fp.logb(q)
            with fp.MPFixedContext(e - 10, fp.RM.RTZ):
                r = fp.round(p)
    return r

@fp.fpy(ctx=fp.REAL)
def _phi_literal_in_loop(xs):
    r = xs[0]
    for j in range(4):
        x = xs[j]
        zs = [x, x, x, x]
        if x > 0:
            w = zs
        else:
            w = xs
        for k in range(4):
            a = zs[k]
            e = fp.logb(xs[k])
            with fp.MPFixedContext(e - 10, fp.RM.RTZ):
                r = fp.round(a)
    return r

@fp.fpy(ctx=fp.REAL)
def _zip_filled_in_loop(xs):
    r = xs[0]
    for j in range(4):
        x = xs[j]
        zs = fp.empty(4)
        for i in range(4):
            zs[i] = x
        z = zip(zs, xs)
        for k in range(4):
            p, q = z[k]
            e = fp.logb(q)
            with fp.MPFixedContext(e - 10, fp.RM.RTZ):
                r = fp.round(p)
    return r

@fp.fpy(ctx=fp.REAL)
def _first(zs):
    return zs[0] * 2

@fp.fpy(ctx=fp.REAL)
def _covered_to_callee(xs):
    out = fp.empty(4)
    rs = fp.empty(4)
    for i in range(4):
        out[i] = xs[i] * 1024
        r = _first(out)
        rs[i] = r + xs[i]
    return rs

@fp.fpy(ctx=fp.REAL)
def _covered_to_callee_rounded(xs):
    out = fp.empty(4)
    rs = fp.empty(4)
    for i in range(4):
        out[i] = xs[i] * 1024
        e = fp.logb(xs[i])
        with fp.MPFixedContext(e - 10, fp.RM.RTZ):
            rs[i] = fp.round(_first(out))
    return rs

@fp.fpy(ctx=fp.REAL)
def _nested_carry(xs, ys):
    xss = [xs, ys]
    r = xs[0]
    for row in xss:
        acc = row[0]
        for x in row:
            e = fp.logb(x)
            with fp.MPFixedContext(e - 10, fp.RM.RTZ):
                r = fp.round(acc)
            acc = x
    return r

@fp.fpy(ctx=fp.REAL)
def _nested_const(xs, ys):
    xss = [xs, ys]
    r = xs[0]
    for row in xss:
        for x in row:
            e = fp.logb(x)
            a = row[0]
            with fp.MPFixedContext(e - 10, fp.RM.RTZ):
                r = fp.round(a)
    return r

@fp.fpy(ctx=fp.REAL)
def _nested_max(xs, ys):
    xss = [xs, ys]
    r = xs[0]
    for row in xss:
        for x in row:
            e = fp.logb(x)
            s = max(row)
            with fp.MPFixedContext(e - 10, fp.RM.RTZ):
                r = fp.round(s)
    return r

@fp.fpy(ctx=fp.REAL)
def _repeat(y: fp.Real):
    return [y for _ in range(4)]

@fp.fpy(ctx=fp.REAL)
def _inferred_correspondence(xs):
    out = fp.empty(4)
    for j in range(4):
        x = xs[j]
        zs = _repeat(x)
        for k in range(4):
            a = zs[k]
            e = fp.logb(xs[k])
            with fp.MPFixedContext(e - 10, fp.RM.RTZ):
                out[k] = fp.round(a)
    return out

@fp.fpy(ctx=fp.REAL)
def _times(zs: list[fp.Real], ws: list[fp.Real]):
    return [z * w for z, w in zip(zs, ws)]

@fp.fpy(ctx=fp.REAL)
def _inferred_correspondence_arg(xs):
    r = xs
    for j in range(4):
        x = xs[j]
        zs = fp.empty(4)
        for i in range(4):
            zs[i] = x
        r = _times(zs, xs)
    return r

@fp.fpy(ctx=fp.REAL)
def _branch_list_value(xs, ys):
    es1 = [fp.logb(x) for x in xs]
    es2 = [fp.logb(x) - 1 for x in xs]
    if ys[0] > 0:
        es = es1
    else:
        es = es2
    a = es[0]
    b = es[3]
    t = xs[3]
    with fp.MPFixedContext(a - 10, fp.RM.RTZ):
        r = fp.round(t)
    return r + b

@fp.fpy(ctx=fp.REAL)
def _ifexpr_list_value(xs, ys):
    es1 = [fp.logb(x) for x in xs]
    es2 = [fp.logb(x) - 1 for x in xs]
    es = es1 if ys[0] > 0 else es2
    a = es[0]
    b = es[3]
    t = xs[3]
    with fp.MPFixedContext(a - 10, fp.RM.RTZ):
        r = fp.round(t)
    return r + b

@fp.fpy(ctx=fp.REAL)
def _same_or_doubled(xs):
    if xs[0] > 0:
        return xs
    return [x * 2 for x in xs]

@fp.fpy(ctx=fp.REAL)
def _returned_list_merge(xs):
    r = _same_or_doubled(xs)
    m = sum(r)
    ys = fp.empty(4)
    for i in range(4):
        e = fp.logb(xs[i])
        with fp.MPFixedContext(e - 10, fp.RM.RTZ):
            ys[i] = fp.round(m)
    return ys


@fp.fpy(ctx=fp.REAL)
def _reassign_then_partial(xs, ys):
    out = [x for x in xs]
    for i in range(4):
        y = out[i]
        out = [z * 0.0009765625 for z in ys]
        out[0] = ys[i] * 0.0009765625
    return out

@fp.fpy(ctx=fp.REAL)
def _zero_trip_exit(xs, ys):
    out = [x for x in xs]
    for i in range(0):
        out = [z * 0.0009765625 for z in ys]
        out[0] = ys[0] * 0.0009765625
    zs = fp.empty(4)
    for k in range(4):
        zs[k] = out[k] + xs[k]
    return zs


@fp.fpy(ctx=fp.REAL)
def _while_carried(xs):
    t = xs[0]
    k = 0
    while k < 3:
        t = t * 1024
        with fp.INTEGER:
            k = k + 1
    return t


@fp.fpy(ctx=fp.REAL)
def _while_written(xs):
    out = [x for x in xs]
    k = 0
    while k < 4:
        out[k] = xs[k] * 1024
        with fp.INTEGER:
            k = k + 1
    return out


@fp.fpy(ctx=fp.REAL)
def _while_rounding(xs):
    t = xs[0]
    r = xs[0]
    k = 0
    while k < 4:
        e = fp.logb(xs[k])
        with fp.MPFixedContext(e - 10, fp.RM.RTZ):
            r = fp.round(t)
        t = xs[k]
        with fp.INTEGER:
            k = k + 1
    return r


@fp.fpy(ctx=fp.FP32)
def _declared_float(xs):
    return [x * x for x in xs]


@fp.fpy(ctx=fp.MPFixedContext(-10, fp.RM.RNE))
def _declared_fixed(x, y):
    return x * y + x


@fp.fpy(ctx=fp.FP16)
def _declared_overflow(x, y):
    t = x * y
    return t + x


# -- a covering write under an inner loop ---------------------------------


@fp.fpy(ctx=fp.REAL)
def _inner_cover(xs, ys):
    out = [x * 0 for x in xs]
    for i in range(len(out)):
        acc = fp.digits(1, -140, 2)
        for j in range(len(ys)):
            p = acc
            out[i] = ys[j] * 2
            t = out[i]
            e = fp.logb(t)
            with fp.MPFixedContext(e - 10, fp.RM.RTZ):
                r = fp.round(p)
            acc = t
    return out


# -- a call site's seed on the callee's result ----------------------------


@fp.fpy(ctx=fp.REAL)
def _double(y):
    return y * 2


@fp.fpy(ctx=fp.REAL)
def _call_seed(x, y):
    r = _double(y)
    a = x * x
    with fp.FP64:
        m = fp.fmod(a, r)
    return m


# -- a constant-index read of a list of lists ------------------------------


@fp.fpy(ctx=fp.REAL)
def _row_read(ys):
    xss = [[x for x in ys], [y for y in ys]]
    row = xss[0]
    t = ys[0]
    return [row[i] for i in range(4)]


# -- a store through a row ------------------------------------------------


@fp.fpy(ctx=fp.REAL)
def _row_alias(xss):
    row = xss[0]
    row[1] = xss[1][0] * 1024
    return xss[0][1]


@fp.fpy(ctx=fp.REAL)
def _row_iterated(xss):
    for row in xss:
        row[1] = row[0] * 1024
    return xss[0][1]


@fp.fpy(ctx=fp.REAL)
def _row_shared(xs):
    a = [x for x in xs]
    xss = [a, a]
    a[1] = xs[0] * 1024
    return xss[0][1]


@fp.fpy(ctx=fp.REAL)
def _row_held(xss):
    r = xss[0]
    xss[0][1] = xss[1][0] * 1024
    return r[1]


@fp.fpy(ctx=fp.REAL)
def _row_iterated_written(xss):
    t = xss[0][0]
    for x in xss[0]:
        xss[0][1] = x * 1024
        t = x
    return t


@fp.fpy(ctx=fp.REAL)
def _same(xs):
    return xs


@fp.fpy(ctx=fp.REAL)
def _returned_alias(xs):
    r = _same(xs)
    r[1] = xs[0] * 1024
    return xs[1]


_L2x4 = ListType(L4, 2)

SHAPES = {
    'paired-sums': (_paired_sums, _fp32s(2)),
    'carried': (_carried, [L4]),
    'reversed-write': (_reversed_write, [L4]),
    'inner-write': (_inner_write, [L4]),
    'product': (_product, [ListType(RealType(fp.FP32), 2), ListType(RealType(fp.FP16), 2)]),
    'outer-max': (_outer_max, [L4]),
    'fused': (_fused, [L4, L4]),
    'affine': (_affine, [L4]),
    'covering': (_covering, [L4]),
    'nested-inner-index': (_nested_inner_index, [L4]),
    'carried-list': (_carried_list, [L4]),
    'branch': (_branch, [L4]),
    'inner-write-paired': (_inner_write_paired, [L4, L4]),
    'gather-counter': (_gather_counter, [L4]),
    'gather-counter-copied': (_gather_counter_copied, [L4]),
    'gather-one-index': (_gather_one_index, [L4]),
    'gather-inner': (_gather_inner, [L4]),
    'after-loop': (_after_loop, [L4]),
    'callee-in-loop': (_callee_in_loop, [L4]),
    'nested-cover': (_nested_cover, [L4]),
    'product-of-one': (_product_of_one, [L4]),
    'comp-in-loop': (_comp_in_loop, [L4]),
    'parts-paired': (_parts_paired, _fp32s(3)),
    'shared-part': (_shared_part, [L4]),
    'list-scalar-exit': (_list_scalar_exit, [L4]),
    'partial-then-cover': (_partial_then_cover, [L4]),
    'outer-rewrite': (_outer_rewrite, [L4, L4]),
    'outer-gather': (_outer_gather, [L4, L4]),
    'tuple-outer': (_tuple_outer, [L4, L4]),
    'read-then-cover': (_read_then_cover, [L4]),
    'unknown-prior': (_unknown_prior, [L4]),
    'partial-leak': (_partial_leak, [L4]),
    'zip-sibling': (_zip_sibling, [L4, L4]),
    'zip-copy': (_zip_copy, [L4, L4]),
    'zip-nested': (_zip_nested, [L4, L4]),
    'inner-partial': (_inner_partial, [L4, L4]),
    'logb-rounded': (_logb_rounded, _F32),
    'int-sub-rounded': (_int_sub_rounded, _F32),
    'sum-fixed-raz': (_sum_fixed_raz, [L4]),
    'sum-float-raz': (_sum_float_raz, _fp32s(8)),
    'sum-of-one-fixed': (_sum_of_one_fixed, _fp32s(1)),
    'sum-of-one-float': (_sum_of_one_float, _fp32s(1)),
    'bounded-fixed': (_bounded_fixed, _F32),
    'symbolic-float': (_symbolic_float, _fp32s(2)),
    'stochastic': (_stochastic, _F32),
    'underflow-mul': (_underflow_mul, _F32),
    'underflow-round': (_underflow_round, _F32),
    'underflow-exp2': (_underflow_exp2, _F32),
    'unsigned-wrap': (_unsigned_wrap, _F32),
    'saturate': (_saturate, _F32),
    'all-of-or': (_all_of_or, [L4]),
    'all-of-empty': (_all_of_empty, [L4]),
    'return-in-loop': (_return_in_loop, [L4]),
    'returns-merged': (_returns_merged, [L4]),
    'alias-write': (_alias_write, [L4]),
    'alias-after-writes': (_alias_after_writes, [L4]),
    'callee-writes': (_callee_writes, [L4]),
    'callee-writes-one': (_callee_writes_one, [L4, L4]),
    'iterate-written': (_iterate_written, [L4]),
    'iterate-written-ahead': (_iterate_written_ahead, [L4]),
    'universal-across-iterations': (_universal_across_iterations, [L4, L4]),
    'arm-seed-leaks': (_arm_seed_leaks, [L4]),
    'dropped-inf-return': (_dropped_inf_return, [*_F32, *_F32]),
    'through-inf': (_through_inf, [*_F32, *_F32]),
    'read-after-cover': (_read_after_cover, [L4]),
    'inner-read-after-cover': (_inner_read_after_cover, [L4]),
    'double-cover': (_double_cover, [L4]),
    'cover-then-partial': (_cover_then_partial, [L4]),
    'literal-in-loop': (_literal_in_loop, [L4]),
    'zip-literal-in-loop': (_zip_literal_in_loop, [L4]),
    'phi-literal-in-loop': (_phi_literal_in_loop, [L4]),
    'zip-filled-in-loop': (_zip_filled_in_loop, [L4]),
    'covered-to-callee': (_covered_to_callee, [L4]),
    'covered-to-callee-rounded': (_covered_to_callee_rounded, [L4]),
    'nested-carry': (_nested_carry, [L4, L4]),
    'nested-const': (_nested_const, [L4, L4]),
    'nested-max': (_nested_max, [L4, L4]),
    'inferred-correspondence': (_inferred_correspondence, [L4]),
    'inferred-correspondence-arg': (_inferred_correspondence_arg, [L4]),
    'branch-list-value': (_branch_list_value, [L4, L4]),
    'ifexpr-list-value': (_ifexpr_list_value, [L4, L4]),
    'returned-list-merge': (_returned_list_merge, [L4]),
    'reassign-then-partial': (_reassign_then_partial, [L4, L4]),
    'zero-trip-exit': (_zero_trip_exit, [L4, L4]),
    'while-carried': (_while_carried, [L4]),
    'while-written': (_while_written, [L4]),
    'while-rounding': (_while_rounding, [L4]),
    'declared-float': (_declared_float, [L4]),
    'declared-fixed': (_declared_fixed, [*_F32, *_F32]),
    'declared-overflow': (_declared_overflow, [RealType(fp.FP16), RealType(fp.FP16)]),
    'inner-cover': (_inner_cover, [L4, L4]),
    'call-seed': (_call_seed, [*_F32, *_F32]),
    'row-alias': (_row_alias, [_L2x4]),
    'row-iterated': (_row_iterated, [_L2x4]),
    'row-shared': (_row_shared, [L4]),
    'row-held': (_row_held, [_L2x4]),
    'row-iterated-written': (_row_iterated_written, [_L2x4]),
    'returned-alias': (_returned_alias, [L4]),
    'row-read': (_row_read, [L4]),
}

PENDING: dict[str, str] = {}
"""Shapes the store is known to fail, by what fixes it: strict xfails until
then."""


def _shapes(names: list[str], pending: dict[str, str]) -> list:
    return [pytest.param(n, marks=pytest.mark.xfail(strict=True, reason=pending[n]))
            if n in pending else n for n in names]


@pytest.mark.parametrize('shape', _shapes(list(SHAPES), PENDING))
def test_the_store_holds_every_run(shape: str) -> None:
    fn, arg_types = SHAPES[shape]
    assert check(fn, arg_types) == []


FORMATS = {
    # the precision a run reaches at the shape's one rounding
    'return-in-loop': 101,      # [2**100, 2**-10, ...]
    'returns-merged': 21,       # [2**-149, 1 + 2**-20, 1, 1]
}


@pytest.mark.parametrize('name', list(FORMATS))
def test_a_rounding_bound_holds(name: str) -> None:
    fn, arg_types = SHAPES[name]
    seen = FORMATS[name]
    ast = monomorphize(fn, args=arg_types).ast
    db = DigitBoundInfer.analyze(ast, FormatInfer.analyze(ast))
    (bounds,) = [db.bounds(e) for e in db.by_expr if isinstance(e, Round)]
    assert bounds is None or bounds.prec >= seen


SEEDS = int(os.environ.get('DIGIT_BOUND_SEEDS', '32'))
"""How many generated programs the suite holds the store against; more with
`DIGIT_BOUND_SEEDS`."""

@pytest.mark.parametrize('seed', range(SEEDS))
def test_the_store_holds_a_generated_program(seed: int, tmp_path: Path) -> None:
    fn = load(generate(seed), tmp_path / f'gen{seed}.py')
    assert check(fn, [L4, L4]) == []
