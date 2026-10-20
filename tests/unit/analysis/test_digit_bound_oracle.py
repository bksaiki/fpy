"""
The digit-bound store against runs: :func:`.digit_bound_oracle.check` over
shapes the store's rules cover, the bugs found so far, and generated programs.
"""

import os

import pytest

import fpy2 as fp
from fpy2.types import ListType, RealType

from .digit_bound_oracle import check, generate, load

# -- the shapes ----------------------------------------------------------

_L4 = [ListType(RealType(fp.FP32), 4)]
_L4x2 = [ListType(RealType(fp.FP32), 4), ListType(RealType(fp.FP32), 4)]


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


_L3 = [ListType(RealType(fp.FP32), 3)]


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


SHAPES = {
    'paired-sums': (_paired_sums, [ListType(RealType(fp.FP32), 2)]),
    'carried': (_carried, _L4),
    'reversed-write': (_reversed_write, _L4),
    'inner-write': (_inner_write, _L4),
    'product': (_product, [ListType(RealType(fp.FP32), 2), ListType(RealType(fp.FP16), 2)]),
    'outer-max': (_outer_max, _L4),
    'fused': (_fused, _L4x2),
    'affine': (_affine, _L4),
    'covering': (_covering, _L4),
    'nested-inner-index': (_nested_inner_index, _L4),
    'carried-list': (_carried_list, _L4),
    'branch': (_branch, _L4),
    'inner-write-paired': (_inner_write_paired, _L4x2),
    'gather-counter': (_gather_counter, _L4),
    'gather-counter-copied': (_gather_counter_copied, _L4),
    'gather-one-index': (_gather_one_index, _L4),
    'gather-inner': (_gather_inner, _L4),
    'after-loop': (_after_loop, _L4),
    'callee-in-loop': (_callee_in_loop, _L4),
    'nested-cover': (_nested_cover, _L4),
    'product-of-one': (_product_of_one, _L4),
    'comp-in-loop': (_comp_in_loop, _L4),
    'parts-paired': (_parts_paired, _L3),
    'shared-part': (_shared_part, _L4),
    'list-scalar-exit': (_list_scalar_exit, _L4),
}


# -- the bugs known, each a strict xfail naming the phase that fixes it --------



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


KNOWN = {
    'partial-then-cover': (_partial_then_cover, _L4, 15),
    'outer-rewrite': (_outer_rewrite, _L4x2, 15),
    'outer-gather': (_outer_gather, _L4x2, 15),
    'tuple-outer': (_tuple_outer, _L4x2, 15),
    'read-then-cover': (_read_then_cover, _L4, 15),
    'unknown-prior': (_unknown_prior, _L4, 15),
    'partial-leak': (_partial_leak, _L4, 15),
}
"""The bugs found by the whole-branch review: `(program, arg types, phase)`."""


@pytest.mark.parametrize('shape', list(SHAPES))
def test_the_store_holds_every_run(shape: str) -> None:
    fn, arg_types = SHAPES[shape]
    assert check(fn, arg_types) == []


@pytest.mark.parametrize('bug', [
    pytest.param(b, marks=pytest.mark.xfail(strict=True, reason=f'Phase {KNOWN[b][2]}'))
    for b in KNOWN
])
def test_a_known_bug_is_caught(bug: str) -> None:
    fn, arg_types, _ = KNOWN[bug]
    assert check(fn, arg_types) == []


SEEDS = int(os.environ.get('DIGIT_BOUND_SEEDS', '32'))
"""How many generated programs the suite holds the store against; more with
`DIGIT_BOUND_SEEDS`."""

@pytest.mark.parametrize('seed', range(SEEDS))
def test_the_store_holds_a_generated_program(seed: int, tmp_path) -> None:
    fn = load(generate(seed), tmp_path / f'gen{seed}.py')
    assert check(fn, [ListType(RealType(fp.FP32), 4)] * 2) == []
