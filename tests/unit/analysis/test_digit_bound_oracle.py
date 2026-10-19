"""
A store-level soundness check for digit-bound inference.

Each program runs on adversarial inputs, and every finite value it computes is
held against what the store claims of it: alone, and paired with each other
value in its loop body at the same iteration.  Most unsoundness the
store can harbor never reaches an inferred format, so this looks at the store
itself (`docs/todos/digit-bound-inference.md`, Part I, section 5).
"""

import ast as pyast
import random
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from typing import Any

import pytest

import fpy2 as fp
from fpy2.analysis import FormatInfer
from fpy2.analysis.digit_bound import DigitBoundInfer, Term, Terms
from fpy2.ast.fpyast import Expr, ForStmt, ListComp
from fpy2.ast.visitor import DefaultVisitor
from fpy2.interpret.byte import BytecodeCompiler
from fpy2.interpret.value import to_value
from fpy2.number import Float
from fpy2.strategies import monomorphize
from fpy2.types import ListType, RealType, Type

_Binder = tuple[int, int]
"""A loop, or a comprehension's clause, as `(id(node), clause)`."""

_MARK = '__fpy_mark'


class _Enclosing(DefaultVisitor):
    """The binders around each expression, outermost first."""

    out: dict[int, tuple[_Binder, ...]]
    _around: tuple[_Binder, ...]

    def __init__(self) -> None:
        self.out = {}
        self._around = ()

    def _visit_expr(self, e: Expr, ctx: Any) -> None:
        self.out[id(e)] = self._around
        super()._visit_expr(e, ctx)

    def _visit_for(self, stmt: ForStmt, ctx: Any) -> None:
        self._visit_expr(stmt.iterable, ctx)
        around, self._around = self._around, (*self._around, (id(stmt), 0))
        self._visit_block(stmt.body, ctx)
        self._around = around

    def _visit_list_comp(self, e: ListComp, ctx: Any) -> None:
        around = self._around
        for k, iterable in enumerate(e.iterables):
            self._visit_expr(iterable, ctx)
            self._around = (*self._around, (id(e), k))
        self._visit_expr(e.elt, ctx)
        self._around = around


def _counting(counts: dict[_Binder, int], b: _Binder, it: Iterable) -> Iterator:
    for i, x in enumerate(it):
        counts[b] = i
        yield x


class _Counting(BytecodeCompiler):
    """The compiler, with each loop's and clause's iterable wrapped to keep its
    iteration in :attr:`counts` while its body runs."""

    counts: dict[_Binder, int]
    _binders: list[_Binder]

    def __init__(self, func: Any, env: Any, hook: Callable[[int, Any], Any]):
        super().__init__(func, env, hook=hook)
        self.counts, self._binders = {}, []
        self.foreign_vals[_MARK] = lambda i, it: _counting(self.counts, self._binders[i], it)

    def _wrap(self, b: _Binder, it: pyast.expr) -> pyast.expr:
        self._binders.append(b)
        at = {k: getattr(it, k) for k in ('lineno', 'col_offset', 'end_lineno', 'end_col_offset')}
        return pyast.Call(pyast.Name(_MARK, pyast.Load(), **at),
                          [pyast.Constant(len(self._binders) - 1, **at), it], [], **at)

    def _visit_for(self, stmt: ForStmt, ctx: None) -> pyast.For:
        out = super()._visit_for(stmt, ctx)
        out.iter = self._wrap((id(stmt), 0), out.iter)
        return out

    def _visit_list_comp(self, e: ListComp, ctx: None) -> pyast.ListComp:
        out = super()._visit_list_comp(e, ctx)
        for k, g in enumerate(out.generators):
            g.iter = self._wrap((id(e), k), g.iter)
        return out


def _digits(v: Float) -> tuple[int, int]:
    """`(E, G)`: the leading digit, and the last."""
    c, g = v.c, v.exp
    while c % 2 == 0:
        c, g = c // 2, g + 1
    return v.e, g


def _inputs(arg_types: list[Type], seed: int = 0) -> list[list[Any]]:
    """Each list with one element at its format's largest magnitude and the
    rest at its smallest, and the reverse, at every position; then a few random
    draws.  Every counterexample so far needed a huge element at one index and
    a tiny one at another."""
    def scalar(t: RealType, big: bool) -> Float:
        return t.fmt.maxval() if big else t.fmt.minval()

    n = max((t.length for t in arg_types if isinstance(t, ListType)), default=1)
    runs = []
    for p in range(n):
        for big in (True, False):
            run = []
            for t in arg_types:
                if isinstance(t, ListType):
                    run.append([scalar(t.elt, big if i == p else not big) for i in range(t.length)])
                else:
                    run.append(scalar(t, big))
            runs.append(run)
    rng = random.Random(seed)

    def draw(t: RealType) -> Float:
        top = t.fmt.to_ordinal(t.fmt.maxval())
        return t.fmt.from_ordinal(rng.randint(-top, top))

    for _ in range(4):
        runs.append([[draw(t.elt) for _ in range(t.length)] if isinstance(t, ListType) else draw(t)
                     for t in arg_types])
    return runs


def check(fn: fp.Function, arg_types: list[Type]) -> list[str]:
    """Every claim of the store a run of *fn* contradicts."""
    mono = monomorphize(fn, args=arg_types)
    ast = mono.ast
    db = DigitBoundInfer.analyze(ast, FormatInfer.analyze(ast))
    enclosing = _Enclosing()
    enclosing._visit_function(ast, None)
    # site -> (run, iteration) -> the values observed there
    seen: dict[int, dict[tuple[int, ...], list[Float]]] = defaultdict(lambda: defaultdict(list))
    compiler: _Counting
    now = 0

    def observe(i: int, v: Any) -> Any:
        if isinstance(v, Float) and v.is_finite():
            e = compiler.hook_sites[i]
            at = (now, *(compiler.counts.get(k, -1) for k in enclosing.out.get(id(e), ())))
            seen[i][at].append(v)
        return v

    compiler = _Counting(ast, mono.env, observe)
    run = compiler.compile()
    for k, args in enumerate(_inputs(arg_types)):
        now = k
        run(*[to_value(a) for a in args], __ctx__=fp.REAL)

    store, bad = db.store, []
    sites = [(i, compiler.hook_sites[i], db.by_expr.get(compiler.hook_sites[i])) for i in seen]
    sites = [(i, e, t) for i, e, t in sites if t is not None and t.msb is not None]
    # each value observed non-zero makes its own literals true at its point
    nonzero: dict[tuple[int, ...], set[int]] = defaultdict(set)
    for i, _, t in sites:
        for k, vs in seen[i].items():
            if any(v.is_nonzero() for v in vs):
                nonzero[k].update(store.not_neg_inf(t.msb))

    def holding(k: tuple[int, ...], *es: Expr) -> frozenset[int]:
        """The literals true at point *k*: those of every value non-zero
        there or around it, and what each `with` around *es* establishes."""
        out = set().union(*(db.assume_at(e) for e in es))
        for m in range(1, len(k) + 1):
            out |= nonzero.get(k[:m], set())
        return frozenset(out)

    def readings(t: Terms) -> list[tuple[str, Term, Callable[[Float], int | None]]]:
        """What the store names of a site, and how to read each off a value:
        `E` and `G` of a non-zero one, `V` of an integer."""
        out = [('E', t.msb, lambda v: _digits(v)[0] if v.is_nonzero() else None)]
        if t.lsb is not None:
            out.append(('G', t.lsb, lambda v: _digits(v)[1] if v.is_nonzero() else None))
        if t.value is not None:
            out.append(('V', t.value, lambda v: int(v) if v.is_integer() else None))
        return out

    def observed(i: int, read: Callable[[Float], int | None]) -> dict[tuple[int, ...], list[int]]:
        return {k: xs for k, vs in seen[i].items() if (xs := [x for v in vs if (x := read(v)) is not None])}

    obs = {(i, name): observed(i, read) for i, _, t in sites for name, _, read in readings(t)}
    for i, e, t in sites:
        for name, term, _ in readings(t):
            if not (got := obs[i, name]):
                continue
            hi, k = max((x, k) for k, xs in got.items() for x in xs)
            if hi > (claim := store.maximum(term, holding(k, e))):
                bad.append(f'{name}({e.format()}) = {hi} > {claim}')
            lo, k = min((x, k) for k, xs in got.items() for x in xs)
            if lo < (claim := -store.maximum(-term, holding(k, e))):
                bad.append(f'{name}({e.format()}) = {lo} < {claim}')
    # pairs in one body, at one point: `E` against `E`, `G` and `V`, and `V`
    # against `E` -- a rounding position is built from `V`
    for i, ea, ta in sites:
        for j, eb, tb in sites:
            if i == j or enclosing.out.get(id(ea)) != enclosing.out.get(id(eb)):
                continue
            for na, ra, _ in readings(ta):
                for nb, rb, _ in readings(tb):
                    if 'E' not in (na, nb) or (na, nb) == ('G', 'E'):
                        continue
                    xa, xb = obs[i, na], obs[j, nb]
                    common = xa.keys() & xb.keys()
                    if not common:
                        continue
                    d, k = max((x - y, k) for k in common for x in xa[k] for y in xb[k])
                    if d > (claim := store.maximum(ra - rb, holding(k, ea, eb))):
                        bad.append(f'{na}({ea.format()}) - {nb}({eb.format()}) = {d} > {claim}')
    return bad


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
}

KNOWN = {
    'inner-write': 'Phase 10: a covering write under an inner loop shares its value',
    'product': 'Phase 10: a product comprehension unifies its result with each clause',
}
"""The shapes the store is known wrong on, and the phase that fixes each."""


@pytest.mark.parametrize('shape', [
    pytest.param(s, marks=pytest.mark.xfail(strict=True, reason=KNOWN[s])) if s in KNOWN else s
    for s in SHAPES
])
def test_the_store_holds_every_run(shape: str) -> None:
    fn, arg_types = SHAPES[shape]
    assert check(fn, arg_types) == []
