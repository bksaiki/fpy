"""
MMA designs from one parameter vector: the paper's (`models/`), and remixes.

An instruction computes `d = c + sum a_k b_k` over `L` elements in stages,
any choice at each with any at the others:

- **terms**: the products, exact or each rounded by `rho`; with `ftz`,
  subnormal inputs flush to +0 and subnormal results to signed 0; with
  `scales`, scaled by one pair per instruction (MX), or summed exactly in
  groups of `G`, each scaled by its pair and aligned at its scales'
  exponents (MXFP4, NVFP4);
- **groups**: `even-odd` splits the terms in two, each summed alone, the two
  combined (CDNA3 FP8);
- **the block sum**: `fused`, aligned at the largest exponent and truncated
  past `F` fraction bits; `tree`, pairwise adds each rounded by `rho`
  (CDNA2); `exact` (CDNA1); `fma`, a chain of adds each rounded by `rho`
  (fused multiply-adds, its terms the products);
- **joining `c`**: `inside`, `c` a term of the fused sum (NVIDIA), the last
  add of a tree or chain, or in the exact sum; or `separate`, the sum and `c`
  rounded down at `F2` and `F` bits and added (CDNA3);
- **the output**, rounded by `rho`: the accumulator the next instruction
  takes.

A call chains `k / L` instructions; a scaled design is one instruction.
Special values, zeros' alignment exponent (`e_zero`) and the models' other
details follow the structure that has them: NVIDIA's for a fused sum joined
inside, TR-FDPA's (products overflowing at 2^128) joined separate, GTR-FDPA's
for even-odd groups joined separate, E-FDPA's for an exact sum, IEEE
arithmetic's for a tree or chain.  :data:`catalog.CATALOG` places every
`compile.DESIGNS` entry here; at those points :func:`build` computes as its
model does, bit for bit (`tests/test_generator.py`).
"""

# ruff: noqa: C419 -- FPy has no generator expressions

import hashlib
import json
from dataclasses import dataclass, fields
from typing import Any, Literal

from models import amd, nv
from models.utils import (
    exp_floor,
    exponent,
    fused_sum,
    join,
    round_down_at,
    sum_special_values,
)

import fpy2 as fp

_L = fp.types.ListType
_R = fp.types.RealType

Sum = Literal['fused', 'tree', 'exact', 'fma']
Join = Literal['inside', 'separate']
Products = Literal['exact', 'rounded']
Groups = Literal['none', 'even-odd']


@dataclass(frozen=True)
class Design:
    """An MMA instruction, and how many elements a call chains."""

    a: fp.EFloatContext
    b: fp.EFloatContext
    c: fp.EFloatContext
    """The accumulator's format (and so `c`'s)."""
    L: int
    """Elements per instruction."""
    rho: fp.EFloatContext
    """How the instruction rounds its result: the next one's accumulator."""
    k: int
    """Elements per call: `k / L` instructions chained."""
    sum: Sum = 'fused'
    F: int = 24
    """Fraction bits a fused sum keeps."""
    join: Join = 'inside'
    F2: int = 31
    """Fraction bits a separate join keeps."""
    products: Products = 'exact'
    ftz: bool = False
    scales: fp.EFloatContext | fp.ExpContext | None = None
    G: int | None = None
    """Elements per scale group (`None`: one scale pair per instruction)."""
    groups: Groups = 'none'
    e_zero: int | None = None
    """A zero term's alignment exponent (`None`: its model's default)."""


def describe(ctx: fp.EFloatContext | fp.ExpContext) -> str:
    """*ctx* in short: `E<es>M<m>`, plus `fnuz` for an FNUZ format."""
    if isinstance(ctx, fp.ExpContext):
        return f'E{ctx.nbits}M0'
    return f'E{ctx.es}M{ctx.nbits - 1 - ctx.es}' + ('fnuz' if ctx.eoffset else '')


def about(d: Design) -> dict[str, Any]:
    """*d*'s parameters, readably (formats by :func:`describe`, `rho` with its
    rounding mode) and exactly (each context's `repr`, under `exact`)."""
    out: dict[str, Any] = {}
    exact: dict[str, str] = {}
    for f in fields(d):
        v = getattr(d, f.name)
        if isinstance(v, (fp.EFloatContext, fp.ExpContext)):
            exact[f.name] = repr(v)
            v = describe(v) + (f' {v.rm.name}' if f.name == 'rho' else '')
        out[f.name] = v
    out['exact'] = exact
    return out


def name(d: Design) -> str:
    """*d*'s name: `gen.` and a digest of its parameters."""
    key = json.dumps(about(d), sort_keys=True)
    return f'gen.{hashlib.sha256(key.encode()).hexdigest()[:12]}'


def invalid(d: Design) -> str | None:
    """Why *d*'s sizes disagree, or `None`."""
    if d.L < 1 or d.k % d.L:
        return f'`k` = {d.k} is not whole instructions of `L` = {d.L}'
    if d.scales is None and d.G is not None:
        return '`G` is a scale group\'s size: it needs scales'
    if d.scales is not None and d.k != d.L:
        return 'a scaled design is one instruction (`k` = `L`)'
    if d.G is not None and (d.G < 1 or d.L % d.G):
        return f'`G` = {d.G} does not divide `L` = {d.L}'
    if d.groups == 'even-odd' and _terms(d) < 2:
        return 'even-odd groups split two terms or more'
    return None


def build(d: Design) -> tuple[fp.Function, list[Any]]:
    """*d*'s function and argument types, as `compile.DESIGNS`' builders give
    them: a chain over `(A, B, c)`, or a scaled instruction over
    `(A, B, c, alpha, beta)` (lists of scales, one per group, with `G`)."""
    if why := invalid(d):
        raise ValueError(why)
    vecs = [_L(_R(d.a), d.k), _L(_R(d.b), d.k), _R(d.c)]
    if (d.sum, d.join, d.groups, d.products) == ('fma', 'inside', 'none', 'exact'):
        return _chain(d, _fma(d)), vecs
    # an FMA chain has no `_Parts`: an FP64 accumulator has no `e_zero`
    p = _Parts(d)
    if d.scales is not None:
        scale = _R(d.scales) if d.G is None else _L(_R(d.scales), d.L // d.G)
        return (_scaled(p) if d.G is None else _group_scaled(p)), vecs + [scale, scale]
    return _chain(d, _flat(p)), vecs


def _terms(d: Design) -> int:
    """How many terms an instruction sums: its products, or its groups."""
    return d.L if d.G is None else d.L // d.G


def _rule(d: Design) -> str:
    """Whose special values and details *d* follows: `nv` (T-FDPA), `tr`,
    `gtr`, `sum` (E-FDPA), or `ieee` (its arithmetic's)."""
    if d.join == 'separate':
        return 'gtr' if d.groups == 'even-odd' else 'tr'
    return {'fused': 'nv', 'exact': 'sum'}.get(d.sum, 'ieee')


def _e_zero(d: Design) -> int:
    """*d*'s `e_zero`: given, else its model's default."""
    if d.e_zero is not None:
        return d.e_zero
    if d.join == 'separate':
        return amd._default_e_zero_tr(d.a, d.b)
    if d.scales is not None:
        return -139 if d.G is not None else nv._default_e_zero(fp.FP32, d.F, False)
    return nv._default_e_zero(d.c, d.F, True)


@fp.fpy(ctx=fp.REAL)
def _keep(x, tiny):
    """*x*: no flush."""
    return x


@fp.fpy(ctx=fp.REAL)
def _same_c(c, es):
    return c


@fp.fpy(ctx=fp.REAL)
def _exact_add(x, y):
    return x + y


@fp.fpy(ctx=fp.REAL)
def _specials_of_products(ps, terms, c):
    """T-FDPA's: of the unscaled products, where they are special; else (a
    finite product rounded to infinity) of the terms."""
    if any([not fp.isfinite(x) for x in ps]) or not fp.isfinite(c):
        return sum_special_values(ps, c)
    return sum_special_values(terms, c)


@fp.fpy(ctx=fp.REAL)
def _specials_of_terms(ps, terms, c):
    return sum_special_values(terms, c)


class _Parts:
    """*d*'s pieces, each an FPy function chosen here, not branched on in FPy
    (a branch would give a variable two types; a list of computed values
    crosses a call only to be read)."""

    d: Design
    keep_in: fp.Function
    keep_out: fp.Function
    tiny_a: float
    tiny_b: float
    tiny_c: float
    tiny_o: float
    e_zero: int
    ex: fp.Function
    check: bool
    special: fp.Function
    rnd: fp.Function
    add: fp.Function
    term: fp.Function
    pre_c: fp.Function
    finish: fp.Function
    tail: fp.Function
    """From the products, terms, their exponents and `c`: the special values
    where any is special, else :attr:`finish`."""

    def __init__(self, d: Design) -> None:
        self.d = d
        rho, F, rule = d.rho, d.F, _rule(d)
        self.e_zero = _e_zero(d)
        self.keep_in = amd.flush_subnormal if d.ftz else _keep
        self.keep_out = amd.flush_subnormal_signed if d.ftz else _keep
        self.tiny_a, self.tiny_b = 2.0 ** d.a.emin, 2.0 ** d.b.emin
        self.tiny_c, self.tiny_o = 2.0 ** d.c.emin, 2.0 ** d.rho.emin
        self.ex = amd.exponent0 if rule == 'gtr' else exponent
        self.check = rule != 'ieee'
        self.special = _specials_of_products if rule == 'nv' else _specials_of_terms
        keep_out, tiny_o = self.keep_out, self.tiny_o

        @fp.fpy(ctx=fp.REAL)
        def rnd(x):
            with rho:
                z = fp.round(x)
            return keep_out(z, tiny_o)

        @fp.fpy(ctx=fp.REAL)
        def add(x, y):
            with rho:
                z = fp.round(x + y)
            return keep_out(z, tiny_o)

        @fp.fpy(ctx=fp.REAL)
        def overflowed(x):
            return amd.overflow_inf(x)

        @fp.fpy(ctx=fp.REAL)
        def overflowed_rounded(x):
            return rnd(amd.overflow_inf(x))

        @fp.fpy(ctx=fp.REAL)
        def exact(x):
            return x

        @fp.fpy(ctx=fp.REAL)
        def drop_c(c, es):
            """GTR-FDPA's: a NaN or infinite `c` read at exponent -1 is dropped
            when the products' exponent exceeds `F`."""
            return 0 if not fp.isfinite(c) and max(es) > F else c

        self.rnd, self.add = rnd, add
        rounded = d.products == 'rounded'
        self.term = ({(False, False): exact, (False, True): rnd, (True, False): overflowed,
                      (True, True): overflowed_rounded}[rule == 'tr', rounded])
        self.pre_c = drop_c if rule == 'gtr' else _same_c
        self.finish = finish = _finish(self, _terms(d))
        CHECK, special, pre_c = self.check, self.special, self.pre_c

        @fp.fpy(ctx=fp.REAL)
        def tail(ps, terms, es, c):
            c2 = pre_c(c, es)
            if CHECK:
                if any([not fp.isfinite(t) for t in terms]) or not fp.isfinite(c2):
                    return special(ps, terms, c2)
            return finish(terms, es, c2)
        self.tail = tail


def _finish(p: _Parts, n: int) -> fp.Function:
    """The rest of an instruction, from its *n* terms, their alignment
    exponents and `c`: the block sum, the groups' combining, the join, the
    output."""
    d = p.d
    if d.groups == 'even-odd':
        return _finish_even_odd(p, n)
    F, rho, e_zero, emin_c = d.F, d.rho, p.e_zero, d.c.emin
    keep_out, tiny_o, add = p.keep_out, p.tiny_o, p.add
    if (d.sum, d.join) == ('fused', 'inside'):
        @fp.fpy(ctx=fp.REAL)
        def fused_inside(xs, es, c):
            e_c = e_zero if c == 0 else exponent(c, emin_c)
            e_max = max(max(es), e_c)
            s = fused_sum(join(xs, [c]), e_max - F - 1, fp.RM.RTZ)
            return keep_out(nv.fdpa_round(s, rho), tiny_o)
        return fused_inside
    if (d.sum, d.join) == ('exact', 'inside'):
        @fp.fpy(ctx=fp.REAL)
        def exact_inside(xs, es, c):
            s = sum(join(xs, [c]))
            if s == 0:
                # an exact zero is +0, even of negative zeros
                s = 0
            with rho:
                r = fp.round(s)
            return keep_out(r, tiny_o)
        return exact_inside
    if (d.sum, d.join) == ('fma', 'inside'):
        N = n

        @fp.fpy(ctx=fp.REAL)
        def chain_inside(xs, es, c):
            acc = c
            for i in range(N):
                acc = add(acc, xs[i])
            return acc
        return chain_inside
    tsum, jn = _sum(p, n), _join(p)

    @fp.fpy(ctx=fp.REAL)
    def flat_finish(xs, es, c):
        return jn(tsum(xs, es), max(es), c)
    return flat_finish


def _finish_even_odd(p: _Parts, n: int) -> fp.Function:
    """Even- and odd-index terms each summed, combined, then joined: a fused
    sum's halves rounded down at `F` and joined as GTR-FDPA joins, or with `c`
    in one fused sum joined inside; another's added as it adds."""
    d = p.d
    F, rho, e_zero, emin_c = d.F, d.rho, p.e_zero, d.c.emin
    keep_out, tiny_o = p.keep_out, p.tiny_o
    N = n
    if (d.sum, d.join) == ('fused', 'separate'):
        F2 = d.F2

        # GTR-FDPA's, its sums inline: a fixed-point sum passed through a
        # call loses its format to the backend
        @fp.fpy(ctx=fp.REAL)
        def gtr(xs, es, c):
            e_even = max([es[i] for i in range(0, N, 2)])
            e_odd = max([es[i] for i in range(1, N, 2)])
            t_even = fused_sum([xs[i] for i in range(0, N, 2)], e_even - F - 1, fp.RM.RTZ)
            t_odd = fused_sum([xs[i] for i in range(1, N, 2)], e_odd - F - 1, fp.RM.RTZ)
            e_dot = max(e_even, e_odd)
            t = round_down_at(t_even, e_dot - F - 1) + round_down_at(t_odd, e_dot - F - 1)
            e_c = exponent(c, emin_c)
            e = max(e_dot, e_c)
            t = round_down_at(t, e - F2 - 2)
            cr = 0 if e_c < e - F - 1 else round_down_at(c, e - F - 1)
            s = t + cr
            s = round_down_at(s, exponent(s, emin_c) - F2 - 1)
            with rho:
                r = fp.round(s)
            return keep_out(r, tiny_o)
        return gtr
    s_even, s_odd = _sum(p, (n + 1) // 2), _sum(p, n // 2)
    if (d.sum, d.join) == ('fused', 'inside'):
        @fp.fpy(ctx=fp.REAL)
        def fused_inside(xs, es, c):
            e_even = max([es[i] for i in range(0, N, 2)])
            e_odd = max([es[i] for i in range(1, N, 2)])
            t_even = s_even([xs[i] for i in range(0, N, 2)], [es[i] for i in range(0, N, 2)])
            t_odd = s_odd([xs[i] for i in range(1, N, 2)], [es[i] for i in range(1, N, 2)])
            e_c = e_zero if c == 0 else exponent(c, emin_c)
            e_max = max(max(e_even, e_odd), e_c)
            s = fused_sum([t_even, t_odd, c], e_max - F - 1, fp.RM.RTZ)
            return keep_out(nv.fdpa_round(s, rho), tiny_o)
        return fused_inside
    comb, jn = (_exact_add if d.sum == 'exact' else p.add), _join(p)

    @fp.fpy(ctx=fp.REAL)
    def halves(xs, es):
        return comb(s_even([xs[i] for i in range(0, N, 2)], [es[i] for i in range(0, N, 2)]),
                    s_odd([xs[i] for i in range(1, N, 2)], [es[i] for i in range(1, N, 2)]))

    @fp.fpy(ctx=fp.REAL)
    def joined(xs, es, c):
        return jn(halves(xs, es), max(es), c)
    return joined


def _sum(p: _Parts, n: int) -> fp.Function:
    """The block sum of *n* terms (no `c`), from them and their exponents."""
    d = p.d
    F, add, rnd = d.F, p.add, p.rnd
    if d.sum == 'fused':
        @fp.fpy(ctx=fp.REAL)
        def fused(xs, es):
            return fused_sum(xs, max(es) - F - 1, fp.RM.RTZ)
        return fused
    if d.sum == 'exact':
        @fp.fpy(ctx=fp.REAL)
        def exact(xs, es):
            return sum(xs)
        return exact
    if d.sum == 'tree':
        whole = _tree(add, 0, n)

        @fp.fpy(ctx=fp.REAL)
        def tree(xs, es):
            return whole(xs)
        return tree

    REST = n - 1

    # `range(REST)` and an offset: the backend emits no two-argument range
    @fp.fpy(ctx=fp.REAL)
    def chain(xs, es):
        acc = rnd(xs[0])
        for i in range(REST):
            acc = add(acc, xs[i + 1])
        return acc
    return chain


def _tree(add: fp.Function, off: int, n: int) -> fp.Function:
    """A pairwise tree over *n* terms from *off*: halves at the largest power of
    two below *n*, adjacent pairs for a power of two."""
    if n == 1:
        OFF = off

        @fp.fpy(ctx=fp.REAL)
        def leaf(xs):
            return xs[OFF]
        return leaf
    h = 1 << ((n - 1).bit_length() - 1)
    left, right = _tree(add, off, h), _tree(add, off + h, n - h)

    @fp.fpy(ctx=fp.REAL)
    def node(xs):
        return add(left(xs), right(xs))
    return node


def _join(p: _Parts) -> fp.Function:
    """How a block sum `t` (its terms aligned at `e_dot`) takes `c`, and the
    output: TR-FDPA's separate join (GTR-FDPA's for even-odd groups, which
    drops a `c` below the sum's grid); a tree's or chain's last add; an exact
    sum's single rounding."""
    d = p.d
    F, F2, rho, emin_c = d.F, d.F2, d.rho, d.c.emin
    keep_out, tiny_o, add = p.keep_out, p.tiny_o, p.add
    if d.join == 'separate':
        GTR = d.groups == 'even-odd'

        @fp.fpy(ctx=fp.REAL)
        def separate(t, e_dot, c):
            e_c = exponent(c, emin_c)
            e = max(e_dot, e_c)
            cr = round_down_at(c, e - F - 1)
            if GTR:
                if e_c < e - F - 1:
                    cr = 0
            s = round_down_at(t, e - F2 - 2) + cr
            s = round_down_at(s, exponent(s, emin_c) - F2 - 1)
            with rho:
                r = fp.round(s)
            return keep_out(r, tiny_o)
        return _overflow_guard(d, separate)
    if d.sum == 'exact':
        @fp.fpy(ctx=fp.REAL)
        def exact(t, e_dot, c):
            # E-FDPA's (an exact zero is +0), for halves' exact sums
            s = t + c
            if s == 0:
                s = 0
            with rho:
                r = fp.round(s)
            return keep_out(r, tiny_o)
        return exact

    @fp.fpy(ctx=fp.REAL)
    def last_add(t, e_dot, c):
        return add(c, t)
    return last_add


def _overflow_guard(d: Design, jn: fp.Function) -> fp.Function:
    """*jn*, a separate join, but for a tree's or chain's block sum that
    overflowed: its special value and `c`'s, with no grid to round down to.
    A fused or exact sum cannot overflow, so has no guard."""
    if d.sum in ('fused', 'exact'):
        return jn

    @fp.fpy(ctx=fp.REAL)
    def guarded(t, e_dot, c):
        if not fp.isfinite(t) or not fp.isfinite(c):
            return sum_special_values([t], c)
        return jn(t, e_dot, c)
    return guarded


def _chain(d: Design, block: fp.Function) -> fp.Function:
    """`k / L` instructions of *block*; with FTZ, `c` flushes first."""
    L, tiny_c = d.L, 2.0 ** d.c.emin
    keep_in = amd.flush_subnormal if d.ftz else _keep

    @fp.fpy(ctx=fp.REAL)
    def chain(A, B, c):
        acc = keep_in(c, tiny_c)
        for i in range(0, len(A), L):
            acc = block(A[i:i + L], B[i:i + L], acc)
        return acc
    return chain


def _flat(p: _Parts) -> fp.Function:
    """An unscaled instruction: its products, each a term aligned at its
    inputs' exponents (a product not 0 has neither input flushed)."""
    d = p.d
    keep_in, tiny_a, tiny_b, term, ex = p.keep_in, p.tiny_a, p.tiny_b, p.term, p.ex
    emin_a, emin_b, e_zero, tail = exp_floor(d.a), exp_floor(d.b), p.e_zero, p.tail

    @fp.fpy(ctx=fp.REAL)
    def flat(A, B, c):
        ps = [keep_in(a, tiny_a) * keep_in(b, tiny_b) for a, b in zip(A, B)]
        terms = [term(x) for x in ps]
        es = [e_zero if t == 0 else ex(a, emin_a) + ex(b, emin_b) for t, a, b in zip(terms, A, B)]
        return tail(ps, terms, es, c)
    return flat


def _scaled(p: _Parts) -> fp.Function:
    """An instruction scaled by one pair: each product scaled, aligned at its
    inputs' and the scales' exponents (ST-FDPA)."""
    d = p.d
    assert d.scales is not None
    keep_in, tiny_a, tiny_b, tiny_c, term, ex = p.keep_in, p.tiny_a, p.tiny_b, p.tiny_c, p.term, p.ex
    emin_a, emin_b, emin_s, e_zero = exp_floor(d.a), exp_floor(d.b), d.scales.emin, p.e_zero
    tail = p.tail

    @fp.fpy(ctx=fp.REAL)
    def scaled(A, B, c, alpha, beta):
        if fp.isnan(alpha) or fp.isnan(beta):
            return fp.nan()
        ps = [keep_in(a, tiny_a) * keep_in(b, tiny_b) for a, b in zip(A, B)]
        terms = [term(x * alpha * beta) for x in ps]
        es = [e_zero if t == 0 else
              ex(a, emin_a) + ex(b, emin_b) + exponent(alpha, emin_s) + exponent(beta, emin_s)
              for t, a, b in zip(terms, A, B)]
        return tail(ps, terms, es, keep_in(c, tiny_c))
    return scaled


def _group_scaled(p: _Parts) -> fp.Function:
    """An instruction scaled per group: each group of `G` products summed
    exactly and scaled, aligned at its scales' exponents (GST-FDPA)."""
    d = p.d
    assert d.scales is not None and d.G is not None
    keep_in, tiny_a, tiny_b, tiny_c, term = p.keep_in, p.tiny_a, p.tiny_b, p.tiny_c, p.term
    G, NG, emin_s, e_zero = d.G, _terms(d), d.scales.emin, p.e_zero
    CHECK, special, pre_c, finish = p.check, p.special, p.pre_c, p.finish

    @fp.fpy(ctx=fp.REAL)
    def group_scaled(A, B, c0, alphas, betas):
        if any([fp.isnan(s1) for s1 in alphas]) or any([fp.isnan(s2) for s2 in betas]):
            return fp.nan()
        c = keep_in(c0, tiny_c)
        ps = [keep_in(a, tiny_a) * keep_in(b, tiny_b) for a, b in zip(A, B)]
        if CHECK:
            if any([not fp.isfinite(x) for x in ps]) or not fp.isfinite(c):
                return special(ps, ps, c)
        ts = fp.empty(NG)
        es = fp.empty(NG)
        for g in range(NG):
            group = ps[g * G:(g + 1) * G]
            scale = alphas[g] * betas[g]
            ts[g] = term(sum(group) * scale)
            if all([x == 0 for x in group]) or scale == 0:
                es[g] = e_zero
            else:
                es[g] = exponent(alphas[g], emin_s) + exponent(betas[g], emin_s)
        return finish(ts, es, pre_c(c, es))
    return group_scaled


def _fma(d: Design) -> fp.Function:
    """A chain of fused multiply-adds of the (flushed) inputs from `c`, each
    rounded by `rho`: an FMA's datapath, as Triton's `tl.fma` computes it."""
    rho, L = d.rho, d.L
    keep_in = amd.flush_subnormal if d.ftz else _keep
    keep_out = amd.flush_subnormal_signed if d.ftz else _keep
    tiny_a, tiny_b, tiny_o = 2.0 ** d.a.emin, 2.0 ** d.b.emin, 2.0 ** d.rho.emin

    @fp.fpy(ctx=fp.REAL)
    def fma(A, B, c):
        acc = c
        for i in range(L):
            with rho:
                acc = fp.fma(keep_in(A[i], tiny_a), keep_in(B[i], tiny_b), acc)
            acc = keep_out(acc, tiny_o)
        return acc
    return fma
