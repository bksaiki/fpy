"""
MMA designs from one parameter vector: the paper's (`models/`), and remixes.

An instruction computes `d = c + sum a_k b_k` over `L` elements in stages:

- **products**, exact or each rounded by `rho`; with `ftz`, subnormal inputs
  flush to +0 and subnormal results to signed 0;
- **scales** (`scales`, the scale format): one pair per instruction (MX), or
  one per group of `G` elements, each group's sum exact (MXFP4, NVFP4);
- **groups**: `even-odd` splits the products in two, each a truncated sum
  (CDNA3 FP8);
- **the block sum**: `fused`, the terms aligned at the largest exponent and
  truncated past `F` fraction bits; `tree`, pairwise adds each rounded by
  `rho` (CDNA2); `exact`, rounded once (CDNA1); `fma`, a chain of fused
  multiply-adds (`L` = 1);
- **joining `c`** (`fused` only): `inside` the fused sum (NVIDIA), or
  `separate`, the sum and `c` rounded down at `F2` and `F` bits and added
  (CDNA3);
- **the output**, rounded by `rho`: the accumulator the next instruction
  takes.

A call chains `k / L` instructions; a scaled design is one instruction.
Zeros' alignment exponent (`e_zero`) and special values follow each
structure's model.  :data:`catalog.CATALOG` places every `compile.DESIGNS`
entry here; at those points :func:`build` computes as its model does, bit for
bit (`tests/test_generator.py`).
"""

# ruff: noqa: C419 -- FPy has no generator expressions

import hashlib
import json
from dataclasses import dataclass, fields
from typing import Any, Literal

from models import amd, nv
from models.utils import (
    dpa_special_values,
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


def describe(ctx: Any) -> str:
    """*ctx* in short: `E<es>M<m>`, an FNUZ format's `fnuz`, a context's
    rounding mode when it rounds (`rho`)."""
    if isinstance(ctx, fp.ExpContext):
        return f'E{ctx.nbits}M0'
    return f'E{ctx.es}M{ctx.nbits - 1 - ctx.es}' + ('fnuz' if getattr(ctx, 'eoffset', 0) else '')


def about(d: Design) -> dict[str, Any]:
    """*d*'s parameters, readably (formats by :func:`describe`, `rho` with its
    rounding mode) and exactly (each context's `repr`, under `exact`)."""
    out: dict[str, Any] = {}
    for f in fields(d):
        v = getattr(d, f.name)
        if isinstance(v, (fp.EFloatContext, fp.ExpContext)):
            v = describe(v) + (f' {v.rm.name}' if f.name == 'rho' else '')
        out[f.name] = v
    out['exact'] = {f.name: repr(getattr(d, f.name)) for f in fields(d)
                    if isinstance(getattr(d, f.name), (fp.EFloatContext, fp.ExpContext))}
    return out


def name(d: Design) -> str:
    """A name for *d* its parameters determine: `gen.` and a digest."""
    key = json.dumps(about(d), sort_keys=True)
    return f'gen.{hashlib.sha256(key.encode()).hexdigest()[:12]}'


def unformable(d: Design) -> str | None:
    """Why *d* cannot be built, or `None`."""
    if d.L < 1 or d.k % d.L:
        return f'`k` = {d.k} is not whole instructions of `L` = {d.L}'
    if d.products == 'rounded' and d.sum != 'tree':
        return 'rounded products need a `tree` sum'
    if d.join == 'separate' and d.sum != 'fused':
        return 'a separate join needs a `fused` sum'
    if d.groups == 'even-odd' and (d.sum, d.join) != ('fused', 'separate'):
        return 'even-odd groups need a `fused` sum joined `separate`'
    if d.groups == 'even-odd' and d.L % 2:
        return 'even-odd groups need an even `L`'
    if d.scales is not None:
        if (d.sum, d.join, d.groups, d.ftz) != ('fused', 'inside', 'none', False):
            return 'scales need a `fused` sum joined `inside`, no groups, no FTZ'
        if d.k != d.L:
            return 'a scaled design is one instruction (`k` = `L`)'
        if d.G is not None and (d.G < 1 or d.L % d.G):
            return f'`G` = {d.G} does not divide `L` = {d.L}'
    elif d.G is not None:
        return '`G` needs scales'
    if d.sum == 'tree' and (d.L < 2 or d.L & (d.L - 1)):
        return 'a tree sums a power of two `L` >= 2'
    if d.sum == 'fma' and d.L != 1:
        return 'a chain of FMAs takes `L` = 1'
    if d.sum == 'fused' and d.join == 'separate':
        if (z := _e_zero(d)) >= d.a.emin + d.b.emin:
            return f'`e_zero` = {z} must read below any product'
    return None


def build(d: Design) -> tuple[fp.Function, list[Any]]:
    """*d*'s function and argument types, as `compile.DESIGNS`' builders give
    them: a chain over `(A, B, c)`, or a scaled instruction over
    `(A, B, c, alpha, beta)` (lists of scales, one per group, with `G`)."""
    if why := unformable(d):
        raise ValueError(why)
    vecs = [_L(_R(d.a), d.k), _L(_R(d.b), d.k), _R(d.c)]
    if d.scales is not None:
        scale = _R(d.scales) if d.G is None else _L(_R(d.scales), d.L // d.G)
        return (_scaled(d) if d.G is None else _group_scaled(d)), vecs + [scale, scale]
    block = {('fused', 'inside'): _fused, ('fused', 'separate'): _even_odd if d.groups == 'even-odd'
             else _separate, ('tree', 'inside'): _tree, ('exact', 'inside'): _exact,
             ('fma', 'inside'): _fma}[d.sum, d.join](d)
    return _chain(d, block), vecs


def _e_zero(d: Design) -> int:
    """*d*'s `e_zero`: given, else its model's default."""
    if d.e_zero is not None:
        return d.e_zero
    if d.scales is not None:
        return -139 if d.G is not None else nv._default_e_zero(fp.FP32, d.F, False)
    if d.join == 'separate':
        return amd._default_e_zero_tr(d.a, d.b)
    return nv._default_e_zero(d.c, d.F, True)


@fp.fpy(ctx=fp.REAL)
def _keep(x, tiny):
    """*x*: no flush."""
    return x


def _flushes(d: Design) -> tuple[fp.Function, fp.Function, float, float, float]:
    """How *d* flushes an input and a result (:func:`_keep` without FTZ), and
    the thresholds: the inputs' and the output's least normals.  Chosen here,
    not branched on in FPy: a branch would give a variable two types."""
    keep_in = amd.flush_subnormal if d.ftz else _keep
    keep_out = amd.flush_subnormal_signed if d.ftz else _keep
    return keep_in, keep_out, 2.0 ** d.a.emin, 2.0 ** d.b.emin, 2.0 ** d.rho.emin


def _chain(d: Design, block: fp.Function) -> fp.Function:
    """`k / L` instructions of *block*; with FTZ, `c` flushes first."""
    L, tiny_c = d.L, 2.0 ** d.c.emin
    keep_in = _flushes(d)[0]

    @fp.fpy(ctx=fp.REAL)
    def chain(A, B, c):
        acc = keep_in(c, tiny_c)
        for i in range(0, len(A), L):
            acc = block(A[i:i + L], B[i:i + L], acc)
        return acc
    return chain


def _fused(d: Design) -> fp.Function:
    """A fused sum with `c` inside, truncated at `F` (T-FDPA).  A product
    is of flushed inputs; one not 0 has neither flushed, so its exponents read
    the inputs."""
    F, rho, e_zero = d.F, d.rho, _e_zero(d)
    emin_a, emin_b, emin_c = exp_floor(d.a), exp_floor(d.b), d.c.emin
    keep_in, keep_out, tiny_a, tiny_b, tiny_o = _flushes(d)

    @fp.fpy(ctx=fp.REAL)
    def fused(A, B, c):
        prods = [keep_in(a, tiny_a) * keep_in(b, tiny_b) for a, b in zip(A, B)]
        if any([not fp.isfinite(p) for p in prods]) or not fp.isfinite(c):
            return sum_special_values(prods, c)
        es = [e_zero if p == 0 else exponent(a, emin_a) + exponent(b, emin_b)
              for p, a, b in zip(prods, A, B)]
        e_c = e_zero if c == 0 else exponent(c, emin_c)
        e_max = max(max(es), e_c)
        s = fused_sum(join(prods, [c]), e_max - F - 1, fp.RM.RTZ)
        return keep_out(nv.fdpa_round(s, rho), tiny_o)
    return fused


def _scaled(d: Design) -> fp.Function:
    """A fused sum with `c` inside, its products scaled (ST-FDPA)."""
    assert d.scales is not None
    F, rho, e_zero = d.F, d.rho, _e_zero(d)
    emin_a, emin_b, emin_c = exp_floor(d.a), exp_floor(d.b), d.c.emin
    emin_s = d.scales.emin

    @fp.fpy(ctx=fp.REAL)
    def scaled(A, B, c, alpha, beta):
        if fp.isnan(alpha) or fp.isnan(beta):
            return fp.nan()
        prods = [a * b * alpha * beta for a, b in zip(A, B)]
        if any([not fp.isfinite(p) for p in prods]) or not fp.isfinite(c):
            return dpa_special_values(A, B, c)
        es = [e_zero if p == 0 else
              exponent(a, emin_a) + exponent(b, emin_b)
              + exponent(alpha, emin_s) + exponent(beta, emin_s)
              for p, a, b in zip(prods, A, B)]
        e_c = e_zero if c == 0 else exponent(c, emin_c)
        e_max = max(max(es), e_c)
        s = fused_sum(join(prods, [c]), e_max - F - 1, fp.RM.RTZ)
        return nv.fdpa_round(s, rho)
    return scaled


def _group_scaled(d: Design) -> fp.Function:
    """A fused sum with `c` inside of exact group sums, each scaled and
    aligned at its scales' exponents (GST-FDPA)."""
    assert d.scales is not None and d.G is not None
    F, rho, e_zero, G = d.F, d.rho, _e_zero(d), d.G
    emin_c, emin_s = d.c.emin, d.scales.emin

    @fp.fpy(ctx=fp.REAL)
    def group_scaled(A, B, c, alphas, betas):
        n_groups = len(alphas)
        if any([fp.isnan(s1) for s1 in alphas]) or any([fp.isnan(s2) for s2 in betas]):
            return fp.nan()
        prods = [a * b for a, b in zip(A, B)]
        if any([not fp.isfinite(p) for p in prods]) or not fp.isfinite(c):
            return dpa_special_values(A, B, c)
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
        e_max = max(max(es), e_c)
        s = fused_sum(join(ts, [c]), e_max - F - 1, fp.RM.RTZ)
        return nv.fdpa_round(s, rho)
    return group_scaled


def _separate(d: Design) -> fp.Function:
    """A fused sum truncated at `F`, then `c` added, each rounded down
    (TR-FDPA)."""
    F, F2, rho, e_zero = d.F, d.F2, d.rho, _e_zero(d)
    emin_a, emin_b, emin_c = exp_floor(d.a), exp_floor(d.b), d.c.emin
    keep_in, keep_out, tiny_a, tiny_b, tiny_o = _flushes(d)

    @fp.fpy(ctx=fp.REAL)
    def separate(A, B, c):
        prods = [amd.overflow_inf(keep_in(a, tiny_a) * keep_in(b, tiny_b)) for a, b in zip(A, B)]
        if any([not fp.isfinite(p) for p in prods]) or not fp.isfinite(c):
            return sum_special_values(prods, c)
        es = [e_zero if p == 0 else exponent(a, emin_a) + exponent(b, emin_b)
              for p, a, b in zip(prods, A, B)]
        e_dot = max(es)
        t = fused_sum(prods, e_dot - F - 1, fp.RM.RTZ)
        e = max(e_dot, exponent(c, emin_c))
        s = round_down_at(t, e - F2 - 2) + round_down_at(c, e - F - 1)
        s = round_down_at(s, exponent(s, emin_c) - F2 - 1)
        with rho:
            r = fp.round(s)
        return keep_out(r, tiny_o)
    return separate


def _even_odd(d: Design) -> fp.Function:
    """Even- and odd-index products each a truncated sum, combined and joined
    with `c` rounded down (GTR-FDPA)."""
    F, F2, rho, e_zero = d.F, d.F2, d.rho, _e_zero(d)
    emin_a, emin_b, emin_c = exp_floor(d.a), exp_floor(d.b), d.c.emin
    keep_in, keep_out, tiny_a, tiny_b, tiny_o = _flushes(d)

    @fp.fpy(ctx=fp.REAL)
    def even_odd(A, B, c):
        prods = [keep_in(a, tiny_a) * keep_in(b, tiny_b) for a, b in zip(A, B)]
        es = [e_zero if p == 0 else amd.exponent0(a, emin_a) + amd.exponent0(b, emin_b)
              for p, a, b in zip(prods, A, B)]
        L = len(A)
        e_even = max([es[i] for i in range(0, L, 2)])
        e_odd = max([es[i] for i in range(1, L, 2)])
        if not fp.isfinite(c) and max(e_even, e_odd) > F:
            c = 0
        if any([not fp.isfinite(p) for p in prods]) or not fp.isfinite(c):
            return sum_special_values(prods, c)
        t_even = fused_sum([prods[i] for i in range(0, L, 2)], e_even - F - 1, fp.RM.RTZ)
        t_odd = fused_sum([prods[i] for i in range(1, L, 2)], e_odd - F - 1, fp.RM.RTZ)
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
    return even_odd


def _tree(d: Design) -> fp.Function:
    """Pairwise adds of the products, then `c`, each rounded by `rho`
    (FTZ-AddMul, `L` = `P`): adjacent pairs, then pairs of those."""
    rho = d.rho
    keep_in, keep_out, tiny_a, tiny_b, tiny_o = _flushes(d)

    @fp.fpy(ctx=fp.REAL)
    def exact_mul(x, y):
        return keep_in(x, tiny_a) * keep_in(y, tiny_b)

    @fp.fpy(ctx=fp.REAL)
    def rounded_mul(x, y):
        with rho:
            z = fp.round(keep_in(x, tiny_a) * keep_in(y, tiny_b))
        return keep_out(z, tiny_o)

    mul = rounded_mul if d.products == 'rounded' else exact_mul

    @fp.fpy(ctx=fp.REAL)
    def add(x, y):
        with rho:
            z = fp.round(x + y)
        return keep_out(z, tiny_o)

    @fp.fpy(ctx=fp.REAL)
    def pair(A, B):
        return add(mul(A[0], B[0]), mul(A[1], B[1]))

    node = pair
    for n in (1 << j for j in range(2, d.L.bit_length())):
        node = _halves(add, node, n)
    whole = node

    @fp.fpy(ctx=fp.REAL)
    def tree(A, B, c):
        return add(c, whole(A, B))
    return tree


def _halves(add: fp.Function, half: fp.Function, n: int) -> fp.Function:
    """A tree over *n* elements: *half* over each half, added.  Scalars only:
    the backend holds no list of computed values across a call."""
    H, N = n // 2, n

    @fp.fpy(ctx=fp.REAL)
    def halves(A, B):
        return add(half(A[0:H], B[0:H]), half(A[H:N], B[H:N]))
    return halves


def _exact(d: Design) -> fp.Function:
    """The products and `c` summed exactly, rounded once (E-FDPA)."""
    rho = d.rho
    keep_in, keep_out, tiny_a, tiny_b, tiny_o = _flushes(d)

    @fp.fpy(ctx=fp.REAL)
    def exact(A, B, c):
        prods = [keep_in(a, tiny_a) * keep_in(b, tiny_b) for a, b in zip(A, B)]
        if any([not fp.isfinite(p) for p in prods]) or not fp.isfinite(c):
            return sum_special_values(prods, c)
        s = sum(join(prods, [c]))
        if s == 0:
            s = 0
        with rho:
            r = fp.round(s)
        return keep_out(r, tiny_o)
    return exact


def _fma(d: Design) -> fp.Function:
    """A fused multiply-add rounded by `rho` (`L` = 1)."""
    rho = d.rho
    keep_in, keep_out, tiny_a, tiny_b, tiny_o = _flushes(d)

    @fp.fpy(ctx=fp.REAL)
    def fma(A, B, c):
        with rho:
            r = fp.fma(keep_in(A[0], tiny_a), keep_in(B[0], tiny_b), c)
        return keep_out(r, tiny_o)
    return fma
