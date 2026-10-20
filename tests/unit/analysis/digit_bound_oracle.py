"""
A store-level soundness checker for digit-bound inference, and a generator of
small programs to aim it at.

:func:`check` runs a program on adversarial inputs and holds every finite value
it computes against what the store claims of it: alone, and paired with each
other value at the same point, in its loop body or around it.  Most
unsoundness the store can harbor reaches no inferred format, so this looks at
the store itself (`docs/todos/digit-bound.md`, section 5).
"""

import ast as pyast
import importlib.util
import random
import sys
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from fractions import Fraction
from pathlib import Path
from typing import Any, NamedTuple

import fpy2 as fp
from fpy2.analysis import FormatInfer
from fpy2.analysis.digit_bound import DigitBoundAnalysis, DigitBoundInfer, Term, Terms
from fpy2.analysis.format_infer import AbstractFormat
from fpy2.ast.fpyast import Call, Expr, ForStmt, FuncDef, ListComp, WhileStmt
from fpy2.ast.visitor import DefaultVisitor
from fpy2.env import ForeignEnv
from fpy2.interpret import get_default_interpreter
from fpy2.interpret.byte import CTX_NAME, BytecodeCompiler
from fpy2.interpret.value import to_value
from fpy2.number import Context, Float
from fpy2.strategies import monomorphize
from fpy2.types import ListType, RealType, Type

_Binder = tuple[int, ...]
"""A loop, or a comprehension's clause, as `(*call, id(node), clause)`, *call*
the calls it was reached through; a `while` has clause `-1`, its iterations
not counted."""

_Point = tuple[int, ...]
"""A run, and the iteration of each binder around a value."""

_MARK = '__fpy_mark'

L4 = ListType(RealType(fp.FP32), 4)


class _Enclosing(DefaultVisitor):
    """The binders around each expression, outermost first."""

    out: dict[int, tuple[_Binder, ...]]
    _call: tuple[int, ...]
    _around: tuple[_Binder, ...]

    def __init__(self, call: tuple[int, ...]) -> None:
        self.out = {}
        self._call = call
        self._around = ()

    def _visit_expr(self, e: Expr, ctx: Any) -> None:
        self.out[id(e)] = self._around
        super()._visit_expr(e, ctx)

    def _visit_for(self, stmt: ForStmt, ctx: Any) -> None:
        self._visit_expr(stmt.iterable, ctx)
        around, self._around = self._around, (*self._around, (*self._call, id(stmt), 0))
        self._visit_block(stmt.body, ctx)
        self._around = around

    def _visit_while(self, stmt: WhileStmt, ctx: Any) -> None:
        around, self._around = self._around, (*self._around, (*self._call, id(stmt), -1))
        super()._visit_while(stmt, ctx)
        self._around = around

    def _visit_list_comp(self, e: ListComp, ctx: Any) -> None:
        around = self._around
        for k, iterable in enumerate(e.iterables):
            self._visit_expr(iterable, ctx)
            self._around = (*self._around, (*self._call, id(e), k))
        self._visit_expr(e.elt, ctx)
        self._around = around


def _counting(counts: dict[_Binder, int], b: _Binder, it: Iterable) -> Iterator:
    for i, x in enumerate(it):
        counts[b] = i
        yield x


class _Site(NamedTuple):
    """Where a hooked value came from: its expression, the analysis naming
    it, and the binders around it, the call's included."""

    e: Expr
    db: DigitBoundAnalysis
    binders: tuple[_Binder, ...]


class _Counting(BytecodeCompiler):
    """The compiler, with each loop's and clause's iterable wrapped to keep its
    iteration in :attr:`counts` while its body runs, and each call to an
    analyzed FPy function compiled the same way, so its values are seen too.
    Every compiler of one run shares :attr:`hook_sites`, :attr:`counts` and :attr:`sites`."""

    counts: dict[_Binder, int]
    sites: list[_Site]
    _db: DigitBoundAnalysis
    _call: tuple[int, ...]
    _around: tuple[_Binder, ...]
    _enclosing: _Enclosing
    _binders: list[_Binder]

    def __init__(
        self, func: FuncDef, env: ForeignEnv, hook: Callable[[int, Any], Any],
        db: DigitBoundAnalysis, *, parent: '_Counting | None' = None,
        call: tuple[int, ...] = (), around: tuple[_Binder, ...] = (),
    ) -> None:
        super().__init__(func, env, hook=hook)
        if parent is not None:
            self.hook_sites, self.counts, self.sites = parent.hook_sites, parent.counts, parent.sites
        else:
            self.counts, self.sites = {}, []
        self._db, self._call, self._around, self._binders = db, call, around, []
        self._enclosing = _Enclosing(call)
        self._enclosing._visit_function(func, None)
        self.foreign_vals[_MARK] = lambda i, it: _counting(self.counts, self._binders[i], it)

    def _visit_expr(self, e: Expr, ctx: None) -> Any:
        out = super()._visit_expr(e, ctx)
        while len(self.sites) < len(self.hook_sites):
            # the sites a callee hooked are its own, already recorded
            self.sites.append(_Site(e, self._db, (*self._around, *self._enclosing.out.get(id(e), ()))))
        return out

    def _wrap(self, b: _Binder, it: pyast.expr) -> pyast.expr:
        self._binders.append(b)
        at = {k: getattr(it, k) for k in ('lineno', 'col_offset', 'end_lineno', 'end_col_offset')}
        return pyast.Call(pyast.Name(_MARK, pyast.Load(), **at),
                          [pyast.Constant(len(self._binders) - 1, **at), it], [], **at)

    def _visit_for(self, stmt: ForStmt, ctx: None) -> pyast.For:
        out = super()._visit_for(stmt, ctx)
        out.iter = self._wrap((*self._call, id(stmt), 0), out.iter)
        return out

    def _visit_list_comp(self, e: ListComp, ctx: None) -> pyast.ListComp:
        out = super()._visit_list_comp(e, ctx)
        for k, g in enumerate(out.generators):
            g.iter = self._wrap((*self._call, id(e), k), g.iter)
        return out

    def _visit_call(self, e: Call, ctx: None) -> pyast.expr:
        sub = self._db.by_call.get(e)
        if sub is None or not isinstance(e.fn, fp.Function):
            return super()._visit_call(e, ctx)
        fn = e.fn
        assert self.hook is not None
        callee = _Counting(
            fn.ast, fn.env, self.hook, sub, parent=self, call=(*self._call, id(e)),
            around=(*self._around, *self._enclosing.out.get(id(e), ())),
        ).compile()
        interp = get_default_interpreter()
        name = str(self.gensym.fresh('__fpy_traced'))
        self.foreign_vals[name] = lambda c, *args: callee(*args, __ctx__=interp._func_ctx(fn.ast, c))
        attrs = self._location_to_attributes(e.loc)
        args = [self._visit_expr(arg, ctx) for arg in e.args]
        return pyast.Call(pyast.Name(name, pyast.Load(), **attrs),
                          [pyast.Name(CTX_NAME, pyast.Load(), **attrs), *args], [], **attrs)


def _finite(v: Any) -> Float | None:
    """*v* as a finite `Float`, where it is a finite dyadic number."""
    if isinstance(v, Fraction) and v.denominator & (v.denominator - 1) == 0:
        v = Float.from_rational(v)
    return v if isinstance(v, Float) and v.is_finite() else None


def _leaves(v: Any) -> list[Any]:
    """A list's innermost elements, which its terms describe; a scalar alone."""
    if isinstance(v, list):
        return [x for e in v for x in _leaves(e)]
    return [] if isinstance(v, tuple) else [v]


def _digits(v: Float) -> tuple[int, int]:
    """`(E, G)`: the leading digit, and the last."""
    return v.e, v.exp + (v.c & -v.c).bit_length() - 1


def _inputs(arg_types: list[Type], seed: int = 0) -> list[list[Any]]:
    """Each list with one element at its format's largest magnitude and the
    rest at its smallest, the reverse, and one zero, infinity or NaN -- where
    its format has one -- among the largest, at every position, and the first
    negated; each argument infinite or NaN with the rest at their largest; then
    a few random draws."""
    def scalar(t: RealType, kind: str) -> Float:
        af = AbstractFormat.from_format(t.fmt.format() if isinstance(t.fmt, Context) else t.fmt)
        match kind:
            case 'big':
                return t.fmt.maxval()
            case 'tiny':
                return t.fmt.minval()
            case 'inf' if af.has_pos_inf:
                return Float(isinf=True)
            case 'nan' if af.has_nan:
                return Float(isnan=True)
            case 'inf' | 'nan':
                return t.fmt.maxval()
            case _:
                return t.fmt.zero()

    def fill(t: Type, leaf: Callable[[RealType, int], Float], k: int = 0) -> Any:
        """A value of *t*, element `k` of each innermost list `leaf(elt, k)`."""
        if isinstance(t, ListType):
            return [fill(t.elt, leaf, i) for i in range(t.length)]
        assert isinstance(t, RealType)
        return leaf(t, k)

    def neg(v: Any) -> Any:
        return [neg(x) for x in v] if isinstance(v, list) else -v

    def width(t: Type) -> int:
        return width(t.elt) if isinstance(t, ListType) and isinstance(t.elt, ListType) else getattr(t, 'length', 1)

    def at(p: int, here: str, rest: str) -> Callable[[RealType, int], Float]:
        return lambda e, i: scalar(e, here if i == p else rest)

    runs = []
    for p in range(max(map(width, arg_types), default=1)):
        for here, rest in (('big', 'tiny'), ('tiny', 'big'), ('zero', 'big'), ('inf', 'big'), ('nan', 'big')):
            runs.append([fill(t, at(p, here, rest)) if isinstance(t, ListType) else scalar(t, here)
                         for t in arg_types])
    runs.append([neg(a) for a in runs[0]])
    for j in range(len(arg_types) if len(arg_types) > 1 else 0):
        for kind in ('inf', 'nan'):
            runs.append([fill(t, at(-1, '', kind if a == j else 'big')) for a, t in enumerate(arg_types)])
    rng = random.Random(seed)

    def draw(t: RealType, _: int) -> Float:
        top = t.fmt.to_ordinal(t.fmt.maxval())
        return t.fmt.from_ordinal(rng.randint(-top, top))

    for _ in range(4):
        runs.append([fill(t, draw) for t in arg_types])
    return runs


def _readings(t: Terms) -> list[tuple[str, Term, Callable[[Float], int | None]]]:
    """What the store names of a site, and how to read each off a value: `E`
    and `G` of a non-zero one, `V` of an integer."""
    out = [('E', t.msb, lambda v: _digits(v)[0] if v.is_nonzero() else None)]
    if t.lsb is not None:
        out.append(('G', t.lsb, lambda v: _digits(v)[1] if v.is_nonzero() else None))
    if t.value is not None:
        out.append(('V', t.value, lambda v: int(v) if v.is_integer() else None))
    return out


_GROUPS = 3
"""How many literal sets each term is asked under, those observing the most
first: a bound on the cost, at the price of a violation that only a stronger set
with a smaller observation shows."""

_PAIRS = {('E', 'E'), ('E', 'G'), ('E', 'V'), ('V', 'E')}
"""What a pair is checked on: `E` against `E`, `G` and `V`, and `V` against
`E` -- a rounding position is built from `V`.  `G` is only bounded below."""


def check(fn: fp.Function, arg_types: list[Type]) -> list[str]:
    """Every claim of the store a run of *fn* contradicts."""
    mono = monomorphize(fn, args=arg_types)
    ast = mono.ast
    db = DigitBoundInfer.analyze(ast, FormatInfer.analyze(ast))
    # site -> point -> the finite values observed there, each element of a list
    seen: dict[int, dict[_Point, list[Float]]] = defaultdict(lambda: defaultdict(list))
    # the points where a site was also not finite, and the sites holding lists
    partial: set[tuple[int, _Point]] = set()
    lists: set[int] = set()
    compiler: _Counting
    now = 0

    def observe(i: int, v: Any) -> Any:
        at = (now, *(compiler.counts.get(k, -1) for k in compiler.sites[i].binders))
        vs = _leaves(v)
        xs = [x for x in map(_finite, vs) if x is not None]
        seen[i][at].extend(xs)
        if len(xs) < len(vs):
            partial.add((i, at))
        if isinstance(v, list):
            lists.add(i)
        return v

    compiler = _Counting(ast, mono.env, observe, db)
    run = compiler.compile()
    ctx = get_default_interpreter()._func_ctx(ast, fp.REAL)
    for k, args in enumerate(_inputs(arg_types)):
        now = k
        try:
            run(*[to_value(a) for a in args], __ctx__=ctx)
        except Exception:  # noqa: BLE001, S112 -- what it computed before stopping still counts
            continue

    store, bad = db.store, []
    sites = []
    for i in seen:
        site = compiler.sites[i]
        t = site.db.by_expr.get(site.e)
        if t is not None and t.msb is not None:
            sites.append((i, site, t, any(b[-1] == -1 for b in site.binders)))
    # each value observed makes its own literals true at its point: it is
    # finite, and non-zero where it was -- every element, for a list
    own: dict[tuple[int, _Point], frozenset[int]] = {}
    observed: dict[tuple[tuple[_Binder, ...], _Point], set[int]] = defaultdict(set)
    for i, site, t, loose in sites:
        for k, vs in seen[i].items():
            if not vs or (i, k) in partial:
                continue
            lits = set(store.high(t.msb))
            if (all if i in lists else any)(v.is_nonzero() for v in vs):
                lits.update(store.not_neg_inf(t.msb))
            if t.value is not None and all(v.is_integer() for v in vs):
                lits.update((*store.high(t.value), *store.not_neg_inf(t.value)))
            own[i, k] = frozenset(lits)
            # a `while`'s iterations are not counted, so its values are one
            # point to their own claims only
            if not loose:
                observed[site.binders, k] |= lits

    def holding(k: _Point, binders: tuple[_Binder, ...], *at: _Site) -> frozenset[int]:
        """The literals true at point *k* under *binders*: what each value
        observed there or around it makes true, and what each `with` around
        *at* establishes."""
        out = set().union(*(a.db.assume_at(a.e) for a in at))
        for m in range(len(binders) + 1):
            out |= observed.get((binders[:m], k[:m + 1]), set())
        return frozenset(out)

    obs = {(i, name): {k: xs for k, vs in seen[i].items()
                       if (xs := [x for v in vs if (x := read(v)) is not None])}
           for i, _, t, _ in sites for name, _, read in _readings(t)}

    # each term the store bounds, what was observed of it under each set of
    # literals, and where -- one question per term and literals, however many
    # sites share the variables
    claims: dict[Term, dict[frozenset[int], tuple[int, str]]] = {}

    def claim(t: Term, d: int, lits: frozenset[int], where: str) -> None:
        groups = claims.setdefault(t, {})
        if lits not in groups or d > groups[lits][0]:
            groups[lits] = (d, where)

    for i, site, t, _ in sites:
        where = site.e.format()
        for name, term, _ in _readings(t):
            for k, xs in obs[i, name].items():
                lits = holding(k, site.binders, site) | own.get((i, k), frozenset())
                if name != 'G':
                    claim(term, max(xs), lits, f'{name}({where})')
                claim(-term, -min(xs), lits, f'-{name}({where})')
        # the query itself: its precision, which no pair reads
        if t.lsb is not None:
            for k, vs in seen[i].items():
                ps = [e - g for v in vs if v.is_nonzero() for e, g in (_digits(v),)]
                if ps:
                    lits = holding(k, site.binders, site) | own.get((i, k), frozenset())
                    claim(t.msb - t.lsb, max(ps), lits, f'P({where})')

    # pairs at one point: in one body, or one around the other.  Not a list's:
    # its terms are an element's, which another site may name at its own index
    for i, sa, ta, la in sites:
        for j, sb, tb, lb in sites:
            if i == j or la or lb or i in lists or j in lists:
                continue
            ba, bb = sa.binders, sb.binders
            deep, shallow = (ba, bb) if len(ba) >= len(bb) else (bb, ba)
            if deep[:len(shallow)] != shallow:
                continue
            n = len(shallow) + 1
            for na, ra, _ in _readings(ta):
                for nb, rb, _ in _readings(tb):
                    if (na, nb) not in _PAIRS:
                        continue
                    xa, xb = obs[i, na], obs[j, nb]
                    for kd, ds in (xa if deep is ba else xb).items():
                        other = (xb if deep is ba else xa).get(kd[:n])
                        if other is not None:
                            pa, pb = (ds, other) if deep is ba else (other, ds)
                            claim(ra - rb, max(x - y for x in pa for y in pb),
                                  holding(kd, deep, sa, sb), f'{na}({sa.e.format()}) - {nb}({sb.e.format()})')

    # the literal sets observing the most, at most `_GROUPS` per term
    for t, groups in claims.items():
        ranked = sorted(groups.items(), key=lambda g: -g[1][0])[:_GROUPS]
        for lits, (d, where) in ranked:
            if not store.reaches([(t, d)], lits):
                bad.append(f'{where} = {d}, above every bound')
    return list(dict.fromkeys(bad))


# -- a generator of small programs ----------------------------------------

_ELT = ('{}', '{} * 2', '{} * 1024', '2 ** (-fp.logb({}))')
"""What an element becomes."""

_CTXS = ('fp.REAL', 'fp.REAL', 'fp.FP32', 'fp.FP16', 'fp.MPFixedContext(-40, fp.RM.RNE)')
"""What `f` is declared under."""

_HEADER = '''\
import fpy2 as fp


@fp.fpy(ctx=fp.REAL)
def _g(zs, y):
    e0 = fp.logb(zs[0])
    with fp.MPFixedContext(e0 - 10, fp.RM.RTZ):
        r = fp.round(y * 2)
    return r


@fp.fpy(ctx={ctx})
def f(xs, ys):
'''

KINDS = (
    'map', 'zip', 'product-read', 'cover', 'reverse', 'inner-cover-past', 'outer-repeat',
    'carried', 'rounded-cover', 'inner-rounding', 'aggregate', 'part-max', 'gather',
    'scalar-add', 'call', 'while-carried', 'while-written', 'while-rounding', 'with-map',
    'with-sum', 'nested', 'nested-store', 'inner-cover-read', 'row-alias', 'row-iterated',
    'row-shared', 'partial-in-loop', 'diagonal-outer', 'diagonal-part', 'carried-sum',
)
"""The blocks :func:`generate` draws from."""

PENDING: dict[str, str] = {}
"""Blocks drawing a known failure, by what fixes it: left out until then."""


def generate(seed: int, kinds: Iterable[str] = tuple(k for k in KINDS if k not in PENDING)) -> str:
    """The source of a module defining `f(xs, ys)`, two lists of four, under a
    drawn context: a few blocks drawn from *kinds*, the shapes the store's
    rules cover."""
    rng = random.Random(seed)
    kinds = tuple(kinds)
    ctx = rng.choice(_CTXS)
    lists, scalars, lines = ['xs', 'ys'], [], []

    def name(base: str) -> str:
        return f'{base}{len(lines)}'

    def elt(x: str) -> str:
        return rng.choice(_ELT).format(x)

    def emit(*block: str) -> None:
        lines.extend('    ' + ln for b in block for ln in b.split('\n'))

    for _ in range(rng.randint(2, 4)):
        a, b = rng.choice(lists), rng.choice(lists)
        k = rng.choice((4, 10, 24))
        out = name('l')
        match rng.choice(kinds):
            case 'map':
                emit(f'{out} = [{elt("x")} for x in {a}]')
            case 'zip':
                emit(f'{out} = [x * y for x, y in zip({a}, {b})]')
            case 'product-read':
                out, zs = name('s'), name('zs')
                emit(f'{zs} = [x * y for x in {a} for y in {b}]', f'{out} = {zs}[{rng.randrange(16)}]')
            case 'cover':
                emit(f'{out} = fp.empty(4)', 'for i in range(4):',
                     f'    {out}[i] = {a}[i] {rng.choice("+-*")} {b}[i]')
            case 'reverse':
                emit(f'{out} = fp.empty(4)', 'for i in range(4):', '    with fp.INTEGER:',
                     '        k = 3 - i', f'    {out}[k] = {elt(f"{a}[i]")}')
            case 'inner-cover-past':
                # read past the inner loop that wrote it
                t = name('t')
                emit(f'{out} = fp.empty(4)', 'for i in range(4):', '    for j in range(4):',
                     f'        {out}[i] = {a}[j] * {b}[i]', f'    {t} = {out}[i]', f'    e = fp.logb({t})',
                     f'    with fp.MPFixedContext(e - {k}, fp.RM.RTZ):', f'        w = fp.round({a}[i])')
            case 'outer-repeat':
                emit(f'{out} = fp.empty(4)', 'for r in range(4):', '    for i in range(4):',
                     f'        {out}[i] = {a}[i] * {b}[r]')
            case 'carried':
                acc = name('acc')
                emit(f'{acc} = {a}[0]', f'{out} = fp.empty(4)', 'for i in range(4):',
                     f'    prev = {acc}', f'    y = {elt(f"{a}[i]")}', '    e = fp.logb(y)',
                     f'    with fp.MPFixedContext(e - {k}, fp.RM.RTZ):',
                     f'        {out}[i] = fp.round(prev)', f'    {acc} = y')
                scalars.append(acc)
            case 'rounded-cover':
                emit(f'{out} = fp.empty(4)', 'for i in range(4):', f'    e = fp.logb({a}[i])',
                     f'    with fp.MPFixedContext(e - {k}, fp.RM.RTZ):',
                     f'        {out}[i] = fp.round({b}[i])')
            case 'inner-rounding':
                emit(f'{out} = fp.empty(4)', 'for j in range(4):', f'    e = fp.logb({a}[j])',
                     f'    w = {a}[j]', '    for i in range(4):',
                     f'        with fp.MPFixedContext(e - {k}, fp.RM.RTZ):',
                     f'            w = fp.round({b}[i])', f'    {out}[j] = w')
            case 'aggregate':
                out = name('s')
                agg = rng.choice(('max', 'min', 'sum'))
                emit(f'{out} = {agg}({a})', f'{name("e")} = max([fp.logb(x) for x in {a}])')
            case 'part-max':
                out = name('s')
                emit(f'{out} = max([{a}[t] for t in range(1, 3)])')
            case 'gather':
                out, g = name('s'), name('g')
                emit(f'{g} = fp.empty(3)', 'k = 0', 'for t in range(1, 4):',
                     f'    {g}[k] = {elt(f"{a}[t]")}', '    with fp.INTEGER:', '        k = k + 1',
                     f'{out} = max({g})')
            case 'scalar-add' if scalars:
                emit(f'{out} = fp.empty(4)', 'for i in range(4):',
                     f'    {out}[i] = {a}[i] + {rng.choice(scalars)}')
            case 'while-carried':
                out = name('s')
                emit(f'{out} = {a}[0]', 'k = 0', 'while k < 3:', f'    {out} = {elt(out)}',
                     '    with fp.INTEGER:', '        k = k + 1')
            case 'while-written':
                emit(f'{out} = [x for x in {a}]', 'k = 0', 'while k < 4:',
                     f'    {out}[k] = {elt(f"{b}[k]")}', '    with fp.INTEGER:', '        k = k + 1')
            case 'while-rounding':
                out, t = name('s'), name('t')
                emit(f'{t} = {a}[0]', f'{out} = {a}[0]', 'k = 0', 'while k < 4:', f'    e = fp.logb({b}[k])',
                     f'    with fp.MPFixedContext(e - {k}, fp.RM.RTZ):', f'        {out} = fp.round({t})',
                     f'    {t} = {b}[k]', '    with fp.INTEGER:', '        k = k + 1')
            case 'with-map':
                inner = rng.choice(('fp.FP32', 'fp.FP16', f'fp.MPFixedContext(-{k}, fp.RM.RTZ)'))
                emit(f'with {inner}:', f'    {out} = [{elt("x")} for x in {a}]')
            case 'with-sum':
                out = name('s')
                emit(f'with {rng.choice(("fp.FP32", "fp.FP16"))}:', f'    {out} = sum({a})')
            case 'nested':
                # rows built fresh, read through the outer list and a row
                xss, row = name('xss'), name('row')
                emit(f'{xss} = [[{elt("x")} for x in {a}], [{elt("y")} for y in {b}]]',
                     f'{row} = {xss}[{rng.randrange(2)}]',
                     f'{out} = [{row}[i] + {xss}[1][i] for i in range(4)]')
            case 'nested-store':
                xss = name('xss')
                emit(f'{xss} = [[x for x in {a}], [y for y in {b}]]', 'for i in range(4):',
                     f'    {xss}[0][i] = {elt(f"{b}[i]")}', f'{out} = [x for x in {xss}[0]]')
            case 'inner-cover-read':
                acc, t = name('acc'), name('t')
                emit(f'{out} = [x * 0 for x in {a}]', 'for i in range(4):', f'    {acc} = fp.digits(1, -140, 2)',
                     '    for j in range(4):', f'        p = {acc}', f'        {out}[i] = {elt(f"{b}[j]")}',
                     f'        {t} = {out}[i]', f'        e = fp.logb({t})',
                     f'        with fp.MPFixedContext(e - {k}, fp.RM.RTZ):', '            r = fp.round(p)',
                     f'        {acc} = {t}')
            case 'row-alias':
                xss, row = name('xss'), name('row')
                emit(f'{xss} = [[x for x in {a}], [y for y in {b}]]', f'{row} = {xss}[0]',
                     f'{row}[{rng.randrange(4)}] = {elt(f"{b}[0]")}', f'{out} = [x for x in {xss}[0]]')
            case 'row-iterated':
                xss, row = name('xss'), name('row')
                emit(f'{xss} = [[x for x in {a}], [y for y in {b}]]', f'for {row} in {xss}:',
                     f'    {row}[1] = {elt(f"{row}[0]")}', f'{out} = [x for x in {xss}[0]]')
            case 'row-shared':
                c, xss = name('c'), name('xss')
                emit(f'{c} = [x for x in {a}]', f'{xss} = [{c}, {c}]',
                     f'{c}[1] = {elt(f"{b}[0]")}', f'{out} = [x for x in {xss}[1]]')
            case 'partial-in-loop':
                v = name('v')
                emit(f'{out} = [{elt("x")} for x in {a}]', 'for t in range(4):',
                     f'    {v} = {out}[{rng.randrange(1, 4)}]', f'    {out}[0] = {elt(f"{b}[t]")}')
            case 'diagonal-outer':
                emit(f'{out} = [x for x in {a}]', 'for r in range(2):', '    for i in range(4):',
                     f'        {out}[i] = {elt(f"{out}[i]")}')
            case 'diagonal-part':
                emit(f'{out} = [x for x in {a}]', f'for i in range({rng.randrange(1, 4)}):',
                     f'    {out}[i] = {elt(f"{out}[i]")}')
            case 'carried-sum':
                c, t = name('c'), name('t')
                emit(f'{c} = [x for x in {a}]', f'{t} = {a}[0]', 'for r in range(3):',
                     f'    {t} = sum({c})', f'    {c} = [{elt("x")} for x in {c}]', f'{out} = {c}')
            case _:
                emit(f'{out} = fp.empty(4)', 'for i in range(4):',
                     f'    {out}[i] = _g({a}, {a}[i]) + {a}[i]')
        (scalars if out.startswith('s') else lists).append(out)
    emit(f'return {lists[-1]}')
    return _HEADER.format(ctx=ctx) + '\n'.join(lines) + '\n'


def load(source: str, path: Path) -> fp.Function:
    """`f` of the module *source*, written to *path*: the decorator reads a
    function's source from its file."""
    path.write_text(source)
    name = f'_digit_bound_gen_{abs(hash(path))}'
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.f
