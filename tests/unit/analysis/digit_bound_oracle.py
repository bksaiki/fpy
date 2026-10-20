"""
A store-level soundness checker for digit-bound inference, and a generator of
small programs to aim it at.

:func:`check` runs a program on adversarial inputs and holds every finite value
it computes against what the store claims of it: alone, and paired with each
other value at the same point, in its loop body or around it.  Most
unsoundness the store can harbor reaches no inferred format, so this looks at
the store itself (`docs/todos/digit-bound-inference.md`, Part I, section 5).
"""

import ast as pyast
import importlib.util
import math
import random
import sys
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

import fpy2 as fp
from fpy2.analysis import FormatInfer
from fpy2.analysis.digit_bound import DigitBoundInfer, Term, Terms
from fpy2.ast.fpyast import Expr, ForStmt, FuncDef, ListComp, WhileStmt
from fpy2.ast.visitor import DefaultVisitor
from fpy2.env import ForeignEnv
from fpy2.interpret.byte import BytecodeCompiler
from fpy2.interpret.value import to_value
from fpy2.number import Float
from fpy2.strategies import monomorphize
from fpy2.types import ListType, RealType, Type

_Binder = tuple[int, int]
"""A loop, or a comprehension's clause, as `(id(node), clause)`; a `while`
is `(id(node), -1)`, whose iterations are not counted."""

_Point = tuple[int, ...]
"""A run, and the iteration of each binder around a value."""

_MARK = '__fpy_mark'

L4 = ListType(RealType(fp.FP32), 4)


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

    def _visit_while(self, stmt: WhileStmt, ctx: Any) -> None:
        around, self._around = self._around, (*self._around, (id(stmt), -1))
        super()._visit_while(stmt, ctx)
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

    def __init__(self, func: FuncDef, env: ForeignEnv, hook: Callable[[int, Any], Any]):
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
    rest at its smallest, the reverse, and one zero among the largest, at every
    position, and the first negated; then a few random draws.  Every
    counterexample so far needed a huge element at one index and a tiny one at
    another."""
    def scalar(t: RealType, kind: str) -> Float:
        match kind:
            case 'big':
                return t.fmt.maxval()
            case 'tiny':
                return t.fmt.minval()
            case _:
                return t.fmt.zero()

    n = max((t.length for t in arg_types if isinstance(t, ListType)), default=1)
    runs = []
    for p in range(n):
        for here, rest in (('big', 'tiny'), ('tiny', 'big'), ('zero', 'big')):
            runs.append([
                [scalar(t.elt, here if i == p else rest) for i in range(t.length)]
                if isinstance(t, ListType) else scalar(t, here)
                for t in arg_types
            ])
    runs.append([[-x for x in a] if isinstance(a, list) else -a for a in runs[0]])
    rng = random.Random(seed)

    def draw(t: RealType) -> Float:
        top = t.fmt.to_ordinal(t.fmt.maxval())
        return t.fmt.from_ordinal(rng.randint(-top, top))

    for _ in range(4):
        runs.append([[draw(t.elt) for _ in range(t.length)] if isinstance(t, ListType) else draw(t)
                     for t in arg_types])
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
    enclosing = _Enclosing()
    enclosing._visit_function(ast, None)
    # site -> point -> the values observed there
    seen: dict[int, dict[_Point, list[Float]]] = defaultdict(lambda: defaultdict(list))
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
        try:
            run(*[to_value(a) for a in args], __ctx__=fp.REAL)
        except Exception:  # noqa: BLE001, S112 -- what it computed before stopping still counts
            continue

    store, bad = db.store, []
    # per term and literals, the most it is known to reach and the least not
    known: dict[tuple[Term, frozenset[int]], tuple[float, float]] = {}

    def reaches(t: Term, d: int, lits: frozenset[int]) -> bool:
        """Can *t* be at least *d* where *lits* hold?  A decision, cheaper
        than the maximum, and all a check compares."""
        key = (t, lits)
        yes, no = known.get(key, (-math.inf, math.inf))
        if d <= yes or d >= no:
            return d <= yes
        if store.reaches([(t, d)], lits):
            known[key] = (d, no)
            return True
        known[key] = (yes, d)
        return False

    sites = []
    for i in seen:
        e = compiler.hook_sites[i]
        t = db.by_expr.get(e)
        binders = enclosing.out.get(id(e), ())
        if t is not None and t.msb is not None:
            sites.append((i, e, t, binders, any(c == -1 for _, c in binders)))
    # each value observed non-zero makes its own literals true at its point
    nonzero: dict[tuple[tuple[_Binder, ...], _Point], set[int]] = defaultdict(set)
    for i, _, t, binders, loose in sites:
        if loose:
            continue
        for k, vs in seen[i].items():
            if any(v.is_nonzero() for v in vs):
                nonzero[binders, k].update(store.not_neg_inf(t.msb))

    def holding(k: _Point, binders: tuple[_Binder, ...], *es: Expr) -> frozenset[int]:
        """The literals true at point *k* under *binders*: those of every value
        non-zero there or around it, and what each `with` around *es*
        establishes."""
        out = set().union(*(db.assume_at(e) for e in es))
        for m in range(len(binders) + 1):
            out |= nonzero.get((binders[:m], k[:m + 1]), set())
        return frozenset(out)

    obs = {(i, name): {k: xs for k, vs in seen[i].items()
                       if (xs := [x for v in vs if (x := read(v)) is not None])}
           for i, _, t, _, _ in sites for name, _, read in _readings(t)}

    # each term the store bounds, what was observed of it under each set of
    # literals, and where -- one question per term and literals, however many
    # sites share the variables
    claims: dict[Term, dict[frozenset[int], tuple[int, str]]] = {}

    def claim(t: Term, d: int, lits: frozenset[int], where: str) -> None:
        groups = claims.setdefault(t, {})
        if lits not in groups or d > groups[lits][0]:
            groups[lits] = (d, where)

    for i, e, t, binders, _ in sites:
        for name, term, _ in _readings(t):
            for k, xs in obs[i, name].items():
                lits = holding(k, binders, e)
                if name != 'G':
                    claim(term, max(xs), lits, f'{name}({e.format()})')
                claim(-term, -min(xs), lits, f'-{name}({e.format()})')

    # pairs at one point: in one body, or one around the other
    for i, ea, ta, ba, la in sites:
        for j, eb, tb, bb, lb in sites:
            if i == j or la or lb:
                continue
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
                                  holding(kd, deep, ea, eb), f'{na}({ea.format()}) - {nb}({eb.format()})')

    # the literal sets observing the most, at most `_GROUPS` per term
    for t, groups in claims.items():
        ranked = sorted(groups.items(), key=lambda g: -g[1][0])[:_GROUPS]
        for lits, (d, where) in ranked:
            if not reaches(t, d, lits):
                bad.append(f'{where} = {d}, above every bound')
    return list(dict.fromkeys(bad))


# -- a generator of small programs ----------------------------------------

_ELT = ('{}', '{} * 2', '{} * 1024', '2 ** (-fp.logb({}))')
"""What an element becomes."""

_HEADER = '''\
import fpy2 as fp


@fp.fpy(ctx=fp.REAL)
def _g(zs, y):
    e0 = fp.logb(zs[0])
    with fp.MPFixedContext(e0 - 10, fp.RM.RTZ):
        r = fp.round(y * 2)
    return r


@fp.fpy(ctx=fp.REAL)
def f(xs, ys):
'''


def generate(seed: int) -> str:
    """The source of a module defining `f(xs, ys)`, two lists of four: a few
    blocks drawn from the shapes the store's rules cover -- comprehensions of
    one and two clauses, covering, partial, nested and gather writes, carried
    values, aggregates, parts, roundings at a position, and a call."""
    rng = random.Random(seed)
    lists, scalars, lines = ['xs', 'ys'], [], []

    def name(base: str) -> str:
        return f'{base}{len(lines)}'

    def elt(x: str) -> str:
        return rng.choice(_ELT).format(x)

    def emit(*block: str) -> None:
        lines.extend('    ' + ln for b in block for ln in b.split('\n'))

    for _ in range(rng.randint(2, 3)):
        a, b = rng.choice(lists), rng.choice(lists)
        k = rng.choice((4, 10, 24))
        match rng.randrange(15):
            case 0:
                out = name('l')
                emit(f'{out} = [{elt("x")} for x in {a}]')
                lists.append(out)
            case 1:
                out = name('l')
                emit(f'{out} = [x * y for x, y in zip({a}, {b})]')
                lists.append(out)
            case 2:
                out, zs = name('s'), name('zs')
                emit(f'{zs} = [x * y for x in {a} for y in {b}]', f'{out} = {zs}[{rng.randrange(16)}]')
                scalars.append(out)
            case 3:
                out = name('l')
                emit(f'{out} = fp.empty(4)', 'for i in range(4):',
                     f'    {out}[i] = {a}[i] {rng.choice("+-*")} {b}[i]')
                lists.append(out)
            case 4:
                out = name('l')
                emit(f'{out} = fp.empty(4)', 'for i in range(4):', '    with fp.INTEGER:',
                     '        k = 3 - i', f'    {out}[k] = {elt(f"{a}[i]")}')
                lists.append(out)
            case 5:
                out = name('l')
                emit(f'{out} = fp.empty(4)', 'for i in range(4):', '    for j in range(4):',
                     f'        {out}[i] = {a}[j] * {b}[i]')
                lists.append(out)
            case 6:
                out = name('l')
                emit(f'{out} = fp.empty(4)', 'for r in range(4):', '    for i in range(4):',
                     f'        {out}[i] = {a}[i] * {b}[r]')
                lists.append(out)
            case 7:
                out, acc = name('l'), name('acc')
                emit(f'{acc} = {a}[0]', f'{out} = fp.empty(4)', 'for i in range(4):',
                     f'    prev = {acc}', f'    y = {elt(f"{a}[i]")}', '    e = fp.logb(y)',
                     f'    with fp.MPFixedContext(e - {k}, fp.RM.RTZ):',
                     f'        {out}[i] = fp.round(prev)', f'    {acc} = y')
                lists.append(out)
                scalars.append(acc)
            case 8:
                out = name('l')
                emit(f'{out} = fp.empty(4)', 'for i in range(4):', f'    e = fp.logb({a}[i])',
                     f'    with fp.MPFixedContext(e - {k}, fp.RM.RTZ):',
                     f'        {out}[i] = fp.round({b}[i])')
                lists.append(out)
            case 9:
                out = name('l')
                emit(f'{out} = fp.empty(4)', 'for j in range(4):', f'    e = fp.logb({a}[j])',
                     f'    w = {a}[j]', '    for i in range(4):',
                     f'        with fp.MPFixedContext(e - {k}, fp.RM.RTZ):',
                     f'            w = fp.round({b}[i])', f'    {out}[j] = w')
                lists.append(out)
            case 10:
                out = name('s')
                agg = rng.choice(('max', 'min', 'sum'))
                emit(f'{out} = {agg}({a})', f'{name("e")} = max([fp.logb(x) for x in {a}])')
                scalars.append(out)
            case 11:
                out = name('s')
                emit(f'{out} = max([{a}[t] for t in range(1, 3)])')
                scalars.append(out)
            case 12:
                out, g = name('s'), name('g')
                emit(f'{g} = fp.empty(3)', 'k = 0', 'for t in range(1, 4):',
                     f'    {g}[k] = {elt(f"{a}[t]")}', '    with fp.INTEGER:', '        k = k + 1',
                     f'{out} = max({g})')
                scalars.append(out)
            case 13 if scalars:
                out, s = name('l'), rng.choice(scalars)
                emit(f'{out} = fp.empty(4)', 'for i in range(4):', f'    {out}[i] = {a}[i] + {s}')
                lists.append(out)
            case _:
                out = name('l')
                emit(f'{out} = fp.empty(4)', 'for i in range(4):',
                     f'    {out}[i] = _g({a}, {a}[i]) + {a}[i]')
                lists.append(out)
    emit(f'return {lists[-1]}')
    return _HEADER + '\n'.join(lines) + '\n'


def load(source: str, path: Path) -> fp.Function:
    """`f` of the module *source*, written to *path*: the decorator reads a
    function's source from its file."""
    path.write_text(source)
    name = f'_digit_bound_gen_{abs(hash(path))}'
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # the decorator finds a function's source through its module
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.f
