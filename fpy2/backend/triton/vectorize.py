"""
Triton backend: may a loop body be evaluated as a tile?

Splitting a loop into outer x inner preserves semantics on its own.
Evaluating the inner body as a tile is what can change the answer, and only
through the variables the loop carries from one iteration to the next -- a
body that carries nothing computes each element independently.

Two things have to hold, and each failure is a refusal rather than a
fallback:

- **Nothing is carried whole.**  That would be a reduction across the loop,
  which the emitter does not lower.
- **A list the loop writes must be written at its own index.**  Every
  subscript up to the loop variable's has to be invariant across the loop, so
  no two iterations touch the same element, and every read of that list,
  through any name, must be under it.  Anything else -- `out[i % 2]`,
  `out[k]`, `out[i + 1]`, `t = out[n - 1 - i]` -- is refused rather than sent
  to a dependence test.  A write through `row = out[i]` is a write to `out`.
"""

from dataclasses import dataclass
from functools import cached_property

from ...analysis import (
    Alias,
    AliasAnalysis,
    ArraySizeAnalysis,
    ArraySizeInfer,
    DefineUse,
    DefineUseAnalysis,
)
from ...analysis.array_size import static_trip_count
from ...analysis.reaching_defs import Definition, same_object_defs
from ...ast import (
    Argument,
    Assign,
    DefaultVisitor,
    Empty,
    Expr,
    ForStmt,
    FuncDef,
    If1Stmt,
    IndexedAssign,
    Integer,
    ListComp,
    ListExpr,
    ListRef,
    ListSlice,
    NamedId,
    Range1,
    Range2,
    Range3,
    Stmt,
    Var,
    WhileStmt,
)
from ...transform import SplitLoop, SplitLoopStrategy
from ...transform.simplify_if import _reads

__all__ = ['TileResult', 'carried_scalars', 'tile_loops', 'why_not_tileable']


class _Body(DefaultVisitor):
    """The carried names one loop body writes whole, and by element."""

    carried: set[NamedId]
    scalar: set[NamedId]
    indexed: set[NamedId]

    def __init__(self, carried: set[NamedId]) -> None:
        super().__init__()
        self.carried = carried
        self.scalar = set()
        self.indexed = set()

    def _visit_assign(self, stmt: Assign, ctx: None) -> None:
        if isinstance(stmt.target, NamedId) and stmt.target in self.carried:
            self.scalar.add(stmt.target)
        return super()._visit_assign(stmt, ctx)

    def _visit_indexed_assign(self, stmt: IndexedAssign, ctx: None) -> None:
        if stmt.var in self.carried:
            self.indexed.add(stmt.var)
        return super()._visit_indexed_assign(stmt, ctx)


_LOOP = object()
"""The key of an index that is the loop's own variable."""

_ALLOCS: tuple[type[Expr], ...] = (Empty, ListExpr, ListComp, ListSlice)
"""What makes a fresh list: a slice copies."""


class _Accesses(DefaultVisitor):
    """A loop body's element writes, the ids of the nodes that may define a
    name in it, and what it reads, a subscript chain taken whole."""

    writes: list[IndexedAssign]
    sites: set[int]
    reads: list[Expr]

    def __init__(self) -> None:
        super().__init__()
        self.writes, self.sites, self.reads = [], set(), []

    def _visit_statement(self, stmt: Stmt, ctx: None) -> None:
        self.sites.add(id(stmt))
        if isinstance(stmt, IndexedAssign):
            self.writes.append(stmt)
        return super()._visit_statement(stmt, ctx)

    def _visit_list_comp(self, e: ListComp, ctx: None) -> None:
        self.sites.add(id(e))
        return super()._visit_list_comp(e, ctx)

    def _visit_var(self, e: Var, ctx: None) -> None:
        self.reads.append(e)

    def _visit_list_ref(self, e: ListRef, ctx: None) -> None:
        self.reads.append(e)
        base: Expr = e
        while isinstance(base, ListRef):
            self._visit_expr(base.index, ctx)
            base = base.value
        if not isinstance(base, Var):
            self._visit_expr(base, ctx)


def _peel(e: Expr) -> tuple[Expr, tuple[Expr, ...]]:
    """*e* as a base and the indices a subscript chain applies to it."""
    idx: tuple[Expr, ...] = ()
    while isinstance(e, ListRef):
        idx, e = (e.index, *idx), e.value
    return e, idx


def _root(
    d: Definition, def_use: DefineUseAnalysis,
) -> tuple[Definition, tuple[Expr, ...]] | None:
    """The one list *d* is part of, and the indices to it, through `x = y[i]`
    and the updates and phis that name the same list; `None` if unclear."""
    prefix: tuple[Expr, ...] = ()
    while True:
        bases, stack, seen = set(), [d], set()
        while stack:
            c = stack.pop()
            if c not in seen:
                seen.add(c)
                prev = same_object_defs(c)
                stack.extend(def_use.defs[i] for i in prev)
                if not prev:
                    bases.add(c)
        if len(bases) != 1:
            return None
        (d,) = bases
        if not isinstance(d.site, Assign):
            return d, prefix
        base, idx = _peel(d.site.expr)
        if not isinstance(base, Var):
            return d, prefix
        d, prefix = def_use.find_def_from_use(base), idx + prefix


def _why_writes_refuse(
    stmt: ForStmt, def_use: DefineUseAnalysis, alias: AliasAnalysis,
) -> str | None:
    """Why the element writes of *stmt*'s body may collide across iterations,
    or `None`.  A write's indices must reach the loop variable through
    invariant ones, and every read of that list must be at that element."""
    body = _Accesses()
    body._visit_block(stmt.body, None)
    ranged = isinstance(stmt.iterable, (Range1, Range2, Range3))

    def key(i: Expr) -> object | None:
        if isinstance(i, Integer):
            return ('int', i.val)
        if not isinstance(i, Var):
            return None
        d = def_use.find_def_from_use(i)
        if d.site is stmt:
            return _LOOP if ranged else None
        return None if id(d.site) in body.sites else d

    held: dict[Definition, tuple[object, ...]] = {}
    for w in body.writes:
        found = _root(def_use.find_def_from_use(w), def_use)
        if found is None:
            return f'`{w.var}` may be a list this backend cannot follow'
        root, prefix = found
        if isinstance(root.site, Assign) and isinstance(root.site.expr, _ALLOCS):
            if id(root.site) in body.sites:
                continue                    # allocated each iteration
        elif not isinstance(root.site, Argument):
            return f'`{w.var}` may be a list this backend cannot follow'
        keys = [key(i) for i in (*prefix, *w.indices)]
        if _LOOP not in keys:
            if None in keys:
                return (f'`{w.var}` is written at an index this backend '
                        'cannot show distinct per iteration')
            return (f'every iteration writes `{w.var}` at the same index, so '
                    'the last write is the answer')
        n = keys.index(_LOOP) + 1
        if None in keys[:n]:
            return (f'`{w.var}` is written at an index this backend cannot '
                    'show distinct per iteration')
        if held.setdefault(root, tuple(keys[:n])) != tuple(keys[:n]):
            return f'`{w.var}` is written at two indices, which may collide'

    regions = {alias.region_of(r) for r in held} - {None}
    for e in body.reads:
        base, idx = _peel(e)
        found = _root(def_use.find_def_from_use(base), def_use) \
            if isinstance(base, Var) else None
        if found is None:
            if alias.region_of_expr(e) in regions:
                return ('a list the loop writes is read through a name this '
                        'backend cannot follow')
            continue
        root, prefix = found
        at = tuple(key(i) for i in (*prefix, *idx))
        for r, path in held.items():
            if (root is r or alias.may_alias(root, r)) and at[:len(path)] != path:
                return (f'`{r.name}` is read back while being written, which '
                        'orders the iterations against each other')
    return None


def why_not_tileable(
    stmt: ForStmt,
    func: FuncDef,
    *,
    def_use: DefineUseAnalysis | None = None,
) -> str | None:
    """Why *stmt*'s body cannot be evaluated as a tile, or `None` if it can.

    *func* is the function *stmt* belongs to; *def_use* is its analysis, if
    the caller already has it.
    """
    if not isinstance(stmt, ForStmt):
        raise TypeError(f"Expected a 'ForStmt', got {stmt}")
    if def_use is None:
        def_use = DefineUse.analyze(func)

    carried = def_use.mutated_in(stmt.body)
    body = _Body(carried)
    body._visit_block(stmt.body, None)
    if body.scalar:
        return (f'`{min(body.scalar)}` is carried whole, which needs a '
                'reduction across the loop')
    if unread := carried - body.indexed:
        return f'`{min(unread)}` is carried by something this backend cannot read'
    return _why_writes_refuse(stmt, def_use, Alias.analyze(func, def_use=def_use))


class _ForLoops(DefaultVisitor):
    """Every `for` in visit order, outermost first."""

    out: list[ForStmt]

    def __init__(self) -> None:
        super().__init__()
        self.out = []

    def _visit_for(self, stmt: ForStmt, ctx: None) -> None:
        self.out.append(stmt)
        return super()._visit_for(stmt, ctx)


def _for_loops(func: FuncDef) -> list[ForStmt]:
    v = _ForLoops()
    v._visit_function(func, None)
    return v.out


def _guard(inner: ForStmt) -> If1Stmt | None:
    """The guard `SplitLoop` wrapped the tile loop *inner*'s body in, if any.

    Exact rather than a shape match: the tile index is fresh, and only the
    guard `SplitLoop` emits reads it -- the loop's own target is bound from it.
    """
    stmts = inner.body.stmts
    if len(stmts) != 1 or not isinstance(stmts[0], If1Stmt):
        return None
    if not isinstance(inner.target, NamedId):
        return None
    return stmts[0] if inner.target in _reads(stmts[0].cond) else None


def _tile_loop(outer: ForStmt) -> ForStmt:
    """The tile loop `SplitLoop` put in *outer*'s body."""
    loops = [s for s in outer.body.stmts if isinstance(s, ForStmt)]
    if len(loops) != 1:
        raise RuntimeError(f'the tiled loop over {outer.target} holds {len(loops)} loops')
    return loops[0]


def carried_scalars(stmt: ForStmt, def_use: DefineUseAnalysis) -> set[NamedId]:
    """The names *stmt*'s body writes whole and carries to the next iteration."""
    carried = def_use.mutated_in(stmt.body)
    body = _Body(carried)
    body._visit_block(stmt.body, None)
    return body.scalar


class _Loops:
    """Every `for` of one function, outermost first, and what tiling asks of
    each.  The analyses run once per function rather than once per loop."""

    func: FuncDef
    all: list[ForStmt]
    def_use: DefineUseAnalysis

    def __init__(self, func: FuncDef) -> None:
        self.func = func
        self.all = _for_loops(func)
        self.def_use = DefineUse.analyze(func)

    @cached_property
    def sizes(self) -> ArraySizeAnalysis:
        return ArraySizeInfer.analyze(self.func)

    def tileable(self, stmt: ForStmt) -> bool:
        return why_not_tileable(stmt, self.func, def_use=self.def_use) is None

    def static(self, stmt: ForStmt) -> bool:
        return static_trip_count(stmt.iterable, self.sizes) is not None

    def lane(self, stmt: ForStmt) -> bool:
        """Whether *stmt* runs across a tile's lanes: a static count, a body
        holding no loop, and each iteration writing only its own elements."""
        return self.static(stmt) and not _has_loop(stmt) and self.tileable(stmt)

    def beneath(self, stmt: ForStmt) -> list[ForStmt]:
        v = _ForLoops()
        v._visit_block(stmt.body, None)
        return v.out

    def within(self, a: ForStmt, b: ForStmt) -> bool:
        return any(a is c for c in self.beneath(b))

    def innermost(self, picks: list[ForStmt]) -> list[ForStmt]:
        """The picks enclosing no other pick."""
        return [p for p in picks if not any(self.within(q, p) for q in picks)]

    def outermost(self, picks: list[ForStmt]) -> list[ForStmt]:
        """The picks no other pick encloses."""
        return [p for p in picks if not any(self.within(p, q) for q in picks)]


class _HasLoop(DefaultVisitor):
    """Whether a block holds a loop."""

    found: bool

    def __init__(self) -> None:
        super().__init__()
        self.found = False

    def _visit_for(self, stmt: ForStmt, ctx: None) -> None:
        self.found = True

    def _visit_while(self, stmt: WhileStmt, ctx: None) -> None:
        self.found = True


def _has_loop(stmt: ForStmt) -> bool:
    v = _HasLoop()
    v._visit_block(stmt.body, None)
    return v.found


def _rows(loops: _Loops) -> list[ForStmt]:
    """The loops to tile: the innermost tileable ones of runtime count, since
    Triton's grid runs over sizes the kernel takes as arguments; failing any,
    the outermost tileable ones.  The loops of static count beneath are a
    row's lanes, or sequential.
    """
    tileable = [s for s in loops.all if loops.tileable(s)]
    runtime = [s for s in tileable if not loops.static(s)]
    return loops.innermost(runtime) if runtime else loops.outermost(tileable)


def _lanes(loops: _Loops, tiled: list[ForStmt]) -> list[ForStmt]:
    """The lane loops beneath each tile, in visit order."""
    return [s for t in tiled for s in loops.beneath(t) if loops.lane(s)]


def _grid(loops: _Loops, tiled: list[ForStmt]) -> list[ForStmt]:
    """The loop directly around a lone tile, where nothing encloses it and it
    carries nothing: its iterations are the grid's second axis, as a matmul's
    rows are."""
    if len(tiled) != 1:
        return []
    around = [s for s in loops.all if loops.within(tiled[0], s)]
    return around if len(around) == 1 and loops.tileable(around[0]) else []


@dataclass
class TileResult:
    """What :func:`tile_loops` did."""

    func: FuncDef
    """The rewritten function."""

    tiled: list[ForStmt]
    """The outer chunk loop of each tile, in visit order."""

    guards: list[If1Stmt]
    """The `j < n` guard `SplitLoop` put around each tile's body, where it
    emitted one: the emitter lowers it as the tile's mask."""

    lanes: list[ForStmt]
    """The loops beneath the tiles that run across a tile's lanes, unsplit."""

    grid: list[ForStmt]
    """The loop the grid's second axis takes, one iteration per program,
    where there is one (:func:`_grid`)."""

    def rewritten(self, func: FuncDef) -> 'TileResult':
        """This result for *func*, a rewrite of :attr:`func` that preserves
        what it computes: each tile found again by its outer loop's target, a
        name this pass minted."""
        names = [t.target for t in self.tiled]
        loops = _Loops(func)
        by_name = {loop.target: loop for loop in loops.all if loop.target in names}
        missing = [str(n) for n in names if n not in by_name]
        if missing:
            raise RuntimeError(f'a rewrite lost the tiled loop over {", ".join(missing)}')
        tiled = [by_name[n] for n in names]
        guards = [g for t in tiled if (g := _guard(_tile_loop(t))) is not None]
        grid = _grid(loops, tiled) if self.grid else []
        return TileResult(func, tiled, guards, _lanes(loops, tiled), grid)


def tile_loops(func: FuncDef, width: int | str) -> TileResult:
    """*func* with each loop :func:`_rows` picks split into tiles of *width*
    rows, the remainder masked.

    One dimension is tiled: the loops enclosing a tile stay loops, or
    become the grid's second axis (:func:`_grid`).  The loops of static count beneath a tile that write only
    their own elements are its **lanes**, left unsplit.  A loop
    :func:`why_not_tileable` refuses stays sequential.

    *width* is a literal, or the name of a free variable holding it -- the
    kernel's ``tl.constexpr``, for which the interpreter takes a value.
    """
    if not isinstance(func, FuncDef):
        raise TypeError(f"Expected a 'FuncDef', got {func}")
    match width:
        case int() if width >= 1:
            factor: Expr = Integer(width, None)
        case int():
            raise ValueError(f'Expected a positive width, got {width}')
        case str():
            factor = Var(NamedId(width), None)
        case _:
            raise TypeError(f"Expected an 'int' or 'str' width, got {width}")

    loops = _Loops(func)
    # Positions, not nodes: each split rebuilds the AST.  A split at `i`
    # leaves the outer loop at `i` and its tile loop at `i + 1`, shifting what
    # follows by one; no pick lies inside another, so none moves otherwise.
    at = sorted(next(k for k, s in enumerate(loops.all) if s is p) for p in _rows(loops))
    tiled = [i + shift for shift, i in enumerate(at)]
    for k in tiled:
        func = SplitLoop.apply(func, factor, k, strategy=SplitLoopStrategy.MASK)
    after = _Loops(func)
    outers = [after.all[k] for k in tiled]
    guards = [g for k in tiled if (g := _guard(after.all[k + 1]))]
    return TileResult(func, outers, guards, _lanes(after, outers), _grid(after, outers))
