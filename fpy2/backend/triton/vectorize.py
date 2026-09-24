"""
Triton backend: may a loop body be evaluated as a tile?

Splitting a loop into outer x inner preserves semantics on its own.
Evaluating the inner body as a tile is what can change the answer, and only
through the variables the loop carries from one iteration to the next -- a
body that carries nothing computes each element independently.

Three things have to hold, and each failure is a refusal rather than a
fallback:

- **A carried scalar must combine in a way that regrouping does not observe.**
  `max`, `min`, `and`, `or` select an operand, so any grouping agrees.  `+`,
  `-`, `*` round, and regrouping them agrees only where every step is exact --
  which :func:`~fpy2.analysis.format_infer.rounds_exactly` decides from the
  inferred formats.  Deciding that by *proof* rather than by a fast-math flag
  is what makes the tile reduction bit-identical rather than merely permitted.
- **A list the loop writes must be written at its own index.**  Every
  subscript has to be the loop variable or invariant across the loop, so no
  two iterations touch the same element.  Anything else -- `out[i % 2]`,
  `out[k]`, `out[i + 1]` -- is refused rather than sent to a dependence test.
- **A write that ignores the carried value is refused.**  Which iteration
  wrote last is exactly what a tile does not preserve.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cached_property

from ...analysis import (
    ArraySizeAnalysis,
    ArraySizeInfer,
    ContextUse,
    ContextUseAnalysis,
    DefineUse,
    DefineUseAnalysis,
    FormatAnalysis,
    FormatInfer,
)
from ...analysis.array_size import static_trip_count
from ...analysis.format_infer import rounds_exactly
from ...ast import (
    Add,
    And,
    Assign,
    BoolVal,
    DefaultVisitor,
    Expr,
    ForStmt,
    FuncDef,
    Id,
    If1Stmt,
    IndexedAssign,
    Integer,
    Max,
    Min,
    Mul,
    NamedId,
    Or,
    Stmt,
    Sub,
    TupleBinding,
    Var,
)
from ...number import Context
from ...transform import SplitLoop, SplitLoopStrategy

__all__ = ['TileResult', 'carried_scalars', 'tile_loops', 'why_not_tileable']


_SELECTS: tuple[type[Expr], ...] = (Max, Min, And, Or)
"""Combines that pick an operand, so any grouping gives the same bits."""

_ROUNDS: tuple[type[Expr], ...] = (Add, Sub, Mul)
"""Combines that compute a new value, so regrouping is observable unless every
step is exact."""


def _target_names(target: Id | TupleBinding) -> set[NamedId]:
    """The names a `for` binds each iteration."""
    if isinstance(target, TupleBinding):
        out: set[NamedId] = set()
        for elt in target.elts:
            out |= _target_names(elt)
        return out
    return {target} if isinstance(target, NamedId) else set()


class _Reads(DefaultVisitor):
    """Names an expression reads."""

    def __init__(self):
        super().__init__()
        self.names: set[NamedId] = set()

    def _visit_var(self, e: Var, ctx):
        self.names.add(e.name)


def _reads(e: Expr) -> set[NamedId]:
    v = _Reads()
    v._visit_expr(e, None)
    return v.names


class _Body(DefaultVisitor):
    """Every write to a carried name in one loop body."""

    def __init__(self, carried: set[NamedId]):
        super().__init__()
        self.carried = carried
        self.scalar: dict[NamedId, list[Expr]] = {v: [] for v in carried}
        self.indexed: dict[NamedId, list[IndexedAssign]] = {
            v: [] for v in carried
        }

    def _visit_assign(self, stmt: Assign, ctx):
        if isinstance(stmt.target, NamedId) and stmt.target in self.carried:
            self.scalar[stmt.target].append(stmt.expr)
        return super()._visit_assign(stmt, ctx)

    def _visit_indexed_assign(self, stmt: IndexedAssign, ctx):
        if stmt.var in self.carried:
            self.indexed[stmt.var].append(stmt)
        return super()._visit_indexed_assign(stmt, ctx)


def _same_literal(writes: list[Expr]) -> bool:
    """Whether every write is one and the same literal.

    Then the combine is idempotent, so the order of the writes stops
    mattering: `if p(x): ok = False` leaves `ok` false exactly when some
    iteration's guard fired, whichever one did.  This is an `and`-fold that
    does not look like one, and it is the common shape for a search.
    """
    seen: set[tuple[str, object]] = set()
    for e in writes:
        if isinstance(e, BoolVal):
            seen.add(('bool', e.val))
        elif isinstance(e, Integer):
            seen.add(('int', e.val))
        else:
            return False
    return len(seen) == 1


def _why_scalar_refuses(
    name: NamedId,
    writes: list[Expr],
    ctx_use: ContextUseAnalysis,
    fmt: FormatAnalysis,
) -> str | None:
    """Why *name*'s combine cannot be regrouped, or `None`."""
    if _same_literal(writes):
        return None
    for e in writes:
        if name not in _reads(e):
            return (
                f'`{name}` is written without reading it, so which iteration '
                'wrote last is the answer'
            )
        if isinstance(e, _SELECTS):
            continue
        if not isinstance(e, _ROUNDS):
            return f'`{name}` combines with `{type(e).__name__.lower()}`, ' \
                   'which is not a combine this backend regroups'
        try:
            scope = ctx_use.find_scope_from_use(e)
        except KeyError:
            return f'the context combining `{name}` does not resolve'
        target = scope.ctx if isinstance(scope.ctx, Context) else None
        if not rounds_exactly(e, fmt.by_expr, target):
            return (
                f'`{name}` accumulates with a rounded '
                f'`{type(e).__name__.lower()}`, so regrouping it moves bits'
            )
    return None


def _why_indexed_refuses(
    name: NamedId,
    writes: list[IndexedAssign],
    loop_vars: set[NamedId],
    moved: set[NamedId],
) -> str | None:
    """Why *name*'s element writes may collide across iterations, or `None`."""
    for stmt in writes:
        if name in _reads(stmt.expr):
            return (
                f'`{name}` is read back while being written, which orders the '
                'iterations against each other'
            )
        saw_loop_var = False
        for idx in stmt.indices:
            if isinstance(idx, Var) and idx.name in loop_vars:
                saw_loop_var = True
            elif isinstance(idx, Var) and idx.name not in moved:
                pass                       # invariant across this loop
            else:
                return (
                    f'`{name}` is written at an index this backend cannot '
                    'show distinct per iteration'
                )
        if not saw_loop_var:
            return (
                f'every iteration writes `{name}` at the same index, so the '
                'last write is the answer'
            )
    return None


def why_not_tileable(
    stmt: ForStmt,
    func: FuncDef,
    *,
    def_use: DefineUseAnalysis | None = None,
    ctx_use: ContextUseAnalysis | None = None,
    fmt: FormatAnalysis | None = None,
) -> str | None:
    """Why *stmt*'s body cannot be evaluated as a tile, or `None` if it can.

    *func* is the function *stmt* belongs to; the analyses are taken from it
    and may be passed in when the caller already has them.
    """
    if not isinstance(stmt, ForStmt):
        raise TypeError(f"Expected a 'ForStmt', got {stmt}")
    if def_use is None:
        def_use = DefineUse.analyze(func)
    if ctx_use is None:
        ctx_use = ContextUse.analyze(func, def_use=def_use)
    if fmt is None:
        fmt = FormatInfer.analyze(func)

    carried = def_use.mutated_in(stmt.body)
    if not carried:
        return None

    loop_vars = _target_names(stmt.target)
    moved = carried | def_use.introed_in(stmt.body) | loop_vars

    body = _Body(carried)
    body._visit_block(stmt.body, None)

    for name in sorted(carried):
        scalar, indexed = body.scalar[name], body.indexed[name]
        if scalar and indexed:
            return (
                f'`{name}` is written both whole and by element, and the '
                'order between the two is the answer'
            )
        if indexed:
            why = _why_indexed_refuses(name, indexed, loop_vars, moved)
        elif scalar:
            why = _why_scalar_refuses(name, scalar, ctx_use, fmt)
        else:
            why = f'`{name}` is carried by something this backend cannot read'
        if why is not None:
            return why
    return None


class _ForLoops(DefaultVisitor):
    """Every `for` in visit order, outermost first."""

    def __init__(self):
        super().__init__()
        self.out: list[ForStmt] = []

    def _visit_for(self, stmt: ForStmt, ctx):
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
    """The names *stmt*'s body writes whole and carries to the next iteration:
    what tiling it turns into a reduction across the tile's lanes."""
    carried = def_use.mutated_in(stmt.body)
    body = _Body(carried)
    body._visit_block(stmt.body, None)
    return {name for name in carried if body.scalar[name]}


class _Loops:
    """Every `for` of one function, outermost first, and what tiling asks of
    each.  The analyses run once per function rather than once per loop."""

    def __init__(self, func: FuncDef, reductions: bool):
        self.func = func
        self.reductions = reductions
        self.all = _for_loops(func)
        self.def_use = DefineUse.analyze(func)
        self.ctx_use = ContextUse.analyze(func, def_use=self.def_use)
        self.fmt = FormatInfer.analyze(func)

    @cached_property
    def sizes(self) -> ArraySizeAnalysis:
        return ArraySizeInfer.analyze(self.func)

    def tileable(self, stmt: ForStmt, *, reductions: bool | None = None) -> bool:
        if why_not_tileable(
            stmt, self.func,
            def_use=self.def_use, ctx_use=self.ctx_use, fmt=self.fmt,
        ) is not None:
            return False
        if self.reductions if reductions is None else reductions:
            return True
        return not carried_scalars(stmt, self.def_use)

    def static(self, stmt: ForStmt) -> bool:
        return static_trip_count(stmt.iterable, self.sizes) is not None

    def lane(self, stmt: ForStmt) -> bool:
        """Whether *stmt* runs across a tile's lanes: a static count, a body
        holding no loop, and each iteration writing only its own elements."""
        return (
            self.static(stmt)
            and not _has_loop(stmt)
            and self.tileable(stmt, reductions=False)
        )

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
    def __init__(self):
        super().__init__()
        self.found = False

    def _visit_for(self, stmt, ctx):
        self.found = True

    def _visit_while(self, stmt, ctx):
        self.found = True


def _has_loop(stmt: ForStmt) -> bool:
    v = _HasLoop()
    v._visit_block(stmt.body, None)
    return v.found


def _rows(loops: _Loops, lanes: bool) -> list[ForStmt]:
    """The loops to tile, by default.

    Without *lanes*, the innermost tileable loops.  With it, the innermost
    tileable ones of runtime count, since Triton's grid runs over sizes the
    kernel takes as arguments; failing any, the outermost tileable ones.  The
    loops of static count beneath are a row's lanes, or sequential.
    """
    tileable = [s for s in loops.all if loops.tileable(s)]
    if not lanes:
        return loops.innermost(tileable)
    runtime = [s for s in tileable if not loops.static(s)]
    return loops.innermost(runtime) if runtime else loops.outermost(tileable)


def _lanes(func: FuncDef, tiled: list[ForStmt]) -> list[ForStmt]:
    """The lane loops beneath each tile, in visit order."""
    loops = _Loops(func, reductions=False)
    return [s for t in tiled for s in loops.beneath(t) if loops.lane(s)]


def _grid(func: FuncDef, tiled: list[ForStmt]) -> list[ForStmt]:
    """The loop directly around a lone tile, where nothing encloses it and it
    carries nothing: its iterations are the grid's second axis, as a matmul's
    rows are."""
    if len(tiled) != 1:
        return []
    loops = _Loops(func, reductions=False)
    around = [s for s in loops.all if loops.within(tiled[0], s)]
    return around if len(around) == 1 and loops.tileable(around[0]) else []


@dataclass
class TileResult:
    """What :func:`tile_loops` did."""

    func: FuncDef
    """The rewritten function."""

    tiled: list[ForStmt]
    """The outer chunk loop of each tile, in visit order.

    The emitter needs to know which loop carries a tile, and recognizing one
    by its *shape* would be pattern-matching this pass's output -- brittle,
    and wrong the moment the shape changes.  This pass knows, so it says.
    """

    guards: list[If1Stmt]
    """The `j < n` guard `SplitLoop` put around each tile's body, where it
    emitted one.  Named for the same reason: the emitter lowers a guard as the
    tile's mask, and any other `if` as a branch."""

    lanes: list[ForStmt] | None = None
    """The loops beneath the tiles that run across a tile's lanes, unsplit;
    `None` where lanes were not asked for."""

    grid: list[ForStmt] = field(default_factory=list)
    """The loop the grid's second axis takes, one iteration per program,
    where there is one (:func:`_grid`)."""

    def rewritten(self, func: FuncDef) -> 'TileResult':
        """This result for *func*, a rewrite of :attr:`func` that preserves
        what it computes.  A rewrite rebuilds the nodes, so each tile is found
        again by its outer loop's target, a name this pass minted and no
        cleanup renames, and its guard as the split left it.  Lanes are found
        again by the rule that found them."""
        names = [t.target for t in self.tiled]
        by_name = {
            loop.target: loop for loop in _for_loops(func)
            if loop.target in names
        }
        missing = [str(n) for n in names if n not in by_name]
        if missing:
            raise RuntimeError(f'a rewrite lost the tiled loop over {", ".join(missing)}')
        tiled = [by_name[n] for n in names]
        guards = [g for t in tiled if (g := _guard(_tile_loop(t))) is not None]
        lanes = None if self.lanes is None else _lanes(func, tiled)
        grid = _grid(func, tiled) if self.grid else []
        return TileResult(func, tiled, guards, lanes, grid)


def tile_loops(
    func: FuncDef,
    width: int | str,
    *,
    reductions: bool = True,
    lanes: bool = False,
    rows: Sequence[ForStmt] | None = None,
) -> TileResult:
    """*func* with each *innermost* tileable loop split into chunks of
    *width*.

    **The innermost loop of a nest carries the tile**, and the ones enclosing
    it are left as loops.  That is the shape both Triton idioms take: a fused
    softmax makes the row dimension the program instance and the columns the
    tile, and a matmul takes its block indices from the program id and loops
    over tiles of `K`.  Tiling an enclosing loop as well would give a nest of
    tiles where the target wants one tiled dimension.

    With *lanes*, a tile's rows are its outputs and the loops of static count
    beneath it that write only their own elements are its **lanes**, left
    unsplit and reported in :attr:`TileResult.lanes`; see :func:`_rows` for
    which loops are rows then.  *rows* overrides that choice with loops of
    *func*.

    *width* is a literal, or the **name of a free variable** holding it.  The
    name is what a target wants: a tile's width is a compile-time constant
    parameter of the kernel, chosen by the launcher rather than fixed in the
    program.  A literal keeps the loop runnable against the interpreter, and a
    free one does too -- the interpreter simply takes a value for it, so a
    differential can vary the width instead of pinning one.

    A loop :func:`why_not_tileable` refuses is left alone: it stays sequential,
    which is still parallel across whatever encloses it.  So is one carrying
    a scalar, where *reductions* is false -- for a target with no lowering of
    a reduction across the tile.

    ``MASK`` is the remainder policy because a tile has to be a constant width
    -- ``PEEL`` would emit a second, narrower body for the tail and ``STRICT``
    would refuse a length the factor does not divide.
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

    loops = _Loops(func, reductions)
    picks = _rows(loops, lanes) if rows is None else list(rows)
    if any(not any(p is s for s in loops.all) for p in picks):
        raise ValueError('a row is not a loop of this function')
    if len(loops.innermost(picks)) != len(picks):
        raise ValueError('a row encloses another; one axis tiles at a time')

    # Positions, not nodes: each split rebuilds the AST.  A split at `i`
    # leaves the outer loop at `i` and its tile loop at `i + 1`, shifting what
    # follows by one; no pick lies inside another, so none moves otherwise.
    at = sorted(next(k for k, s in enumerate(loops.all) if s is p) for p in picks)
    tiled: list[int] = []
    for shift, i in enumerate(at):
        func = SplitLoop.apply(
            func, factor, i + shift, strategy=SplitLoopStrategy.MASK,
        )
        tiled.append(i + shift)
    final = _for_loops(func)
    outers = [final[k] for k in tiled]
    guards = [g for k in tiled if (g := _guard(final[k + 1]))]
    if not lanes:
        return TileResult(func, outers, guards)
    return TileResult(func, outers, guards, _lanes(func, outers), _grid(func, outers))
