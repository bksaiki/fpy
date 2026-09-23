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

from dataclasses import dataclass

from ...analysis import (
    ContextUse,
    ContextUseAnalysis,
    DefineUse,
    DefineUseAnalysis,
    FormatAnalysis,
    FormatInfer,
)
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

__all__ = ['TileResult', 'tile_loops', 'why_not_tileable']


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


def _encloses_tileable(stmt: ForStmt, func: FuncDef) -> bool:
    """Whether a tileable loop sits beneath *stmt*."""
    v = _ForLoops()
    v._visit_block(stmt.body, None)
    return any(why_not_tileable(c, func) is None for c in v.out)


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


def tile_loops(func: FuncDef, width: int | str) -> TileResult:
    """*func* with each *innermost* tileable loop split into chunks of
    *width*.

    **The innermost loop of a nest carries the tile**, and the ones enclosing
    it are left as loops.  That is the shape both Triton idioms take: a fused
    softmax makes the row dimension the program instance and the columns the
    tile, and a matmul takes its block indices from the program id and loops
    over tiles of `K`.  Tiling an enclosing loop as well would give a nest of
    tiles where the target wants one tiled dimension.

    *width* is a literal, or the **name of a free variable** holding it.  The
    name is what a target wants: a tile's width is a compile-time constant
    parameter of the kernel, chosen by the launcher rather than fixed in the
    program.  A literal keeps the loop runnable against the interpreter, and a
    free one does too -- the interpreter simply takes a value for it, so a
    differential can vary the width instead of pinning one.

    A loop :func:`why_not_tileable` refuses is left alone: it stays sequential,
    which is still parallel across whatever encloses it.

    ``MASK`` is the remainder policy because a tile has to be a constant width
    -- ``PEEL`` would emit a second, narrower body for the tail and ``STRICT``
    would refuse a length the factor does not divide.

    Splitting rewrites the loop into a nest, so the indices of everything after
    it shift; the scan therefore resumes past the pair it just created rather
    than restarting.
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

    # Positions, not nodes: each split rebuilds the AST, so a node captured
    # after one is not in the function after the next.  An index survives --
    # a split at `i` leaves the outer loop at `i` and only shifts what
    # follows, and the scan never returns below `i`.
    tiled: list[int] = []
    i = 0
    while True:
        loops = _for_loops(func)
        if i >= len(loops):
            guards = [g for k in tiled if (g := _guard(loops[k + 1]))]
            return TileResult(func, [loops[k] for k in tiled], guards)
        stmt = loops[i]
        if (why_not_tileable(stmt, func) is not None
                or _encloses_tileable(stmt, func)):
            i += 1
            continue
        func = SplitLoop.apply(
            func, factor, i, strategy=SplitLoopStrategy.MASK,
        )
        # the split left an outer/inner pair where one loop was; the outer is
        # the one that carries the tile
        tiled.append(i)
        i += 2
