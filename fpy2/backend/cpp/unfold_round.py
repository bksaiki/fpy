"""
cpp backend: which roundings the op table cannot spell.

The emitter refuses a rounding under a context its op table does not dispatch
on, and every refusal names the operator that would fix it.  This module asks
the same question early enough to act on it, so the answer is a program point
to rewrite rather than a message to print.

The classification belongs to the backend because :func:`is_native_ctx` does.
The operators it names are backend-independent, and none of them changes.
"""

from dataclasses import dataclass
from enum import Enum, auto

from ...analysis import ContextUse, DefineUse
from ...ast.fpyast import (
    Assign,
    BinaryOp,
    Cast,
    ContextStmt,
    Expr,
    ForeignVal,
    FuncDef,
    NamedId,
    ReturnStmt,
    Round,
    Stmt,
    StmtBlock,
    TernaryOp,
    UnaryOp,
    UnderscoreId,
    Var,
)
from ...ast.visitor import DefaultTransformVisitor
from ...number import (
    REAL,
    RM,
    IEEEContext,
    MPBFixedContext,
    MPFixedContext,
    OverflowMode,
)
from ...number.context.context import Context
from ...transform import (
    Cursor,
    ExprCursor,
    FloatToFixed,
    RescaleFixed,
    SplitRound,
    TransformDeclined,
    UnfoldOverflow,
    UnfoldSpecial,
)
from ...transform.cursor import expr_sites
from ...transform.path import resolve_stmt
from .target import is_native_ctx, make_op_table

__all__ = ['UnfoldKind', 'UnfoldMode', 'UnfoldSite', 'sites', 'unfold', 'unfold_arith']

_TABLE = make_op_table()

_FIXED = (MPFixedContext, MPBFixedContext)


class UnfoldMode(Enum):
    """How much of an unsupported rounding the compiler rewrites rather than
    refuses.

    - ``NONE``: refuse, and name the operator that would fix it.
    - ``ROUNDINGS``: lower a rounding the op table cannot spell into integer
      arithmetic.  Arithmetic *under* such a context still refuses: rewriting it
      means rounding twice, which is a different claim.
    - ``DOUBLE_ROUND``: also compute that arithmetic at a native intermediate
      and re-round, where the correct-double-rounding rules say the two compose
      to what the one gave.  The intermediate always rounds to nearest -- see
      `docs/todos/rounding-recovery.md` for why, and for why a round-to-odd
      level would not close.
    """

    NONE = 0
    ROUNDINGS = 1
    DOUBLE_ROUND = 2


class UnfoldKind(Enum):
    """What is unsupported at a site, which is also which recovery it takes."""

    ARITH = auto()
    """An operation the op table has no signature for under this context.
    Recovered by computing at a native intermediate and re-rounding."""

    FLOAT_ROUND = auto()
    """A rounding to a float context with no C++ analogue: its storage
    *contains* the format rather than equalling it, so a cast rounds to the
    storage's own format.  Recovered by lowering the rounding to fixed-point."""

    FIXED_ROUND = auto()
    """A rounding to a fixed-point context the emitter cannot lower as it
    stands -- its digits are away from position zero, or its bound has a rule
    other than an assertion."""


@dataclass(frozen=True)
class UnfoldSite:
    """One program point the emitter would refuse, and why."""

    cursor: ExprCursor
    kind: UnfoldKind
    ctx: Context
    """The active context that made it a site."""


class _Scopes:
    """The active context per expression.

    `RoundingScopes` answers the same question and also infers formats, which
    this cannot: it runs *before* the rewrite that makes format inference
    succeed on these programs.
    """

    def __init__(self, func: FuncDef):
        self.ctx_use = ContextUse.analyze(func, def_use=DefineUse.analyze(func))

    def __call__(self, e: Expr) -> Context | None:
        """*e*'s active context, or `None` where the scope stays symbolic."""
        scope = self.ctx_use.find_scope_from_use(e)   # type: ignore[arg-type]
        return scope.ctx if isinstance(scope.ctx, Context) else None


def _dispatches(e: Expr) -> bool:
    """Whether the op table is what emits *e*.

    Its keys are the definition: a node it does not key reaches the emitter
    another way -- `Min` and `Max` select an operand rather than rounding, `Len`
    is exact -- so it has no signature to miss.
    """
    match e:
        case UnaryOp():
            return type(e) in _TABLE.unary
        case BinaryOp():
            return type(e) in _TABLE.binary
        case TernaryOp():
            return type(e) in _TABLE.ternary
        case _:
            return False


def _fixed_is_lowerable(ctx: MPFixedContext | MPBFixedContext) -> bool:
    """Whether `_emit_integral_round` lowers *ctx* as it stands.

    Its digits at position zero (``nmin == -1`` is the last unrepresentable
    one), no random bits, and either unbounded or asserting its bound.  Read
    from the fields alone, so no analysis is needed to ask.
    """
    if ctx.nmin != -1 or ctx.num_randbits != 0:
        return False
    return (
        not isinstance(ctx, MPBFixedContext)
        or ctx.overflow is OverflowMode.ASSERT
    )


def _classify(e: Expr, active_of: _Scopes) -> tuple[UnfoldKind, Context] | None:
    """*e*'s kind and the context that gives it one, or `None` where the
    emitter needs no help."""
    if isinstance(e, Round | Cast):
        active = active_of(e)
        if active is None or is_native_ctx(active):
            return None
        if isinstance(active, _FIXED):
            if _fixed_is_lowerable(active):
                return None
            return UnfoldKind.FIXED_ROUND, active
        return UnfoldKind.FLOAT_ROUND, active
    if _dispatches(e):
        # `REAL` is the one non-native context the table reaches, by widening to
        # an op that gives the exact result and rounds to itself.
        active = active_of(e)
        if active is None or active is REAL or is_native_ctx(active):
            return None
        return UnfoldKind.ARITH, active
    return None


def sites(func: FuncDef, within: Cursor | None = None) -> list[UnfoldSite]:
    """The program points of *func* the emitter would refuse, in visit order.

    *func* is a specialized :class:`FuncDef`, before the analyses the emitter
    runs on.  `within` keeps the sites at or beneath the point it names.
    """
    if not isinstance(func, FuncDef):
        raise TypeError(f'Expected \'FuncDef\', got {func}')
    active_of = _Scopes(func)
    out: list[UnfoldSite] = []
    for cursor in expr_sites(
        func, lambda e: _classify(e, active_of) is not None, within,
    ):
        got = _classify(cursor.resolve(), active_of)
        assert got is not None
        out.append(UnfoldSite(cursor, *got))
    return out


def _intermediates() -> list[Context]:
    """Native contexts to offer as an intermediate, narrowest first.

    Narrowest because the intermediate's width becomes the arithmetic's
    storage, and a wider one is never *less* admissible, so the order costs
    nothing.

    Round-to-nearest only.  It is the mode the per-operation rules take, and
    the exactness rule takes any mode, so the two rules that matter here are
    both reached.  It is also the mode the machine is already in: an
    intermediate rounding some other way would put an ``fesetround`` boundary
    around arithmetic whose whole purpose is to be the native one.  Filtered by
    :func:`is_native_ctx`, so a mode the backend stops dispatching on stops
    being offered.
    """
    return [
        cand
        for es, nbits in ((8, 32), (11, 64))
        if is_native_ctx(cand := IEEEContext(es, nbits, RM.RNE))
    ]


def _split_arith(func: FuncDef, site: UnfoldSite) -> FuncDef | None:
    """*func* with *site*'s operation computed at a native intermediate and
    re-rounded to the target, or `None` where no intermediate is admissible.

    `SplitRound` owns the soundness -- it holds the correct-double-rounding
    rules and refuses what they do not cover -- so this only proposes.  Which
    is why the candidates are *native* contexts and not
    :func:`derive_intermediate`'s: that one is deliberately unbounded, so the
    composition agrees at the ends of the range, but unbounded arithmetic is no
    more emittable than the target's own.

    A refusal is an ordinary outcome: an operation with no rule keeps the
    refusal it has.
    """
    for cand in _intermediates():
        try:
            return SplitRound.apply(func, cand, where=site.cursor)
        except TransformDeclined:
            continue
    return None


def _arith(func: FuncDef) -> list[UnfoldSite]:
    return [s for s in sites(func) if s.kind is UnfoldKind.ARITH]


def _step(func: FuncDef, todo: list[UnfoldSite]) -> FuncDef | None:
    for site in todo:
        out = _split_arith(func, site)
        if out is not None:
            return out
    return None


def unfold_arith(func: FuncDef) -> FuncDef:
    """*func* with every arithmetic site the op table cannot spell computed at
    a native intermediate instead.

    Operand formats are the precondition: the per-operation rules hold only for
    operands the *target* represents, so an argument carrying no context of its
    own refuses every candidate.  `Specialize` pins them in the compiler's
    pipeline; a caller reaching this directly runs `monomorphize` first.

    Sites are re-derived after each rewrite rather than forwarded: the rewrite
    lifts its operation into a new block, so the cursors below it move.
    """
    todo = _arith(func)
    while todo:
        out = _step(func, todo)
        if out is None:
            return func
        func = out
        left = _arith(func)
        # the operation lands under a native context and the rounding it gains
        # is to the target, which is a rounding site rather than an arithmetic
        # one -- so this is what makes the loop finite
        assert len(left) < len(todo), 'a split left as much arithmetic as it found'
        todo = left
    return func


def _isolatable(stmt: Stmt, e: Expr) -> bool:
    """Whether wrapping *stmt* in a `with` gives the ladder a rounding block.

    :func:`fpy2.transform.utils.rounding_block`'s condition for a single
    statement: an unannotated assign to a name, or a return, whose whole
    expression rounds a *variable*.  Anything else keeps its refusal -- and a
    rounding of a literal never gets here, the emitter folding those.
    """
    match stmt:
        case Assign(target=NamedId(), type=None) | ReturnStmt():
            return stmt.expr is e and isinstance(e.arg, Var)   # type: ignore[attr-defined]
        case _:
            return False


class _Isolate(DefaultTransformVisitor):
    """Puts each named statement in a `with` block of its own."""

    def __init__(self, blocks: dict[int, Context]):
        self.blocks = blocks
        """the statement to wrap, by identity, and the context to wrap it in"""

    def apply(self, func: FuncDef) -> FuncDef:
        return self._visit_function(func, None)

    def _visit_block(self, block: StmtBlock, ctx):
        out: list[Stmt] = []
        for stmt in block.stmts:
            want = self.blocks.get(id(stmt))
            new, ctx = self._visit_statement(stmt, ctx)
            if want is not None:
                new = ContextStmt(
                    UnderscoreId(), ForeignVal(want, new.loc),
                    StmtBlock([new]), new.loc,
                )
            out.append(new)
        return StmtBlock(out), ctx


def _isolate(func: FuncDef, todo: list[UnfoldSite]) -> FuncDef:
    """*func* with each of *todo*'s roundings inside a block of its own.

    Every pass of the ladder takes a *rounding block*, and a specialized
    function has none: `Specialize` folds a block whose context is the
    function's own into the annotation, which is where all of these end up.  So
    the shape has to be put back.

    Only the sites are wrapped, which is what makes running the ladder over the
    whole program safe: a rounding the emitter already spells is not a block, so
    no pass considers it.
    """
    blocks: dict[int, Context] = {}
    for site in todo:
        stmt = resolve_stmt(func, site.cursor.path.stmt())
        if _isolatable(stmt, site.cursor.resolve()):
            blocks[id(stmt)] = site.ctx
    return _Isolate(blocks).apply(func) if blocks else func


def _unfold_roundings(func: FuncDef) -> FuncDef:
    """*func* with every rounding the op table cannot spell expressed as
    integer arithmetic.

    The sequence of `docs/todos/native-lowering-roadmap.md`, and the order is
    its: `UnfoldSpecial` first, so the branches it states are upstream of
    everything and `FloatToFixed` emits no ladder of its own; `UnfoldOverflow`
    before `FloatToFixed`, so the latter sees an unbounded format and does the
    position axis alone.

    One pass, not a fixpoint: each step selects its own candidates, so the two
    rows of the ladder -- a non-native float context, and a fixed-point one the
    backend cannot lower -- are the same call with different steps declining.
    """
    todo = [s for s in sites(func) if s.kind is not UnfoldKind.ARITH]
    if not todo:
        return func
    func = _isolate(func, todo)
    func = UnfoldSpecial.apply(func)
    func = UnfoldOverflow.apply(func, early_check=True)
    func = FloatToFixed.apply(func)
    return RescaleFixed.apply(func)


def unfold(func: FuncDef, mode: UnfoldMode) -> FuncDef:
    """*func* with every rounding the cpp op table cannot spell replaced, as
    far as *mode* allows.

    Arithmetic first: an operation under an unsupported context becomes a
    native one plus a rounding, so the roundings the second half lowers are all
    the roundings there are.
    """
    if not isinstance(func, FuncDef):
        raise TypeError(f'Expected \'FuncDef\', got {func}')
    if mode is UnfoldMode.NONE:
        return func
    if mode is UnfoldMode.DOUBLE_ROUND:
        func = unfold_arith(func)
    return _unfold_roundings(func)
