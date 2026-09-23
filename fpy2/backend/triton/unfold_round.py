"""
triton backend: which roundings the op table cannot spell.

A copy of the cpp backend's module with this backend's answers: what is native,
what the op table emits, the intermediates, and which fixed-point roundings the
emitter lowers.
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from fractions import Fraction

from ...analysis import ContextUse, DefineUse, FormatAnalysis, FormatInfer
from ...analysis.format_infer import to_abstract
from ...ast import DefaultTransformVisitor
from ...ast.fpyast import (
    BinaryOp,
    Cast,
    ContextStmt,
    Expr,
    ForeignVal,
    FuncDef,
    Round,
    TernaryOp,
    UnaryOp,
)
from ...number import REAL, MPBFixedContext, MPFixedContext, OverflowMode
from ...number.context.context import Context
from ...transform import (
    Cursor,
    EditLog,
    ExprCursor,
    FloatToFixed,
    RescaleFixed,
    SplitRound,
    StmtCursor,
    TransformDeclined,
    TransformReferenceError,
    UnfoldOverflow,
    UnfoldSpecial,
)
from ...transform.cursor import expr_sites
from .emitter import _integral_round
from .target import _DIV_SHAPES, _fp_ctxs, is_native_ctx, make_op_table

__all__ = [
    'UnfoldKind',
    'UnfoldMode',
    'UnfoldSite',
    'sites',
    'unfold',
    'unfold_arith',
]

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
    """A rounding to a float context with no Triton type: its storage
    *contains* the format rather than equalling it, so a cast rounds to the
    storage's own format.  Recovered by lowering the rounding to fixed-point."""

    FIXED_ROUND = auto()
    """A rounding to a fixed-point context the emitter cannot lower as it
    stands -- its digits are away from position zero, it is bounded, or its
    rounding mode has no integral-round spelling."""


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
        scope = self.ctx_use.find_scope_from_use(e)
        return scope.ctx if isinstance(scope.ctx, Context) else None


def _dispatches(e: Expr) -> bool:
    """Whether the op table is what emits *e*.

    Its keys are the definition: a node it does not key reaches the emitter
    another way -- `Min` and `Max` select an operand rather than rounding, `Len`
    is exact -- so it has no signature to miss.
    """
    table = make_op_table()
    match e:
        case UnaryOp():
            return type(e) in table.unary
        case BinaryOp():
            return type(e) in table.binary
        case TernaryOp():
            return type(e) in table.ternary
        case _:
            return False


def _intermediates() -> list[Context]:
    """Native contexts to offer as an intermediate, narrowest first.

    FP32 and FP64 under RNE: unlike FP16, every operation in the table has a
    signature at both.
    """
    return _fp_ctxs(_DIV_SHAPES)


def _classify(
    e: Expr, active_of: _Scopes,
) -> tuple[UnfoldKind, Context] | None:
    """*e*'s kind and the context that gives it one, or `None` where the
    emitter needs no help.

    A narrower reading of native makes a site here rather than a refusal there.
    """
    if isinstance(e, Round | Cast):
        active = active_of(e)
        if active is None or is_native_ctx(active):
            return None
        if active.is_stochastic():
            # no step of the ladder draws random bits, so this is not a site --
            # the emitter's own refusal says so, and better
            return None
        if isinstance(active, _FIXED):
            if _integral_round(active) is not None:
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


def sites(
    func: FuncDef, within: Cursor | None = None,
) -> list[UnfoldSite]:
    """The program points of *func* the emitter would refuse, in visit order.

    *func* is a specialized :class:`FuncDef`, before the analyses the emitter
    runs on.  `within` keeps the sites at or beneath the point it names.
    """
    if not isinstance(func, FuncDef):
        raise TypeError(f'Expected \'FuncDef\', got {func}')
    active_of = _Scopes(func)
    out: list[UnfoldSite] = []
    for cursor in expr_sites(
        func,
        lambda e: _classify(e, active_of) is not None,
        within,
    ):
        got = _classify(cursor.resolve(), active_of)
        assert got is not None
        out.append(UnfoldSite(cursor, *got))
    return out


def _split_arith(
    func: FuncDef, site: UnfoldSite,
) -> FuncDef | None:
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


def _step(
    func: FuncDef, todo: list[UnfoldSite],
) -> FuncDef | None:
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


_LADDER: tuple[Callable[[FuncDef, Cursor], EditLog], ...] = (
    lambda f, w: UnfoldSpecial.apply_with_edits(f, where=w),
    lambda f, w: UnfoldOverflow.apply_with_edits(f, where=w, early_check=True),
    lambda f, w: FloatToFixed.apply_with_edits(f, where=w),
    lambda f, w: RescaleFixed.apply_with_edits(f, where=w),
)
"""The sequence of `docs/todos/native-lowering-roadmap.md`.

`UnfoldSpecial` first, so the branches it states are upstream of everything and
`FloatToFixed` emits no ladder of its own; `UnfoldOverflow` before
`FloatToFixed`, so the latter sees an unbounded format and does the position
axis alone.
"""


def _unfold_roundings(func: FuncDef) -> FuncDef:
    """*func* with every rounding the op table cannot spell expressed as
    integer arithmetic.

    The ladder is *aimed*: each of its passes finds its own sites by active
    context, so run over the whole program it would lower the roundings the
    emitter already spells too -- correct, and pure waste.  The sites are this
    module's, and one pass of the ladder clears each.
    """
    todo = [s for s in sites(func) if s.kind is not UnfoldKind.ARITH]
    if not todo:
        return func
    # the anchor is the *statement* holding the rounding: a step consumes the
    # rounding it acts on, so the expression `sites` reported names nothing
    # afterwards, while the statement survives with what replaced it beneath
    anchors: list[Cursor] = [StmtCursor(func, s.cursor.path.stmt()) for s in todo]
    for i in range(len(anchors)):
        for step in _LADDER:
            # a step that does not apply is an ordinary outcome: the two rows
            # of the ladder are the same call with different steps declining
            try:
                log = step(func, anchors[i])
            except (TransformDeclined, TransformReferenceError):
                continue
            func = log.result
            anchors = [log.forward(a) for a in anchors]
    return func


def _within(e: Expr, bound: Fraction, nmin: int, info: FormatAnalysis) -> bool:
    """Is every finite value of *e* at most *bound* in magnitude, once rounded
    to digit position ``nmin + 1``?"""
    af = to_abstract(info.by_expr[e])
    if af is not None and max(abs(af.pos_bound), abs(af.neg_bound)) <= bound:
        return True
    # the store says ``|e| < 2 ** (msb + 1)``, and rounding stays below the
    # next power of two when that power is on the grid
    db = info.digit_bound
    terms = db.by_expr.get(e) if db is not None else None
    if db is None or terms is None or terms.msb is None:
        return False
    msb = db.store.maximum(terms.msb)
    return isinstance(msb, int) and nmin < msb + 1 and 2 ** (msb + 1) <= bound


class _Unbound(DefaultTransformVisitor):
    def __init__(self, drop: dict[ContextStmt, MPFixedContext]):
        self.drop = drop

    def _visit_context(self, stmt: ContextStmt, ctx):
        s, ctx = super()._visit_context(stmt, ctx)
        if stmt in self.drop:
            s = ContextStmt(s.target, ForeignVal(self.drop[stmt], stmt.loc), s.body, s.loc)
        return s, ctx


def _drop_proven_bounds(func: FuncDef) -> FuncDef:
    """*func* with each asserted fixed-point bound its roundings provably meet
    removed.

    A kernel cannot raise, so the emitter refuses a bounded context; where the
    assertion cannot fire, the unbounded context is the same one.  Infinities
    and NaN round the same way under both.
    """
    ctx_use = ContextUse.analyze(func, def_use=DefineUse.analyze(func))
    todo = [
        s for s in ctx_use.scopes
        if isinstance(s.site, ContextStmt)
        and isinstance(s.ctx, MPBFixedContext)
        and s.ctx.overflow is OverflowMode.ASSERT
    ]
    if not todo:
        return func
    info = FormatInfer.analyze(func, use_digit_bounds=True)
    drop: dict[ContextStmt, MPFixedContext] = {}
    for scope in todo:
        c = scope.ctx
        assert isinstance(c, MPBFixedContext) and isinstance(scope.site, ContextStmt)
        bound = min(c.pos_maxval.as_rational(), -c.neg_maxval.as_rational())
        if all(
            isinstance(u, Round | Cast) and _within(u.arg, bound, c.nmin, info)
            for u in ctx_use.uses[scope]
        ):
            drop[scope.site] = MPFixedContext(
                c.nmin, c.rm, c.num_randbits, rng=c.rng,
                enable_nan=c.enable_nan, enable_inf=c.enable_inf,
                enable_neg_zero=c.enable_neg_zero,
                nan_value=c.nan_value, inf_value=c.inf_value,
            )
    return _Unbound(drop)._visit_function(func, None) if drop else func


def unfold(func: FuncDef, mode: UnfoldMode) -> FuncDef:
    """*func* with every rounding the op table cannot spell replaced, as
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
    return _drop_proven_bounds(_unfold_roundings(func))
