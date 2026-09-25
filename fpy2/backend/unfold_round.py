"""
Which roundings a backend's op table cannot spell, and their rewrite.

The emitter refuses a rounding under a context its op table does not dispatch
on, and every refusal names the operator that would fix it.  This module asks
the same question early enough to act on it, so the answer is a program point
to rewrite rather than a message to print.

The question is the same for every backend; the answers -- what is native,
what the op table emits, the intermediates -- are an :class:`UnfoldTarget`.
The operators it names are backend-independent, and none of them changes.
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto

from ..analysis import ContextUse, DefineUse
from ..ast.fpyast import Cast, Expr, FuncDef, Round
from ..number import REAL, MPBFixedContext, MPFixedContext
from ..number.context.context import Context
from ..transform import (
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
from ..transform.cursor import expr_sites

__all__ = [
    'UnfoldKind',
    'UnfoldMode',
    'UnfoldSite',
    'UnfoldTarget',
    'sites',
    'unfold',
    'unfold_arith',
]


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
    """A rounding to a float context with no target type: its storage
    *contains* the format rather than equalling it, so a cast rounds to the
    storage's own format.  Recovered by lowering the rounding to fixed-point."""

    FIXED_ROUND = auto()
    """A rounding to a fixed-point context the emitter cannot lower as it
    stands."""


@dataclass(frozen=True)
class UnfoldSite:
    """One program point the emitter would refuse, and why."""

    cursor: ExprCursor
    kind: UnfoldKind
    ctx: Context
    """The active context that made it a site."""


@dataclass(frozen=True)
class UnfoldTarget:
    """A backend's answers."""

    native: Callable[[Context], bool]
    """Whether the op table dispatches on a context."""

    rounds: Callable[[Context], bool]
    """Whether the emitter spells a rounding into a context as it stands."""

    ops: frozenset[type]
    """The node types the op table keys.  A node it does not key reaches the
    emitter another way -- `Min` and `Max` select an operand rather than
    rounding, `Len` is exact -- so it has no signature to miss."""

    intermediates: tuple[Context, ...]
    """Native contexts to offer as an intermediate, narrowest first: the
    intermediate's width becomes the arithmetic's storage, and a wider one is
    never *less* admissible."""

    post: Callable[[FuncDef], FuncDef] = lambda f: f
    """Run after the roundings are lowered."""


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


def _classify(
    e: Expr, active_of: _Scopes, target: UnfoldTarget,
) -> tuple[UnfoldKind, Context] | None:
    """*e*'s kind and the context that gives it one, or `None` where the
    emitter needs no help."""
    if isinstance(e, Round | Cast):
        active = active_of(e)
        if active is None or target.rounds(active):
            return None
        if active.is_stochastic():
            # no step of the ladder draws random bits, so this is not a site --
            # the emitter's own refusal says so, and better
            return None
        if isinstance(active, MPFixedContext | MPBFixedContext):
            return UnfoldKind.FIXED_ROUND, active
        return UnfoldKind.FLOAT_ROUND, active
    if type(e) in target.ops:
        # `REAL` is the one non-native context the table reaches, by widening to
        # an op that gives the exact result and rounds to itself.
        active = active_of(e)
        if active is None or active is REAL or target.native(active):
            return None
        return UnfoldKind.ARITH, active
    return None


def sites(
    func: FuncDef, target: UnfoldTarget, within: Cursor | None = None,
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
        lambda e: _classify(e, active_of, target) is not None,
        within,
    ):
        got = _classify(cursor.resolve(), active_of, target)
        assert got is not None
        out.append(UnfoldSite(cursor, *got))
    return out


def _split_arith(
    func: FuncDef, site: UnfoldSite, target: UnfoldTarget,
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
    for cand in target.intermediates:
        try:
            return SplitRound.apply(func, cand, where=site.cursor)
        except TransformDeclined:
            continue
    return None


def _arith(func: FuncDef, target: UnfoldTarget) -> list[UnfoldSite]:
    return [s for s in sites(func, target) if s.kind is UnfoldKind.ARITH]


def _step(
    func: FuncDef, todo: list[UnfoldSite], target: UnfoldTarget,
) -> FuncDef | None:
    for site in todo:
        out = _split_arith(func, site, target)
        if out is not None:
            return out
    return None


def unfold_arith(func: FuncDef, target: UnfoldTarget) -> FuncDef:
    """*func* with every arithmetic site the op table cannot spell computed at
    a native intermediate instead.

    Operand formats are the precondition: the per-operation rules hold only for
    operands the *target* represents, so an argument carrying no context of its
    own refuses every candidate.  `Specialize` pins them in the compiler's
    pipeline; a caller reaching this directly runs `monomorphize` first.

    Sites are re-derived after each rewrite rather than forwarded: the rewrite
    lifts its operation into a new block, so the cursors below it move.
    """
    todo = _arith(func, target)
    while todo:
        out = _step(func, todo, target)
        if out is None:
            return func
        func = out
        left = _arith(func, target)
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


def _unfold_roundings(func: FuncDef, target: UnfoldTarget) -> FuncDef:
    """*func* with every rounding the op table cannot spell expressed as
    integer arithmetic.

    The ladder is *aimed*: each of its passes finds its own sites by active
    context, so run over the whole program it would lower the roundings the
    emitter already spells too -- correct, and pure waste.  The sites are this
    module's, and one pass of the ladder clears each.
    """
    todo = [s for s in sites(func, target) if s.kind is not UnfoldKind.ARITH]
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


def unfold(func: FuncDef, mode: UnfoldMode, target: UnfoldTarget) -> FuncDef:
    """*func* with every rounding *target*'s op table cannot spell replaced, as
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
        func = unfold_arith(func, target)
    return target.post(_unfold_roundings(func, target))
