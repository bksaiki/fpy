"""
State a rounding context's overflow as program text.

A bounded format decides two things at once: where its grid lies, and what
becomes of a value too large for it.  IEEE 754 defines the second in terms of
the first — round with an unbounded exponent range, then see whether the result
fits.  For a bounded context ``C`` with unbounded counterpart ``U``, and a
finite ``x``,

.. math::

   \\mathrm{round}_C(x) =
     \\begin{cases}
       \\mathit{overflow}(\\mathrm{sign}\\ x)
         & |\\mathrm{round}_U(x)| > \\mathit{maxval} \\\\
       \\mathrm{round}_U(x) & \\text{otherwise}
     \\end{cases}

This transform makes that structural: the block rounds under ``U``, and the
bound becomes a comparison.

.. code-block:: python

   # Before
   with fp.FP16:
       y = fp.round(x)

   # After
   with fp.REAL:
       with fp.MPSFloatContext(11, -14):
           t = fp.round(x)
       if t > 65504:
           y = fp.inf()
       elif t < -65504:
           y = -fp.inf()
       else:
           y = t

With `pre_check`, a guard on the *operand* comes first, so nothing past the
bound reaches the rounding at all:

.. code-block:: python

   if x >= 65536:
       y = fp.inf()
   elif x <= -65536:
       y = -fp.inf()
   else:
       ...

Its threshold is :meth:`fpy2.Context.infval`, the next value above ``maxval``
in the unbounded format — *not* ``maxval`` itself.  Under ``RNE``, ``FP16``
maps ``65510`` to ``65504``, so exceeding the bound does not imply overflow;
reaching ``infval`` does, for any monotone rounding, since
``round_U(x) >= round_U(infval) = infval > maxval``.  The guard is sound but
not complete — ``FP16``'s tie at ``65520`` lies below ``infval`` yet rounds up
to ``65536`` — so the comparison after the rounding stays either way.

What overflow produces, and what the format makes of NaN and the infinities,
are asked of the source context rather than assumed.  A special value gets a
branch only where the emitted program would otherwise disagree with it, so an
IEEE source gets none and a format that substitutes its bound for NaN gets one.

Only a block whose body is entirely ``x = fp.round(v)`` (or a returned round)
over variables is rewritten.  ``Cast`` is excluded: it asserts exactness, which
this rewrite does not preserve.
"""

from dataclasses import dataclass, replace

from ..analysis import PartialEval, PartialEvalInfo
from ..ast.fpyast import (
    Assign,
    Call,
    Compare,
    ContextStmt,
    Expr,
    ForeignVal,
    FuncDef,
    IfStmt,
    Integer,
    IsInf,
    IsNan,
    Location,
    NamedId,
    ReturnStmt,
    Round,
    Stmt,
    StmtBlock,
    UnderscoreId,
    Var,
)
from ..env import fpy_alias
from ..number import (
    REAL,
    EFloatContext,
    Float,
    MPBFloatContext,
    MPSFloatContext,
    RealFloat,
    RoundingMode,
)
from ..utils import CompareOp, Gensym
from .utils import (
    BlockRewriter,
    attribute,
    check_where,
    number_literal,
    sign_choice,
    value_literal,
)

_NAN = Float(isnan=True)
_POS_INF = Float(isinf=True)
_NEG_INF = Float(isinf=True, s=True)

_BoundedCtx = EFloatContext | MPBFloatContext
"""the bounded float contexts, whose bound this rewrite states as program text"""


@dataclass(frozen=True)
class _Source:
    """A bounded format, in the terms the rewrite needs."""

    unbounded: MPSFloatContext
    """the same format with no bound, which the block rounds under instead"""
    maxval: RealFloat
    """largest representable value; above it a rounding has overflowed"""
    neg_maxval: RealFloat
    """its negative counterpart, which a format need not mirror"""
    infval: RealFloat
    """smallest value that certainly overflows, for `pre_check`"""
    neg_infval: RealFloat
    """its negative counterpart"""
    over_pos: Float
    """what an overflow above the bound produces"""
    over_neg: Float
    """what an overflow below it produces"""
    neg_zero: bool
    """whether the format keeps a negative zero"""
    specials: tuple[tuple[Float, Float] | None, ...]
    """
    what the format makes of NaN and of the infinities, as a `(positive,
    negative)` pair each; `None` where the rewrite reproduces it already
    """


def _cmp(x: Float, b: RealFloat) -> int | None:
    """The sign of ``x - b``, or `None` if `x` is a NaN, which compares to nothing."""
    if x.isnan:
        return None
    if x.isinf:
        return -1 if x.s else 1
    xr = x.as_rational()
    br = b.as_rational()
    return (xr > br) - (xr < br)


def _scaled(x: RealFloat, k: int) -> RealFloat:
    """``x * 2**k``, exactly."""
    return RealFloat(s=x.s, exp=x.exp + k, c=x.c)


class _Prober:
    """
    What a bounded context does at its edges, and whether a rounding under its
    unbounded counterpart plus a comparison reproduces it.
    """

    ctx: _BoundedCtx
    unbounded: MPSFloatContext
    pre_check: bool

    def __init__(self, ctx: _BoundedCtx, unbounded: MPSFloatContext, pre_check: bool):
        self.ctx = ctx
        self.unbounded = unbounded
        self.pre_check = pre_check

    def describe(self) -> _Source | None:
        """`ctx` as a lowerable bounded format, or `None`."""
        ctx = self.ctx
        maxval = ctx.maxval().as_real()
        neg_maxval = ctx.maxval(s=True).as_real()
        infval = ctx.infval().as_real()
        neg_infval = ctx.infval(s=True).as_real()
        if maxval.is_zero() or not neg_maxval.s:
            # a format representing no non-zero value has no overflow to state
            return None

        over = self._overflow(maxval, neg_maxval)
        if over is None:
            return None
        over_pos, over_neg = over

        # a value rounding to zero keeps its sign only if the format has one
        # to keep; the unbounded counterpart always does
        neg_zero = ctx.round(Float(c=0, s=True)).s

        src = _Source(
            self.unbounded, maxval, neg_maxval, infval, neg_infval,
            over_pos, over_neg, neg_zero, (),
        )
        specials = self._specials(src)
        if specials is None:
            return None
        return _Source(
            self.unbounded, maxval, neg_maxval, infval, neg_infval,
            over_pos, over_neg, neg_zero, specials,
        )

    def _overflow(
        self, maxval: RealFloat, neg_maxval: RealFloat
    ) -> tuple[Float, Float] | None:
        """
        What `ctx` returns for a value past its bound.

        The rewrite writes that value out as a constant, so it has to be the
        same for every overflowing operand of a given sign.  Two magnitudes
        far apart stand in for the check; a format whose answer varies between
        them is declined rather than silently mis-lowered.
        """
        try:
            near = [self.ctx.round(_scaled(b, 1)) for b in (maxval, neg_maxval)]
            far = [self.ctx.round(_scaled(b, 64)) for b in (maxval, neg_maxval)]
        except (ValueError, OverflowError):
            # a format that refuses to round an overflow at all
            return None

        if not all(_same(a, b) for a, b in zip(near, far)):
            return None
        return near[0], near[1]

    def _specials(
        self, src: _Source
    ) -> tuple[tuple[Float, Float] | None, ...] | None:
        """
        What the rewrite has to say about NaN and the infinities: the pair the
        format makes of each, or `None` where the emitted program agrees
        already and needs no branch.
        """
        out: list[tuple[Float, Float] | None] = []
        for pos, neg in ((_NAN, Float(x=_NAN, s=True)), (_POS_INF, _NEG_INF)):
            try:
                want = (self.ctx.round(pos), self.ctx.round(neg))
            except ValueError:
                # a format that cannot represent it at all
                return None
            got = (self._emitted(pos, src), self._emitted(neg, src))
            agrees = all(_same(a, b) for a, b in zip(want, got))
            out.append(None if agrees else want)
        return tuple(out)

    def _emitted(self, x: Float, src: _Source) -> Float:
        """What the generated code yields for `x`, special branches aside."""
        if self.pre_check:
            if (c := _cmp(x, src.infval)) is not None and c >= 0:
                return src.over_pos
            if (c := _cmp(x, src.neg_infval)) is not None and c <= 0:
                return src.over_neg

        t = self.unbounded.round(x)
        if (c := _cmp(t, src.maxval)) is not None and c > 0:
            return src.over_pos
        if (c := _cmp(t, src.neg_maxval)) is not None and c < 0:
            return src.over_neg
        if not src.neg_zero and not t.is_nar() and t.is_zero():
            return Float(c=0)
        return t


def _same(a: Float, b: Float) -> bool:
    """Whether two values are the same, sign and all."""
    if a.is_nar():
        return a.isnan == b.isnan and a.isinf == b.isinf and (a.isnan or a.s == b.s)
    return not b.is_nar() and a.as_real() == b.as_real() and a.s == b.s


def _unbounded(ctx: _BoundedCtx) -> MPSFloatContext:
    """
    `ctx` with its bound removed: the same grid, with nothing to overflow.

    Built through the constructor rather than ``from_format``, which rejects a
    format without NaN or infinity (see the ``TODO`` in
    :mod:`fpy2.number.context.mps_float`).  Handing those back is harmless:
    they reach a rounding only as its operand, and the caller branches on
    whichever of them the source treats differently.  A shifted exponent
    encoding needs no attention either — ``pmax`` and ``emin`` account for it.
    """
    return MPSFloatContext(ctx.pmax, ctx.emin, ctx.rm)


def _unbounded_expr(
    ctx: MPSFloatContext, alias: str | None, loc: Location | None
) -> Expr:
    """
    `ctx` as an expression, written as a constructor call where the program
    has a name for `fpy2` to write it with.
    """
    if alias is None:
        return ForeignVal(ctx, loc)
    kwargs: list[tuple[str, Expr]] = []
    if ctx.rm is not RoundingMode.RNE:
        kwargs.append(('rm', attribute(alias, 'RoundingMode', ctx.rm.name, loc=loc)))
    return Call(
        attribute(alias, 'MPSFloatContext', loc=loc),
        MPSFloatContext,
        (Integer(ctx.pmax, loc), Integer(ctx.emin, loc)),
        tuple(kwargs),
        loc,
    )


class _ExternalizeOverflowInstance(BlockRewriter):
    """Rewrites every qualifying context statement in a function."""

    func: FuncDef
    eval_info: PartialEvalInfo
    gensym: Gensym
    where: int | None
    pre_check: bool
    alias: str | None
    used_alias: bool
    site_idx: int

    def __init__(
        self, func: FuncDef, eval_info: PartialEvalInfo,
        where: int | None = None, pre_check: bool = False,
    ):
        self.func = func
        self.eval_info = eval_info
        self.gensym = Gensym(eval_info.def_use.names())
        self.where = where
        self.pre_check = pre_check
        # the name the program calls `fpy2` by, which the emitted context is
        # written with; without one it falls back to the context value itself
        self.alias = fpy_alias(func.env)
        self.used_alias = False
        # Counts *candidate* blocks (those the rewrite could externalize) in
        # visit order, outermost-first.  `where` selects one by this index.
        self.site_idx = 0

    def apply(self) -> FuncDef:
        func = self._visit_function(self.func, None)
        if self.used_alias:
            # the emitted context names `fpy2`, which the environment binds
            # but the body may not have referred to before
            assert self.alias is not None
            meta = replace(
                func.meta, free_vars=func.free_vars | {NamedId(self.alias)},
            )
            func = FuncDef(func.name, func.args, func.body, meta, loc=func.loc)
        return func

    def _candidate(self, stmt: ContextStmt) -> _Source | None:
        """The block's format, if its bound can be taken out of its context."""
        # a bound context is visible to the body as a value, which the rewrite changes
        if not isinstance(stmt.target, UnderscoreId):
            return None

        ctx = self.eval_info.by_expr.get(stmt.ctx)
        if not isinstance(ctx, _BoundedCtx):
            return None
        # stochastic rounding would have to draw its bits under the same format
        if ctx.num_randbits != 0:
            return None

        # only a rounding is a bounded rounding in disguise; `Cast` asserts
        # exactness, which the rewrite would not preserve
        for s in stmt.body.stmts:
            match s:
                case Assign(target=NamedId()) | ReturnStmt():
                    if not isinstance(s.expr, Round) or not isinstance(s.expr.arg, Var):
                        return None
                case _:
                    return None

        return _Prober(ctx, _unbounded(ctx), self.pre_check).describe()

    def _externalize(
        self, e: Round, target: NamedId, loc: Location | None, src: _Source
    ) -> Stmt:
        """`target = round(v)` as an unbounded rounding plus a bound check."""
        assert isinstance(e.arg, Var)
        name = e.arg.name

        def arg() -> Var:
            return Var(name, loc)

        def assign(v: Expr) -> StmtBlock:
            return StmtBlock([Assign(target, None, v, loc)])

        def past(
            operand: Expr, op: CompareOp, bound: RealFloat, over: Float,
            rest: StmtBlock,
        ) -> StmtBlock:
            """`rest`, behind a branch taking `operand` past `bound`."""
            return StmtBlock([IfStmt(
                Compare([op], [operand, number_literal(bound, loc)], loc),
                assign(value_literal(over, loc)), rest, loc,
            )])

        # the rounding, under the format the bound came out of: with nothing
        # left to overflow, it is the grid and no more
        t = self.gensym.fresh('t')
        self.used_alias |= self.alias is not None
        rounding = ContextStmt(
            UnderscoreId(), _unbounded_expr(src.unbounded, self.alias, loc),
            StmtBlock([Assign(t, None, Round(None, arg(), loc), loc)]), loc,
        )

        # a result past the bound overflowed, whichever side it left by
        rest: StmtBlock = assign(Var(t, loc))
        if not src.neg_zero:
            # the format spends that encoding elsewhere, so a value rounding
            # to zero comes back positive
            rest = StmtBlock([IfStmt(
                Compare([CompareOp.EQ], [Var(t, loc), Integer(0, loc)], loc),
                assign(Integer(0, loc)), rest, loc,
            )])
        rest = past(Var(t, loc), CompareOp.LT, src.neg_maxval, src.over_neg, rest)
        rest = past(Var(t, loc), CompareOp.GT, src.maxval, src.over_pos, rest)

        body = StmtBlock([rounding, *rest.stmts])

        # an operand already past the bound needs no rounding to know it
        if self.pre_check:
            body = past(arg(), CompareOp.LE, src.neg_infval, src.over_neg, body)
            body = past(arg(), CompareOp.GE, src.infval, src.over_pos, body)

        # a special value the rounding and the checks would not reproduce
        for test, want in zip((IsNan, IsInf), src.specials):
            if want is None:
                continue
            body = StmtBlock([IfStmt(
                test(None, arg(), loc),
                assign(sign_choice(want[0], want[1], arg(), loc)),
                body, loc,
            )])

        # the checks compare against constants, so they are exact whatever
        # context encloses this statement; the rounding sets its own
        return ContextStmt(UnderscoreId(), ForeignVal(REAL, loc), body, loc)

    def _rewrite(self, stmt: ContextStmt, src: _Source) -> list[Stmt]:
        """The block's rounds, with the bound taken out of the context.
        Nothing rounds under the bounded context afterwards, so the block
        itself goes away."""
        stmts: list[Stmt] = []
        for s in stmt.body.stmts:
            if isinstance(s, Assign):
                assert isinstance(s.expr, Round) and isinstance(s.target, NamedId)
                stmts.append(self._externalize(s.expr, s.target, s.loc, src))
            else:
                # a returned round lands in a temporary, which the return names
                assert isinstance(s, ReturnStmt) and isinstance(s.expr, Round)
                out = self.gensym.fresh('t')
                stmts.append(self._externalize(s.expr, out, s.loc, src))
                stmts.append(ReturnStmt(Var(out, s.loc), s.loc))
        return stmts


class ExternalizeOverflow:
    """
    Transformation pass to state a context's overflow as program text.
    """

    @staticmethod
    def apply(
        func: FuncDef, *,
        where: int | None = None,
        pre_check: bool = False,
        eval_info: PartialEvalInfo | None = None,
    ) -> FuncDef:
        """
        Takes the bound out of every qualifying rounding context in `func`.

        `where` selects a single candidate block by index, in visit order
        (outermost-first); candidates are the blocks this pass could rewrite.
        If `None`, every candidate is rewritten.

        With `pre_check`, a guard on the operand precedes the rounding, so
        nothing certain to overflow is rounded at all.
        """
        if not isinstance(func, FuncDef):
            raise TypeError(f'Expected \'FuncDef\', got {func}')
        check_where(where)

        if eval_info is None:
            eval_info = PartialEval.apply(func)

        return _ExternalizeOverflowInstance(func, eval_info, where, pre_check).apply()
