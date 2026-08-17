"""
Express floating-point rounding as fixed-point rounding.

A float format rounds at a digit position that depends on the value: its
representable values thin out as the magnitude grows.  For a format with precision ``P``, subnormal position
``EXP``, largest exponent ``EMAX``, and bound ``B``,

.. math::

   \\mathrm{round}_F(x) = \\mathrm{round}_{A(\\infty, n, B)}(x),
   \\quad n = \\mathrm{clamp}(\\mathrm{logb}(x) - P + 1,\\; EXP,\\; EMAX - P + 1)

That is, float rounding *is* fixed-point rounding, once the position is known.
The value itself is never scaled, and the bound stays the format's own ``B``.

This transform makes that structural: the position is computed under
``fp.REAL``, and the rounding happens under a fixed-point context built at
that position.

.. code-block:: python

   # Before
   with fp.FP16:
       y = fp.round(x)

   # After
   with fp.REAL:
       if fp.isnan(x):
           y = fp.nan()
       elif fp.isinf(x):
           y = (-fp.inf() if fp.signbit(x) else fp.inf())
       elif x == 0:
           y = (-0.0 if fp.signbit(x) else 0)
       else:
           e = fp.logb(x)
           if e < -14:
               with fp.MPBFixedContext(-25, 65504, overflow=fp.OverflowMode.OVERFLOW, enable_inf=True):
                   y = fp.round(x)
           else:
               exp = min((e - 10), 5)
               with fp.MPBFixedContext((exp - 1), 65504, overflow=fp.OverflowMode.OVERFLOW, enable_inf=True):
                   y = fp.round(x)

Below ``emin`` the format is fixed-point already: every value there rounds at
``EXP``, so that branch's context is a constant and nothing in it depends on
the exponent.  The normal branch needs no lower clamp because of it.

The upper clamp keeps the context constructible and the format shiftable:
``B`` is representable at every position up to ``EMAX - P + 1`` and stops being
so immediately above.  Anything above that exceeds ``B`` and overflows, which
rounding at the clamped position against ``B`` produces.

``logb`` is undefined on NaN, an infinity, and a zero, so each takes a branch
of its own.  All three are constants, so the branches assign what the format
makes of them.  Only the roundings sit under a rounding context; the rest is
exact under ``REAL``, whatever context encloses the statement.

Run :func:`fpy2.strategies.rescale_fixed` afterwards to shift the resulting
fixed-point rounding to digit position zero, where its values are integers.

Applies to a float format that rounds deterministically and whose overflow a
fixed-point round can reproduce — an infinity, the bound, or a NaN.  An
unbounded format (``MPSFloatContext``, ``MPFloatContext``) needs no upper
clamp -- its target's bound states how far the operand reaches, not what the
format does at an edge; one without subnormals needs no branch for them.  Only a block whose body is entirely ``x = fp.round(v)`` (or a returned
round) over variables is rewritten.  The rewrite needs ``fpy2`` in scope,
since it names the context constructor.
"""

from dataclasses import dataclass, replace
from enum import Enum, auto

from ..analysis import PartialEval, PartialEvalInfo
from ..ast.fpyast import (
    Add,
    Assign,
    BoolVal,
    Call,
    Compare,
    ConstNan,
    ContextStmt,
    Copysign,
    Expr,
    ForeignVal,
    FuncDef,
    IfExpr,
    IfStmt,
    Integer,
    IsInf,
    IsNan,
    Location,
    Logb,
    Min,
    NamedId,
    Pow,
    ReturnStmt,
    Round,
    Stmt,
    StmtBlock,
    Sub,
    UnderscoreId,
    Var,
)
from ..env import fpy_alias
from ..number import (
    REAL,
    Context,
    EFloatContext,
    Float,
    IEEEContext,
    MPBFixedContext,
    MPBFloatContext,
    MPFloatContext,
    MPSFloatContext,
    OverflowMode,
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


class _Policy(Enum):
    """What a float format does with a value too large to represent."""

    UNBOUNDED = auto()
    """the format has no bound, so nothing can overflow it"""
    INFINITE = auto()
    """overflow becomes an infinity, as IEEE 754 does"""
    SATURATING = auto()
    """overflow becomes the bound; the format has neither NaN nor infinity"""
    NAN_ON_OVERFLOW = auto()
    """overflow becomes a NaN, which the format encodes at its largest value"""


@dataclass(frozen=True)
class _Source:
    """A float format, in the terms the lowering needs."""

    pmax: int
    """precision"""
    emin: int | None
    """
    smallest normal exponent: below it the format is fixed-point.

    `None` for a format without subnormals, which then needs no branch for
    them: every value rounds at a position taken from its own exponent.
    """
    expmin: int | None
    """position of the format's finest digit, if it has one"""
    expmax: int | None
    """
    position of the bound's last digit, above which the bound is unrepresentable.

    `None` for an unbounded format, whose position needs no upper clamp.
    """
    maxval: RealFloat | None
    """the bound, if the format has one"""
    rm: RoundingMode
    """rounding mode"""
    policy: _Policy
    """what overflow produces"""
    specials: tuple[Float, Float, Float, Float, Float]
    """what NaN, `+Inf`, `-Inf`, `+0`, and `-0` round to"""
    neg_zero: bool
    """whether the format keeps a negative zero"""


def _overflow_policy(
    ctx: Context, maxval: RealFloat, neg_maxval: RealFloat
) -> _Policy | None:
    """
    What `ctx` returns for a value past its bound, or `None` if a fixed-point
    round cannot reproduce it.
    """
    def past(bound: RealFloat) -> RealFloat:
        return RealFloat(s=bound.s, exp=bound.exp + 1, c=bound.c)  # twice it

    try:
        pos = ctx.round(past(maxval))
        neg = ctx.round(past(neg_maxval))
    except (ValueError, OverflowError):
        # a format that refuses to round one at all
        return None

    if pos.isinf and neg.isinf and not pos.s and neg.s:
        return _Policy.INFINITE
    if pos.isnan and neg.isnan:
        return _Policy.NAN_ON_OVERFLOW
    if not pos.is_nar() and not neg.is_nar():
        # saturation keeps each bound and its sign
        if pos.as_real() == maxval and neg.as_real() == neg_maxval:
            return _Policy.SATURATING
    return None


def _describe(ctx: Context) -> _Source | None:
    """
    `ctx` as a lowerable float format, or `None`.

    A format qualifies when the fixed-point round can reproduce it exactly:
    it must state a precision, its rounding must be deterministic, and its
    overflow must land somewhere a fixed-point context can also land.
    """
    match ctx:
        # `IEEEContext` derives from `EFloatContext`, so it matches first
        case IEEEContext():
            pass
        case EFloatContext():
            # a shifted exponent encoding is not accounted for below
            if ctx.eoffset != 0:
                return None
        case MPBFloatContext() | MPSFloatContext() | MPFloatContext():
            pass
        case _:
            return None

    # stochastic rounding would have to draw its bits at the same position
    if ctx.num_randbits != 0:
        return None

    # an unbounded format cannot overflow; a bounded one has to say where to
    maxval: RealFloat | None = None
    neg_maxval: RealFloat | None = None
    expmax: int | None = None
    policy = _Policy.UNBOUNDED
    if isinstance(ctx, (EFloatContext, MPBFloatContext)):
        maxval = ctx.maxval().as_real()
        neg_maxval = ctx.maxval(s=True).as_real()
        # an emitted context states one bound and mirrors it, and FPy's context
        # construction has no way to pass the other, so the two must agree
        if neg_maxval != RealFloat(s=True, x=maxval):
            return None
        # the format's finest position in its top binade, which is where the clamp
        # puts a value too large for one of its own; the bound has to be
        # representable there
        expmax = ctx.emax - ctx.pmax + 1
        if maxval.exp < expmax:
            return None
        found = _overflow_policy(ctx, maxval, neg_maxval)
        if found is None:
            return None
        policy = found

    # constants, so what the format makes of them is known here
    try:
        specials = (
            ctx.round(Float(isnan=True)),
            ctx.round(Float(isinf=True)),
            ctx.round(Float(isinf=True, s=True)),
            ctx.round(Float(c=0)),
            ctx.round(Float(c=0, s=True)),
        )
    except ValueError:
        # a format that cannot represent one of them at all
        return None

    # a format without subnormals states no `emin`: every value then rounds
    # at a position read off its own exponent
    return _Source(
        ctx.pmax, getattr(ctx, 'emin', None), getattr(ctx, 'expmin', None),
        expmax, maxval, ctx.rm, policy, specials, specials[4].s,
    )


def _ctx_call(
    nmin: Expr, src: _Source, alias: str, loc: Location | None, reach: Expr,
) -> Call:
    """
    The fixed-point format holding `src`'s values, with its digits at `nmin + 1`.

    The bound and the rounding mode are `src`'s own, and the remaining flags
    put overflow where the float format puts it.  Arguments matching the
    constructor's defaults are left out.

    `reach` is how far the operand can reach in this branch, which the branch
    itself fixes and no downstream analysis can recover.  Stating it gives the
    rescaled rounding a width.
    """
    def mode(name: str, value) -> tuple[str, Expr]:
        return (name, attribute(alias, type(value).__name__, value.name, loc=loc))

    kwargs: list[tuple[str, Expr]] = []
    if src.rm is not RoundingMode.RNE:
        kwargs.append(mode('rm', src.rm))
    if not src.neg_zero:
        # the format spends that encoding elsewhere, so a value rounding to
        # zero comes back positive
        kwargs.append(('enable_neg_zero', BoolVal(False, loc)))

    # a NaN reaches a rounding only as its operand, and the branches above
    # take that case, so the format needs NaN only where an *overflow*
    # produces one
    match src.policy:
        case _Policy.UNBOUNDED:
            # `reach` is a claim that overflow cannot happen, not an edge rule
            kwargs.append(mode('overflow', OverflowMode.ASSERT))
            return Call(
                attribute(alias, 'MPBFixedContext', loc=loc),
                MPBFixedContext, (nmin, reach), tuple(kwargs), loc,
            )
        case _Policy.INFINITE:
            kwargs.append(mode('overflow', OverflowMode.OVERFLOW))
            kwargs.append(('enable_inf', BoolVal(True, loc)))
        case _Policy.SATURATING:
            # no NaN, no infinity: overflow has nowhere to go but the bound
            kwargs.append(mode('overflow', OverflowMode.SATURATE))
        case _Policy.NAN_ON_OVERFLOW:
            # an overflow would be infinite, which this format substitutes
            kwargs.append(mode('overflow', OverflowMode.OVERFLOW))
            kwargs.append(('enable_nan', BoolVal(True, loc)))
            kwargs.append(('inf_value', ConstNan(None, loc)))

    assert src.maxval is not None
    return Call(
        attribute(alias, 'MPBFixedContext', loc=loc),
        MPBFixedContext,
        (nmin, number_literal(src.maxval, loc)),
        tuple(kwargs),
        loc,
    )


class _FloatToFixedInstance(BlockRewriter):
    """Rewrites every qualifying context statement in a function."""

    func: FuncDef
    eval_info: PartialEvalInfo
    gensym: Gensym
    alias: str | None
    where: int | None
    site_idx: int

    def __init__(
        self, func: FuncDef, eval_info: PartialEvalInfo, where: int | None = None
    ):
        self.func = func
        self.eval_info = eval_info
        self.gensym = Gensym(eval_info.def_use.names())
        self.where = where
        # Counts *candidate* blocks (those the rewrite could lower) in visit
        # order, outermost-first.  `where` selects one by this index.
        self.site_idx = 0
        # the name the program calls `fpy2` by: the rewrite constructs a
        # context per value, so it has to name the constructor
        self.alias = fpy_alias(func.env)

    def apply(self) -> FuncDef:
        func = self._visit_function(self.func, None)
        if self.alias is not None:
            # the emitted contexts name `fpy2`, which the environment binds
            # but the body may not have referred to before
            meta = replace(
                func.meta,
                free_vars=func.free_vars | {NamedId(self.alias)},
            )
            func = FuncDef(func.name, func.args, func.body, meta, loc=func.loc)
        return func

    def _candidate(self, stmt: ContextStmt) -> _Source | None:
        """The block's context, if its rounding can be lowered to fixed-point."""
        # the position varies per value, so the rewrite has to name a context
        # constructor; without `fpy2` in scope there is nothing to name it by
        if self.alias is None:
            return None
        # a bound context is visible to the body as a value, which the rewrite changes
        if not isinstance(stmt.target, UnderscoreId):
            return None

        ctx = self.eval_info.by_expr.get(stmt.ctx)
        if not isinstance(ctx, Context):
            return None
        src = _describe(ctx)
        if src is None:
            return None

        # only a rounding is a fixed-point rounding in disguise; `Cast` asserts
        # exactness, which the lowering would not preserve
        for s in stmt.body.stmts:
            match s:
                case Assign(target=NamedId()) | ReturnStmt():
                    if not isinstance(s.expr, Round) or not isinstance(s.expr.arg, Var):
                        return None
                case _:
                    return None

        return src

    def _lower_round(
        self, e: Round, target: NamedId, loc: Location | None, src: _Source
    ) -> Stmt:
        """`target = round(v)` as a fixed-point rounding at a computed position."""
        assert isinstance(e.arg, Var) and self.alias is not None
        name = e.arg.name
        alias = self.alias

        def arg() -> Var:
            return Var(name, loc)

        # only a substituting format loses the sign of an overflow
        restore_sign = src.policy is _Policy.NAN_ON_OVERFLOW

        def rounding(nmin: Expr, reach: Expr) -> list[Stmt]:
            """`target = round(v)` under the format at `nmin`."""
            # the block holds the rounding alone, so a later pass can shift it
            out = target if not restore_sign else self.gensym.fresh('_t')
            stmts: list[Stmt] = [ContextStmt(
                UnderscoreId(), _ctx_call(nmin, src, alias, loc, reach),
                StmtBlock([Assign(out, None, Round(None, arg(), loc), loc)]), loc,
            )]
            if restore_sign:
                # a substituted NaN carries no sign; restore the input's.  Not
                # for a zero, which may have dropped its sign legitimately
                stmts.append(Assign(target, None, IfExpr(
                    IsNan(None, Var(out, loc), loc),
                    Copysign(None, Var(out, loc), arg(), loc),
                    Var(out, loc), loc,
                ), loc))
            return stmts

        e_name = self.gensym.fresh('e')
        exponent = Assign(e_name, None, Logb(None, arg(), loc), loc)

        # in the normal range the position follows the magnitude, capped by the
        # position of the bound's last digit: above that the bound is
        # unrepresentable, and everything up there overflows anyway
        pos_name = self.gensym.fresh('exp')
        scale: Expr = Sub(Var(e_name, loc), Integer(src.pmax - 1, loc), loc)
        if src.expmax is not None:
            scale = Min(None, [scale, Integer(src.expmax, loc)], loc)
        position = Assign(pos_name, None, scale, loc)
        # `exp = logb(x) - P + 1`, so `|x| < 2 ** (logb(x) + 1) == 2 ** (exp + P)`
        normal_reach = Pow(
            None, Integer(2, loc),
            Add(Var(pos_name, loc), Integer(src.pmax, loc), loc), loc,
        )
        at_scale = rounding(
            Sub(Var(pos_name, loc), Integer(1, loc), loc), normal_reach,
        )

        if src.emin is None:
            # no subnormals: every value rounds at a position of its own
            body: list[Stmt] = [exponent, position, *at_scale]
        else:
            # below `emin` the format is itself fixed-point: every value in
            # that range rounds at the same position, a constant
            assert src.expmin is not None
            # this branch is `logb(x) < emin`, so `|x| < 2 ** emin`
            sub_reach = number_literal(RealFloat(exp=src.emin, c=1), loc)
            normal = IfStmt(
                Compare([CompareOp.LT], [Var(e_name, loc), Integer(src.emin, loc)], loc),
                StmtBlock(rounding(Integer(src.expmin - 1, loc), sub_reach)),
                StmtBlock([position, *at_scale]),
                loc,
            )
            body = [exponent, normal]

        nan_v, pos_inf, neg_inf, pos_zero, neg_zero = src.specials

        def by_sign(pos: Float, neg: Float) -> Expr:
            return sign_choice(pos, neg, arg(), loc)

        def assign(value: Expr) -> StmtBlock:
            return StmtBlock([Assign(target, None, value, loc)])

        chain = IfStmt(
            IsNan(None, arg(), loc),
            assign(value_literal(nan_v, loc)),
            StmtBlock([IfStmt(
                IsInf(None, arg(), loc),
                assign(by_sign(pos_inf, neg_inf)),
                StmtBlock([IfStmt(
                    Compare([CompareOp.EQ], [arg(), Integer(0, loc)], loc),
                    assign(by_sign(pos_zero, neg_zero)),
                    StmtBlock(body),
                    loc,
                )]),
                loc,
            )]),
            loc,
        )

        # only the roundings need a rounding context; the rest is exact
        return ContextStmt(
            UnderscoreId(), ForeignVal(REAL, loc), StmtBlock([chain]), loc,
        )

    def _rewrite(self, stmt: ContextStmt, src: _Source) -> list[Stmt]:
        """The block's rounds, lowered.  Nothing rounds under the float context
        afterwards, so the block itself goes away."""
        stmts: list[Stmt] = []
        for s in stmt.body.stmts:
            if isinstance(s, Assign):
                assert isinstance(s.expr, Round) and isinstance(s.target, NamedId)
                stmts.append(self._lower_round(s.expr, s.target, s.loc, src))
            else:
                # a returned round lands in a temporary, which the return names
                assert isinstance(s, ReturnStmt) and isinstance(s.expr, Round)
                out = self.gensym.fresh('_t')
                stmts.append(self._lower_round(s.expr, out, s.loc, src))
                stmts.append(ReturnStmt(Var(out, s.loc), s.loc))
        return stmts


class FloatToFixed:
    """
    Transformation pass to express float rounding as fixed-point rounding.
    """

    @staticmethod
    def apply(
        func: FuncDef, *,
        where: int | None = None,
        eval_info: PartialEvalInfo | None = None,
    ) -> FuncDef:
        """
        Expresses float rounding in `func` as fixed-point rounding.

        `where` selects a single candidate block by index, in visit order
        (outermost-first); candidates are the blocks this pass could lower.
        If `None`, every candidate is lowered.
        """
        if not isinstance(func, FuncDef):
            raise TypeError(f'Expected \'FuncDef\', got {func}')
        check_where(where)

        if eval_info is None:
            eval_info = PartialEval.apply(func)

        return _FloatToFixedInstance(func, eval_info, where).apply()
