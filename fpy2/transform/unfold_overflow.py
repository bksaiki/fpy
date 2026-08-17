"""
Unfold a rounding context's overflow into program text.

A bounded format decides two things at once: which values it represents, and
what becomes of a value too large for it.  IEEE 754 defines the second in terms of
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

With `early_check`, a check on the operand comes first, so nothing past the
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
``round_U(x) >= round_U(infval) = infval > maxval``.  The check is sound but
not complete — ``FP16``'s tie at ``65520`` lies below ``infval`` yet rounds up
to ``65536`` — so the comparison after the rounding stays either way.

What overflow produces, and what the format makes of NaN and the infinities,
are asked of the source context rather than assumed.  A special value gets a
branch only where the emitted program would otherwise disagree with it, so an
IEEE source gets none and a format that substitutes its bound for NaN gets one.

A bounded *fixed-point* format unfolds the same way, its counterpart being
:class:`fpy2.MPFixedContext` at the same digit position.  That counterpart can
state what it does with NaN, an infinity and a negative zero, so it inherits all
three from the source and the checks have nothing to say about them.  What it
cannot state is a refusal, and a fixed-point format commonly refuses NaN and the
infinities outright — hence the finiteness test in front of the early
check, which would otherwise claim an infinity as an overflow.

Applies to a format whose overflow is a constant of its own: wrapping gives a
different answer at every magnitude, and an unsigned format states no bound
below zero, so neither is rewritten.

Only a block whose body is entirely ``x = fp.round(v)`` (or a returned round)
over variables is rewritten.  ``Cast`` is excluded: it asserts exactness, which
this rewrite does not preserve.
"""

from dataclasses import dataclass, replace

from ..analysis import (
    PartialEval,
    PartialEvalInfo,
    ValueClass,
    ValueClassAnalysis,
    ValueClassInfer,
)
from ..ast.fpyast import (
    And,
    Assign,
    BoolVal,
    Call,
    Compare,
    ContextStmt,
    Expr,
    ForeignVal,
    FuncDef,
    IfStmt,
    Integer,
    IsFinite,
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
    MPBFixedContext,
    MPBFloatContext,
    MPFixedContext,
    MPSFloatContext,
    RealFloat,
    RoundingMode,
)
from ..utils import CompareOp, Gensym
from .utils import (
    BlockRewriter,
    agrees,
    attribute,
    check_where,
    number_literal,
    same_value,
    shift,
    sign_choice,
    try_round,
    value_literal,
)

_NAN = Float(isnan=True)
_POS_INF = Float(isinf=True)
_NEG_INF = Float(isinf=True, s=True)

_BoundedCtx = EFloatContext | MPBFloatContext | MPBFixedContext
"""
the bounded contexts, whose bound this rewrite states as program text.

`FixedContext` and `SMFixedContext` derive from `MPBFixedContext`, so the last
member covers every bounded fixed-point format.
"""

_Unbounded = MPSFloatContext | MPFixedContext
"""their counterparts, which state only the representable values"""


@dataclass(frozen=True)
class _Source:
    """A bounded format, in the terms the rewrite needs."""

    unbounded: _Unbounded
    """the same format with no bound, which the block rounds under instead"""
    maxval: RealFloat
    """largest representable value; above it a rounding has overflowed"""
    neg_maxval: RealFloat
    """its negative counterpart, which a format need not mirror"""
    infval: RealFloat
    """smallest value that certainly overflows, for `early_check`"""
    neg_infval: RealFloat
    """its negative counterpart"""
    over_pos: Float
    """what an overflow above the bound produces"""
    over_neg: Float
    """what an overflow below it produces"""
    drop_neg_zero: bool
    """
    whether a value rounding to zero has to be given back a positive sign.

    Only where the source drops the negative zero and the counterpart keeps it,
    which a float counterpart cannot help doing — `MPSFloatContext` has no way
    to turn it off, while `MPFixedContext` does.
    """
    check_finite: bool
    """
    whether the early check has to exclude the infinities.

    An infinity is past every bound, so the check would claim it as an overflow
    — but a fixed-point format rejects it outright, and the rewrite has no way
    to state a refusal.  Testing finiteness first lets it reach the rounding,
    which refuses it exactly as the source did.
    """
    specials: tuple[tuple[Float, Float] | None, ...]
    """
    what the format makes of NaN and of the infinities, as a `(positive,
    negative)` pair each; `None` where the rewrite reproduces it already
    """


def _holds(x: Float, op: CompareOp, b: RealFloat) -> bool:
    """
    Whether ``x op b``, as the emitted comparison decides it.

    Takes the same `CompareOp` the emitter writes out, so the two cannot
    drift.  A NaN compares false to everything, which is what leaves it to a
    branch of its own.
    """
    if x.isnan:
        return False
    if x.isinf:
        c = -1 if x.s else 1
    else:
        xr, br = x.as_rational(), b.as_rational()
        c = (xr > br) - (xr < br)
    match op:
        case CompareOp.GE:
            return c >= 0
        case CompareOp.LE:
            return c <= 0
        case CompareOp.GT:
            return c > 0
        case CompareOp.LT:
            return c < 0
        case _:
            raise RuntimeError(f'unexpected comparison {op}')


class _Prober:
    """
    What a bounded context does at its edges, and whether a rounding under its
    unbounded counterpart plus a comparison reproduces it.
    """

    ctx: _BoundedCtx
    unbounded: _Unbounded
    early_check: bool

    def __init__(self, ctx: _BoundedCtx, unbounded: _Unbounded, early_check: bool):
        self.ctx = ctx
        self.unbounded = unbounded
        self.early_check = early_check

    def describe(self) -> _Source | None:
        """`ctx` as a lowerable bounded format, or `None`."""
        ctx = self.ctx
        try:
            maxval = ctx.maxval().as_real()
            neg_maxval = ctx.maxval(s=True).as_real()
            infval = ctx.infval().as_real()
            neg_infval = ctx.infval(s=True).as_real()
        except ValueError:
            # an unsigned format states no bound below zero
            return None
        if maxval.is_zero() or not neg_maxval.s:
            # a format representing no non-zero value has no overflow to state
            return None

        over = self._overflow(maxval, neg_maxval)
        if over is None:
            return None
        over_pos, over_neg = over

        zero = Float(c=0, s=True)
        src = _Source(
            self.unbounded, maxval, neg_maxval, infval, neg_infval,
            over_pos, over_neg,
            drop_neg_zero=self.unbounded.round(zero).s and not ctx.round(zero).s,
            check_finite=try_round(self.unbounded, _POS_INF) is None,
            specials=(),
        )
        # which specials need a branch depends on what the rest of `src` makes
        # of them, so they are filled in against it
        specials = self._specials(src)
        return None if specials is None else replace(src, specials=specials)

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
            near = [self.ctx.round(shift(b, 1)) for b in (maxval, neg_maxval)]
            far = [self.ctx.round(shift(b, 64)) for b in (maxval, neg_maxval)]
        except (ValueError, OverflowError):
            # a format that refuses to round an overflow at all
            return None

        if not all(same_value(a, b) for a, b in zip(near, far)):
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
            want = (try_round(self.ctx, pos), try_round(self.ctx, neg))
            got = (self._emitted(pos, src), self._emitted(neg, src))
            if all(agrees(a, b) for a, b in zip(want, got)):
                out.append(None)
            elif want[0] is None or want[1] is None:
                # the source rejects it where the rewrite would not, and a
                # branch can only assign a value, not refuse one
                return None
            else:
                out.append((want[0], want[1]))
        return tuple(out)

    def _emitted(self, x: Float, src: _Source) -> Float | None:
        """
        What the generated code yields for `x`, special branches aside;
        `None` where the rounding rejects it, as the source may too.
        """
        if self.early_check and not (src.check_finite and x.is_nar()):
            if _holds(x, CompareOp.GE, src.infval):
                return src.over_pos
            if _holds(x, CompareOp.LE, src.neg_infval):
                return src.over_neg

        t = try_round(self.unbounded, x)
        if t is None:
            return None
        if _holds(t, CompareOp.GT, src.maxval):
            return src.over_pos
        if _holds(t, CompareOp.LT, src.neg_maxval):
            return src.over_neg
        if src.drop_neg_zero and not t.is_nar() and t.is_zero():
            return Float(c=0)
        return t


def _unbounded(ctx: _BoundedCtx) -> _Unbounded | None:
    """
    `ctx` with its bound removed: the same representable values, with nothing to
    overflow.

    A float format keeps its precision and subnormal floor; a fixed-point one
    keeps its digit position.  `None` if the result will not construct.
    """
    try:
        if isinstance(ctx, MPBFixedContext):
            # a fixed-point format states what it does with NaN, an infinity
            # and a negative zero, and none of that depends on the bound
            return MPFixedContext(
                ctx.nmin, ctx.rm,
                enable_nan=ctx.enable_nan,
                enable_inf=ctx.enable_inf,
                enable_neg_zero=ctx.round(Float(c=0, s=True)).s,
                nan_value=ctx.nan_value,
                inf_value=ctx.inf_value,
            )
        # `MPSFloatContext` is built through its constructor rather than
        # `from_format`, which rejects a format without NaN or infinity (see
        # the `TODO` in `fpy2.number.context.mps_float`).  Handing those back
        # is harmless: they reach a rounding only as its operand, and the
        # caller branches on whichever of them the source treats differently.
        # A shifted exponent encoding needs no attention either — `pmax` and
        # `emin` account for it.
        return MPSFloatContext(ctx.pmax, ctx.emin, ctx.rm)
    except ValueError:
        return None


def _unbounded_expr(
    ctx: _Unbounded, alias: str | None, loc: Location | None
) -> Expr:
    """
    `ctx` as an expression, written as a constructor call where the program
    has a name for `fpy2` to write it with.  Arguments matching the
    constructor's defaults are left out.
    """
    if alias is None:
        return ForeignVal(ctx, loc)

    kwargs: list[tuple[str, Expr]] = []
    if ctx.rm is not RoundingMode.RNE:
        kwargs.append(('rm', attribute(alias, 'RoundingMode', ctx.rm.name, loc=loc)))

    if isinstance(ctx, MPSFloatContext):
        args: tuple[Expr, ...] = (Integer(ctx.pmax, loc), Integer(ctx.emin, loc))
    else:
        args = (Integer(ctx.nmin, loc),)
        if ctx.enable_nan:
            kwargs.append(('enable_nan', BoolVal(True, loc)))
        if ctx.enable_inf:
            kwargs.append(('enable_inf', BoolVal(True, loc)))
        if not ctx.enable_neg_zero:
            kwargs.append(('enable_neg_zero', BoolVal(False, loc)))
        for name, v in (('nan_value', ctx.nan_value), ('inf_value', ctx.inf_value)):
            if v is not None:
                kwargs.append((name, value_literal(v, loc)))

    return Call(
        attribute(alias, type(ctx).__name__, loc=loc),
        type(ctx), args, tuple(kwargs), loc,
    )


class _UnfoldOverflowInstance(BlockRewriter):
    """Rewrites every qualifying context statement in a function."""

    func: FuncDef
    eval_info: PartialEvalInfo
    class_info: ValueClassAnalysis
    gensym: Gensym
    where: int | None
    early_check: bool
    alias: str | None
    used_alias: bool
    site_idx: int

    def __init__(
        self, func: FuncDef, eval_info: PartialEvalInfo,
        class_info: ValueClassAnalysis,
        where: int | None = None, early_check: bool = False,
    ):
        self.func = func
        self.eval_info = eval_info
        self.class_info = class_info
        self.gensym = Gensym(eval_info.def_use.names())
        self.where = where
        self.early_check = early_check
        # the name the program calls `fpy2` by, which the emitted context is
        # written with; without one it falls back to the context value itself
        self.alias = fpy_alias(func.env)
        self.used_alias = False
        # Counts *candidate* blocks (those the rewrite could unfold) in
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

        unbounded = _unbounded(ctx)
        if unbounded is None:
            return None
        return _Prober(ctx, unbounded, self.early_check).describe()

    def _unfold(
        self, e: Round, target: NamedId, loc: Location | None, src: _Source
    ) -> Stmt:
        """`target = round(v)` as an unbounded rounding plus a bound check."""
        assert isinstance(e.arg, Var)
        name = e.arg.name
        cls = self.class_info.classify(e.arg)

        def arg() -> Var:
            return Var(name, loc)

        def assign(v: Expr) -> StmtBlock:
            return StmtBlock([Assign(target, None, v, loc)])

        def past(
            operand: Expr, op: CompareOp, bound: RealFloat, over: Float,
            rest: StmtBlock, finite: bool = False,
        ) -> StmtBlock:
            """`rest`, behind a branch taking `operand` past `bound`."""
            cond: Expr = Compare([op], [operand, number_literal(bound, loc)], loc)
            if finite:
                cond = And([IsFinite(None, operand, loc), cond], loc)
            return StmtBlock([IfStmt(cond, assign(value_literal(over, loc)), rest, loc)])

        # the rounding, under the format the bound came out of: with nothing
        # left to overflow, it only rounds
        t = self.gensym.fresh('t')
        self.used_alias |= self.alias is not None
        rounding = ContextStmt(
            UnderscoreId(), _unbounded_expr(src.unbounded, self.alias, loc),
            StmtBlock([Assign(t, None, Round(None, arg(), loc), loc)]), loc,
        )

        # a result past the bound overflowed, whichever side it left by
        rest: StmtBlock = assign(Var(t, loc))
        if src.drop_neg_zero:
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
        if self.early_check:
            # the finiteness test keeps an infinity from being claimed as an
            # overflow; an operand that cannot be one does not need it
            g = src.check_finite and bool(cls & (ValueClass.NAN | ValueClass.INF))
            body = past(arg(), CompareOp.LE, src.neg_infval, src.over_neg, body, g)
            body = past(arg(), CompareOp.GE, src.infval, src.over_pos, body, g)

        # a special value the rounding and the checks would not reproduce --
        # unless the operand is never that kind of value
        atoms = (ValueClass.NAN, ValueClass.INF)
        for atom, test, want in zip(atoms, (IsNan, IsInf), src.specials):
            if want is None or not (atom & cls):
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
                stmts.append(self._unfold(s.expr, s.target, s.loc, src))
            else:
                # a returned round lands in a temporary, which the return names
                assert isinstance(s, ReturnStmt) and isinstance(s.expr, Round)
                out = self.gensym.fresh('t')
                stmts.append(self._unfold(s.expr, out, s.loc, src))
                stmts.append(ReturnStmt(Var(out, s.loc), s.loc))
        return stmts


class UnfoldOverflow:
    """
    Transformation pass to state a context's overflow as program text.
    """

    @staticmethod
    def apply(
        func: FuncDef, *,
        where: int | None = None,
        early_check: bool = False,
        eval_info: PartialEvalInfo | None = None,
        class_info: ValueClassAnalysis | None = None,
    ) -> FuncDef:
        """
        Takes the bound out of every qualifying rounding context in `func`.

        `where` selects a single candidate block by index, in visit order
        (outermost-first); candidates are the blocks this pass could rewrite.
        If `None`, every candidate is rewritten.

        With `early_check`, a check on the operand precedes the rounding, so
        nothing certain to overflow is rounded at all.
        """
        if not isinstance(func, FuncDef):
            raise TypeError(f'Expected \'FuncDef\', got {func}')
        check_where(where)

        if eval_info is None:
            eval_info = PartialEval.apply(func)
        if class_info is None:
            class_info = ValueClassInfer.analyze(func)

        return _UnfoldOverflowInstance(
            func, eval_info, class_info, where, early_check,
        ).apply()
