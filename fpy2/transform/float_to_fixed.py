"""
Express floating-point rounding as fixed-point rounding.

A float format rounds at a digit position that depends on the value: its grid
coarsens with magnitude.  For a format with precision ``P``, subnormal position
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
               with fp.MPBFixedContext(-25, 65504, overflow=..., enable_nan=True, enable_inf=True):
                   y = fp.round(x)
           else:
               exp = min((e - 10), 5)
               with fp.MPBFixedContext((exp - 1), 65504, overflow=..., enable_nan=True, enable_inf=True):
                   y = fp.round(x)

Below ``emin`` the format is fixed-point already: every value there rounds at
``EXP``, so that branch's context is a constant and nothing in it depends on
the exponent.  The normal branch needs no lower clamp because of it.

The upper clamp is what keeps the context constructible *and* keeps the format
shiftable afterwards: ``B`` lies on the grid at every position up to
``EMAX - P + 1`` and falls off it immediately above.  Clamping is also correct —
any ``x`` above that exceeds ``B`` and must overflow, which rounding at the
clamped position against bound ``B`` produces.

``logb`` is undefined on NaN, an infinity, and a zero, so each takes a branch
of its own.  All three are constants, so the branches assign what the format
makes of them, worked out when the format was described rather than at run
time.  Only the rounding blocks are under a rounding context; everything else
sits under ``REAL``, where it cannot be perturbed by whatever context encloses
the statement.

Run :func:`fpy2.strategies.rescale_fixed` afterwards to shift the resulting
fixed-point rounding to digit position zero, where its values are integers.

Applies to a float format whose overflow a fixed-point round can reproduce:
an ``IEEEContext``, or an ``EFloatContext`` that saturates or substitutes a
NaN, in each case rounding deterministically.  A format that substitutes for
its overflow gives no sign back, so the input's is restored afterwards.  Only
a block whose body is entirely ``x = fp.round(v)`` (or a returned round) over
variables is rewritten; every other block is left unchanged.  The rewrite also
needs the program to have ``fpy2`` in scope, since it names the context
constructor.
"""

from dataclasses import dataclass, replace
from enum import Enum, auto

from ..analysis import PartialEval, PartialEvalInfo
from ..ast.fpyast import (
    Assign,
    Attribute,
    BoolVal,
    Call,
    Compare,
    ConstInf,
    ConstNan,
    ContextStmt,
    Copysign,
    Decnum,
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
    Neg,
    Rational,
    ReturnStmt,
    Round,
    Signbit,
    Stmt,
    StmtBlock,
    Sub,
    UnderscoreId,
    Var,
)
from ..ast.visitor import DefaultTransformVisitor
from ..env import fpy_alias
from ..number import (
    REAL,
    Context,
    Float,
    EFloatContext,
    EFloatNanKind,
    IEEEContext,
    MPBFixedContext,
    OverflowMode,
    RealFloat,
    RoundingMode,
)
from ..utils import CompareOp, Gensym


class _Policy(Enum):
    """What a float format does with a value too large to represent."""

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
    emin: int
    """smallest normal exponent: below it the format is fixed-point"""
    expmin: int
    """position of the format's finest grid"""
    expmax: int
    """position of the bound's last digit, above which the bound leaves the grid"""
    maxval: RealFloat
    """the bound"""
    rm: RoundingMode
    """rounding mode"""
    policy: _Policy
    """what overflow produces"""
    specials: tuple[Float, Float, Float, Float, Float]
    """what NaN, `+Inf`, `-Inf`, `+0`, and `-0` round to"""


def _describe(ctx: Context) -> _Source | None:
    """
    `ctx` as a lowerable float format, or `None`.

    A format qualifies when the fixed-point round can reproduce it exactly:
    its rounding must be deterministic, and its overflow must land somewhere
    a fixed-point context can also land.
    """
    match ctx:
        # `IEEEContext` derives from `EFloatContext`, so it matches first
        case IEEEContext():
            if not ctx.enable_inf:
                return None
            policy = _Policy.INFINITE
        case EFloatContext():
            # a shifted exponent encoding is not accounted for below
            if ctx.eoffset != 0 or ctx.enable_inf:
                return None
            match ctx.nan_kind:
                case EFloatNanKind.NONE:
                    policy = _Policy.SATURATING
                case EFloatNanKind.MAX_VAL:
                    policy = _Policy.NAN_ON_OVERFLOW
                case _:
                    return None
        case _:
            return None

    if ctx.overflow is not OverflowMode.OVERFLOW:
        return None
    # stochastic rounding would have to draw its bits at the same position
    if ctx.num_randbits != 0:
        return None

    # `logb` is undefined on these, so each takes a branch of its own; since
    # they are constants, what the format makes of them is known right here
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

    return _Source(
        ctx.pmax, ctx.emin, ctx.expmin, ctx.emax - ctx.pmax + 1,
        ctx.maxval().as_real(), ctx.rm, policy, specials,
    )


def _number(x: RealFloat, loc: Location | None) -> Expr:
    """`x` as an exact literal: every `RealFloat` is a dyadic rational."""
    if x.is_integer():
        return Integer(int(x), loc)
    r = x.as_rational()
    return Rational(None, r.numerator, r.denominator, loc)


def _literal(v: Float, loc: Location | None) -> Expr:
    """`v` as a literal, whatever kind of value it is."""
    if v.is_zero() and v.s:
        # a negative zero has no rational form
        return Decnum('-0.0', loc)
    if not v.is_nar():
        return _number(v.as_real(), loc)
    e: Expr = ConstNan(None, loc) if v.isnan else ConstInf(None, loc)
    return Neg(e, loc) if v.s else e


def _identical(a: Float, b: Float) -> bool:
    """Whether two values are the same, sign and all."""
    if a.is_nar():
        return a.isnan == b.isnan and a.isinf == b.isinf and a.s == b.s
    return not b.is_nar() and a.as_real() == b.as_real() and a.s == b.s


def _attribute(alias: str, *names: str, loc: Location | None = None) -> Attribute:
    """The dotted name `alias.names[0].names[1]...`."""
    e: Expr = Var(NamedId(alias), loc)
    for name in names[:-1]:
        e = Attribute(e, name, loc)
    return Attribute(e, names[-1], loc)


def _ctx_call(
    nmin: Expr, src: _Source, alias: str, loc: Location | None
) -> Call:
    """
    The fixed-point format holding `src`'s values, with its digits at `nmin + 1`.

    The bound and the rounding mode are `src`'s own, and the remaining flags
    put overflow where the float format puts it.  Arguments matching the
    constructor's defaults are left out.
    """
    def mode(name: str, value) -> tuple[str, Expr]:
        return (name, _attribute(alias, type(value).__name__, value.name, loc=loc))

    kwargs: list[tuple[str, Expr]] = []
    if src.rm is not RoundingMode.RNE:
        kwargs.append(mode('rm', src.rm))

    match src.policy:
        case _Policy.INFINITE:
            # the specials round into this format too, so it holds them
            kwargs.append(mode('overflow', OverflowMode.OVERFLOW))
            kwargs.append(('enable_nan', BoolVal(True, loc)))
            kwargs.append(('enable_inf', BoolVal(True, loc)))
        case _Policy.SATURATING:
            # no NaN, no infinity: overflow has nowhere to go but the bound
            kwargs.append(mode('overflow', OverflowMode.SATURATE))
        case _Policy.NAN_ON_OVERFLOW:
            # an overflow would be infinite, which this format substitutes
            kwargs.append(mode('overflow', OverflowMode.OVERFLOW))
            kwargs.append(('enable_nan', BoolVal(True, loc)))
            kwargs.append(('inf_value', ConstNan(None, loc)))

    return Call(
        _attribute(alias, 'MPBFixedContext', loc=loc),
        MPBFixedContext,
        (nmin, _number(src.maxval, loc)),
        tuple(kwargs),
        loc,
    )


class _FloatToFixedInstance(DefaultTransformVisitor):
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

    def _source_ctx(self, stmt: ContextStmt) -> '_Source | None':
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
        signed = src.policy is _Policy.NAN_ON_OVERFLOW

        def rounding(nmin: Expr) -> list[Stmt]:
            """`target = round(v)` under the format at `nmin`."""
            # the block holds the rounding alone, so a later pass can shift it
            out = target if not signed else self.gensym.fresh('_t')
            stmts: list[Stmt] = [ContextStmt(
                UnderscoreId(), _ctx_call(nmin, src, alias, loc),
                StmtBlock([Assign(out, None, Round(None, arg(), loc), loc)]), loc,
            )]
            if signed:
                # a substituted NaN carries no sign of its own, so the input's
                # is put back; rounding never flips a sign, so this is the
                # identity on every other value
                stmts.append(
                    Assign(target, None, Copysign(None, Var(out, loc), arg(), loc), loc)
                )
            return stmts

        # below `emin` the format is itself fixed-point: every value in that
        # range rounds at the same position, so the context is a constant
        finest = Integer(src.expmin - 1, loc)

        e_name = self.gensym.fresh('e')
        exponent = Assign(e_name, None, Logb(None, arg(), loc), loc)

        # in the normal range the grid follows the magnitude, bounded above by
        # the position of the bound's last digit; the subnormal branch already
        # covers everything below, so no lower clamp is needed here.  The name
        # is the scale it holds, not the context's `n`, which is one below it
        pos_name = self.gensym.fresh('exp')
        position = Assign(pos_name, None, Min(None, [
            Sub(Var(e_name, loc), Integer(src.pmax - 1, loc), loc),
            Integer(src.expmax, loc),
        ], loc), loc)
        normal = IfStmt(
            Compare([CompareOp.LT], [Var(e_name, loc), Integer(src.emin, loc)], loc),
            StmtBlock(rounding(finest)),
            StmtBlock([position, *rounding(Sub(Var(pos_name, loc), Integer(1, loc), loc))]),
            loc,
        )

        # `logb` is undefined on NaN, an infinity, and a zero, so each takes a
        # branch of its own.  All three are constants, so what the format makes
        # of them was settled when the format was described: the branches assign
        # the results outright rather than rounding.
        nan_v, pos_inf, neg_inf, pos_zero, neg_zero = src.specials

        def by_sign(pos: Float, neg: Float) -> Expr:
            """The result for `+x` or `-x`, chosen by the sign of the input."""
            if _identical(pos, neg):
                return _literal(pos, loc)
            return IfExpr(
                Signbit(None, arg(), loc), _literal(neg, loc), _literal(pos, loc), loc,
            )

        def assign(value: Expr) -> StmtBlock:
            return StmtBlock([Assign(target, None, value, loc)])

        chain = IfStmt(
            IsNan(None, arg(), loc),
            assign(_literal(nan_v, loc)),
            StmtBlock([IfStmt(
                IsInf(None, arg(), loc),
                assign(by_sign(pos_inf, neg_inf)),
                StmtBlock([IfStmt(
                    Compare([CompareOp.EQ], [arg(), Integer(0, loc)], loc),
                    assign(by_sign(pos_zero, neg_zero)),
                    StmtBlock([exponent, normal]),
                    loc,
                )]),
                loc,
            )]),
            loc,
        )

        # everything outside the rounding blocks is exact: the constants above,
        # and the exponent arithmetic.  Under `REAL` none of it is at the mercy
        # of whatever context encloses this statement.
        return ContextStmt(
            UnderscoreId(), ForeignVal(REAL, loc), StmtBlock([chain]), loc,
        )

    def _lower_block(self, stmt: ContextStmt, src: '_Source') -> list[Stmt]:
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

    def _visit_block(self, block: StmtBlock, ctx: None):
        # a lowered block expands into several statements, so the splice
        # happens here rather than in `_visit_context`
        stmts: list[Stmt] = []
        for s in block.stmts:
            if isinstance(s, ContextStmt):
                src = self._source_ctx(s)
                if src is not None:
                    idx = self.site_idx
                    self.site_idx += 1
                    if self.where is None or idx == self.where:
                        stmts.extend(self._lower_block(s, src))
                        continue
            new_s, ctx = self._visit_statement(s, ctx)
            stmts.append(new_s)
        return StmtBlock(stmts), ctx


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
        if where is not None and not isinstance(where, int):
            raise TypeError(f'expected an \'int\' or None for where, got `{where}`')

        if eval_info is None:
            eval_info = PartialEval.apply(func)

        return _FloatToFixedInstance(func, eval_info, where).apply()
