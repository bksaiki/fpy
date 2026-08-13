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
   if fp.isnan(x) or fp.isinf(x) or x == 0:
       with fp.MPBFixedContext(-25, 65504, overflow=..., enable_nan=True, enable_inf=True):
           y = fp.round(x)
   else:
       with fp.REAL:
           e = fp.logb(x)
       if e < -14:
           with fp.MPBFixedContext(-25, 65504, overflow=..., enable_nan=True, enable_inf=True):
               y = fp.round(x)
       else:
           with fp.REAL:
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

``logb`` is undefined on NaN, an infinity, and a zero, so those take their own
branch and round at the format's finest grid, which represents each of them
exactly — including the sign of a zero and of an infinity.  Rounding them into
the fixed format rather than passing them through keeps format inference from
widening the result to an unconstrained real.

Run :func:`fpy2.strategies.rescale_fixed` afterwards to shift the resulting
fixed-point rounding to digit position zero, where its values are integers.

Only a block whose body is entirely ``x = fp.round(v)`` (or a returned round)
over variables is rewritten, and only for an ``IEEEContext`` that overflows to
infinity and rounds deterministically.  Every other block is left unchanged.
The rewrite also needs the program to have ``fpy2`` in scope, since it names
the context constructor.
"""

from dataclasses import replace

from ..analysis import PartialEval, PartialEvalInfo
from ..ast.fpyast import (
    Assign,
    Attribute,
    BoolVal,
    Call,
    Compare,
    ContextStmt,
    ForeignVal,
    FuncDef,
    IfStmt,
    Integer,
    IsInf,
    IsNan,
    Location,
    Logb,
    Max,
    Min,
    NamedId,
    Or,
    Rational,
    ReturnStmt,
    Round,
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
    IEEEContext,
    MPBFixedContext,
    OverflowMode,
    RealFloat,
    RoundingMode,
)
from ..utils import CompareOp, Gensym


def _number(x: RealFloat, loc: Location | None) -> Expr:
    """`x` as an exact literal: every `RealFloat` is a dyadic rational."""
    if x.is_integer():
        return Integer(int(x), loc)
    r = x.as_rational()
    return Rational(None, r.numerator, r.denominator, loc)


def _attribute(alias: str, *names: str, loc: Location | None = None) -> Attribute:
    """The dotted name `alias.names[0].names[1]...`."""
    e: Expr = Var(NamedId(alias), loc)
    for name in names[:-1]:
        e = Attribute(e, name, loc)
    return Attribute(e, names[-1], loc)


def _ctx_call(
    nmin: Expr, src: IEEEContext, alias: str, loc: Location | None
) -> Call:
    """
    The fixed-point format holding `src`'s values, with its digits at `nmin + 1`.

    The bound and the rounding mode are `src`'s own, so overflow lands where
    the float format puts it.  NaN and infinity are enabled because the
    specials round into this format too.  Arguments matching the constructor's
    defaults are left out.
    """
    kwargs: list[tuple[str, Expr]] = [
        ('overflow', _attribute(alias, 'OverflowMode', OverflowMode.OVERFLOW.name, loc=loc)),
        ('enable_nan', BoolVal(True, loc)),
        ('enable_inf', BoolVal(True, loc)),
    ]
    if src.rm is not RoundingMode.RNE:
        kwargs.insert(0, ('rm', _attribute(alias, 'RoundingMode', src.rm.name, loc=loc)))

    return Call(
        _attribute(alias, 'MPBFixedContext', loc=loc),
        MPBFixedContext,
        (nmin, _number(src.maxval().as_real(), loc)),
        tuple(kwargs),
        loc,
    )


class _FloatToFixedInstance(DefaultTransformVisitor):
    """Rewrites every qualifying context statement in a function."""

    func: FuncDef
    eval_info: PartialEvalInfo
    gensym: Gensym
    alias: str | None

    def __init__(self, func: FuncDef, eval_info: PartialEvalInfo):
        self.func = func
        self.eval_info = eval_info
        self.gensym = Gensym(eval_info.def_use.names())
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

    def _source_ctx(self, stmt: ContextStmt) -> IEEEContext | None:
        """The block's context, if its rounding can be lowered to fixed-point."""
        # the position varies per value, so the rewrite has to name a context
        # constructor; without `fpy2` in scope there is nothing to name it by
        if self.alias is None:
            return None
        # a bound context is visible to the body as a value, which the rewrite changes
        if not isinstance(stmt.target, UnderscoreId):
            return None

        ctx = self.eval_info.by_expr.get(stmt.ctx)
        if not isinstance(ctx, IEEEContext):
            return None
        # the fixed target must be able to produce what overflow rounds to
        if ctx.overflow is not OverflowMode.OVERFLOW or not ctx.enable_inf:
            return None
        # stochastic rounding would have to draw its bits at the same position
        if ctx.num_randbits != 0:
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

        return ctx

    def _lower_round(
        self, e: Round, target: NamedId, loc: Location | None, src: IEEEContext
    ) -> Stmt:
        """`target = round(v)` as a fixed-point rounding at a computed position."""
        assert isinstance(e.arg, Var) and self.alias is not None
        name = e.arg.name
        alias = self.alias

        def arg() -> Var:
            return Var(name, loc)

        def rounding(nmin: Expr) -> ContextStmt:
            return ContextStmt(
                UnderscoreId(), _ctx_call(nmin, src, alias, loc),
                StmtBlock([Assign(target, None, Round(None, arg(), loc), loc)]), loc,
            )

        # below `emin` the format is itself fixed-point: every value in that
        # range rounds at the same position, so the context is a constant
        finest = Integer(src.expmin - 1, loc)

        e_name = self.gensym.fresh('e')
        exponent = ContextStmt(
            UnderscoreId(), ForeignVal(REAL, loc),
            StmtBlock([Assign(e_name, None, Logb(None, arg(), loc), loc)]), loc,
        )

        # in the normal range the grid follows the magnitude, bounded above by
        # the position of the bound's last digit; the subnormal branch already
        # covers everything below, so no lower clamp is needed here
        # named for the scale it holds, not for the context's `n` (which is
        # one position below it)
        pos_name = self.gensym.fresh('exp')
        position = ContextStmt(
            UnderscoreId(), ForeignVal(REAL, loc),
            StmtBlock([Assign(pos_name, None, Min(None, [
                Sub(Var(e_name, loc), Integer(src.pmax - 1, loc), loc),
                Integer(src.emax - src.pmax + 1, loc),
            ], loc), loc)]), loc,
        )
        normal = IfStmt(
            Compare([CompareOp.LT], [Var(e_name, loc), Integer(src.emin, loc)], loc),
            StmtBlock([rounding(finest)]),
            StmtBlock([position, rounding(Sub(Var(pos_name, loc), Integer(1, loc), loc))]),
            loc,
        )

        # `logb` is undefined on the specials, so they round at the format's
        # finest grid instead, which represents them exactly
        return IfStmt(
            Or([
                IsNan(None, arg(), loc),
                IsInf(None, arg(), loc),
                Compare([CompareOp.EQ], [arg(), Integer(0, loc)], loc),
            ], loc),
            StmtBlock([rounding(finest)]),
            StmtBlock([exponent, normal]),
            loc,
        )

    def _lower_block(self, stmt: ContextStmt, src: IEEEContext) -> list[Stmt]:
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
    def apply(func: FuncDef, *, eval_info: PartialEvalInfo | None = None) -> FuncDef:
        if not isinstance(func, FuncDef):
            raise TypeError(f'Expected \'FuncDef\', got {func}')

        if eval_info is None:
            eval_info = PartialEval.apply(func)

        return _FloatToFixedInstance(func, eval_info).apply()
