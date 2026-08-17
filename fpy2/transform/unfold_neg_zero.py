"""
Unfold the sign of zero out of a rounding context.

A fixed-point format with ``enable_neg_zero`` keeps the sign of a value that
rounds to zero: ``round_C(-1e-30)`` is ``-0.0``, not ``+0.0``.  A format
without it returns ``+0.0``.  Stated as program text, the first is the second
plus a sign restoration:

.. code-block:: python

   # Before
   with C:                       # enable_neg_zero=True
       r = fp.round(x)

   # After
   with fp.REAL:
       with C_:                  # C with enable_neg_zero=False
           t = fp.round(x)
       if t == 0:
           r = fp.copysign(t, x)
       else:
           r = t

The sign comes from the operand, which is what the format would have kept.
``fp.copysign`` under ``fp.REAL`` is exact for every value.

C++ has no integer type with a signed zero, so no integer rung on the storage
ladder admits one: this one flag decides whether a rounding reaches integer
storage.  Anything targeting integer arithmetic — fixed-point hardware, an
integer-only DSP, a bit-exact reference in ``int16_t`` — needs the sign out of
the format.

The claim behind the rewrite is that ``C`` and ``C_`` agree everywhere except
on the sign of a zero result, and that a zero result carries the operand's
sign.  Both are checked against the source rather than assumed, over the
values where they could fail: a format is declined when the emitted program
would disagree with it.  Wrapping overflow is the common decliner — it wraps
by ordinal over the full signed range, so a negative operand can land on
``+0``, which no sign restoration from the operand reproduces.  A ``nan_value``
or ``inf_value`` that is itself a zero is declined for the same reason: the
fixup would hand it the special operand's sign.

Only a block whose body is entirely ``x = fp.round(v)`` (or a returned round)
over variables is rewritten.  ``Cast`` is excluded: it asserts exactness, and
an exact result never rounds to zero from anything but zero.

`SMFixedContext` has its signed zero by construction, so it is rebuilt as the
`MPBFixedContext` it derives from; the emitted context no longer names the
source's own class.  `FixedContext` (two's complement) already has no signed
zero, so it is never a candidate.
"""

from dataclasses import dataclass

from ..analysis import PartialEval, PartialEvalInfo
from ..ast.fpyast import (
    Assign,
    BoolVal,
    Call,
    Compare,
    ContextStmt,
    Copysign,
    Expr,
    ForeignVal,
    FuncDef,
    IfStmt,
    Integer,
    Location,
    NamedId,
    ReturnStmt,
    Round,
    Stmt,
    StmtBlock,
    UnderscoreId,
    Var,
)
from ..number import (
    REAL,
    Context,
    Float,
    MPBFixedContext,
    MPFixedContext,
    RealFloat,
)
from ..utils import CompareOp, Gensym
from .utils import BlockRewriter, check_where, same_value, shift

_NAN = Float(isnan=True)

_FixedCtx = MPFixedContext | MPBFixedContext
"""
the fixed-point contexts that state a sign of zero as a flag.

`SMFixedContext` derives from `MPBFixedContext`; `FixedContext` does too, but
two's complement has no signed zero, so the probe never claims it.
"""


@dataclass(frozen=True)
class _Source:
    """A format that keeps its signed zero, in the terms the rewrite needs."""

    ctx: _FixedCtx
    """the source format"""
    dropped: _FixedCtx
    """the same format with the signed zero dropped, which the block rounds
    under instead"""


def _rounded(ctx: Context, x: Float | RealFloat) -> Float | None:
    """`x` under `ctx`, or `None` where the format has no value for it."""
    try:
        return ctx.round(x)
    except (ValueError, OverflowError):
        return None


def _agrees(a: Float | None, b: Float | None) -> bool:
    """Whether two rounding outcomes match, a refusal counting as an outcome."""
    if a is None or b is None:
        return a is None and b is None
    return same_value(a, b)


def _without_neg_zero(ctx: _FixedCtx) -> _FixedCtx | None:
    """`ctx` with its signed zero dropped: the same format, one zero.
    `None` if the result will not construct."""
    try:
        if type(ctx) is MPFixedContext or type(ctx) is MPBFixedContext:
            return ctx.with_params(enable_neg_zero=False)
        if isinstance(ctx, MPBFixedContext):
            # a subclass (`SMFixedContext`) has its signed zero by
            # construction, so the flag comes off in the base class
            return MPBFixedContext(
                ctx.nmin, ctx.pos_maxval, ctx.rm, ctx.overflow,
                ctx.num_randbits,
                neg_maxval=ctx.neg_maxval, rng=ctx.rng,
                enable_nan=ctx.enable_nan, enable_inf=ctx.enable_inf,
                enable_neg_zero=False,
                nan_value=ctx.nan_value, inf_value=ctx.inf_value,
            )
        # an `MPFixedContext` subclass this rewrite does not know how to rebuild
        return None
    except ValueError:
        return None


def _probes(ctx: _FixedCtx) -> list[Float | RealFloat]:
    """
    Values on which the emitted program could disagree with `ctx`: both
    zeros, values rounding to zero from either side, the specials, and — for
    a bounded format — operands past the bound, including one full trip
    around the wrapped range, which lands back on zero.
    """
    xs: list[Float | RealFloat] = [
        _NAN, Float(x=_NAN, s=True),
        Float(isinf=True), Float(isinf=True, s=True),
        Float(c=0), Float(c=0, s=True),
    ]
    grid = [
        RealFloat(exp=ctx.nmin - 3, c=1),   # far below the grid
        RealFloat(exp=ctx.nmin, c=1),       # the tie at half a step
        RealFloat(exp=ctx.nmin, c=3),
        RealFloat(exp=ctx.nmin + 1, c=1),   # the grid's finest step
    ]
    if isinstance(ctx, MPBFixedContext):
        step = RealFloat(exp=ctx.nmin + 1, c=1)
        span = ctx.pos_maxval - ctx.neg_maxval + step
        grid += [
            ctx.pos_maxval, ctx.neg_maxval,
            shift(ctx.pos_maxval, 1), shift(ctx.pos_maxval, 64),
            span, span + step, shift(span, 1),
        ]
    for g in grid:
        xs.append(RealFloat(exp=g.exp, c=g.c))
        xs.append(RealFloat(s=True, exp=g.exp, c=g.c))
    return xs


def _emitted(dropped: _FixedCtx, x: Float | RealFloat) -> Float | None:
    """What the generated code yields for `x`: the rounding under the
    format without the signed zero, then `copysign` from the operand where
    the result is zero."""
    t = _rounded(dropped, x)
    if t is None:
        return None
    if not t.is_nar() and t.is_zero():
        return Float(c=0, s=x.s)
    return t


def _reproduced(ctx: _FixedCtx, dropped: _FixedCtx) -> bool:
    """Whether the emitted program agrees with `ctx` on every probe."""
    return all(
        _agrees(_rounded(ctx, x), _emitted(dropped, x)) for x in _probes(ctx)
    )


def _ctx_expr(e: Expr, src: _Source) -> Expr:
    """
    The dropped context as an expression.  A constructor call keeps its
    written form with only the flag stated, so the rewritten program reads
    like the original; anything else — the rebuilt context itself.
    """
    if (
        isinstance(e, Call) and e.fn is type(src.ctx)
        and type(src.dropped) is type(src.ctx)
    ):
        # the flag is keyword-only in both constructors
        kwargs = tuple(kv for kv in e.kwargs if kv[0] != 'enable_neg_zero')
        kwargs += (('enable_neg_zero', BoolVal(False, e.loc)),)
        return Call(e.func, e.fn, e.args, kwargs, e.loc)
    return ForeignVal(src.dropped, e.loc)


class _UnfoldNegZeroInstance(BlockRewriter):
    """Rewrites every qualifying context statement in a function."""

    func: FuncDef
    eval_info: PartialEvalInfo
    gensym: Gensym
    where: int | None
    site_idx: int

    def __init__(
        self, func: FuncDef, eval_info: PartialEvalInfo,
        where: int | None = None,
    ):
        self.func = func
        self.eval_info = eval_info
        self.gensym = Gensym(eval_info.def_use.names())
        self.where = where
        # Counts *candidate* blocks (those the rewrite could unfold) in
        # visit order, outermost-first.  `where` selects one by this index.
        self.site_idx = 0

    def apply(self) -> FuncDef:
        return self._visit_function(self.func, None)

    def _candidate(self, stmt: ContextStmt) -> _Source | None:
        """The block's format, if its sign of zero can be taken out of its
        context."""
        # a bound context is visible to the body as a value, which the rewrite changes
        if not isinstance(stmt.target, UnderscoreId):
            return None

        ctx = self.eval_info.by_expr.get(stmt.ctx)
        if not isinstance(ctx, _FixedCtx):
            return None
        # stochastic rounding would have to draw its bits under the same format
        if ctx.num_randbits != 0:
            return None

        # only a rounding rounds onto zero; `Cast` asserts exactness, and an
        # exact result never rounds to zero from anything but zero
        for s in stmt.body.stmts:
            match s:
                case Assign(target=NamedId()) | ReturnStmt():
                    if not isinstance(s.expr, Round) or not isinstance(s.expr.arg, Var):
                        return None
                case _:
                    return None

        # only a format that keeps its signed zero has anything to unfold
        if not ctx.round(Float(c=0, s=True)).s:
            return None
        dropped = _without_neg_zero(ctx)
        if dropped is None or not _reproduced(ctx, dropped):
            return None
        return _Source(ctx, dropped)

    def _unfold(
        self, e: Round, target: NamedId, loc: Location | None, src: _Source,
        ctx_expr: Expr,
    ) -> Stmt:
        """`target = round(v)` as a one-zero rounding plus a sign restoration."""
        assert isinstance(e.arg, Var)
        name = e.arg.name

        def arg() -> Var:
            return Var(name, loc)

        # the rounding, under the format the sign came out of
        t = self.gensym.fresh('t')
        rounding = ContextStmt(
            UnderscoreId(), ctx_expr,
            StmtBlock([Assign(t, None, Round(None, arg(), loc), loc)]), loc,
        )

        # a rounding onto zero has lost only its sign, which the operand
        # still holds; the branch is dead for every other value
        fixup = IfStmt(
            Compare([CompareOp.EQ], [Var(t, loc), Integer(0, loc)], loc),
            StmtBlock([Assign(
                target, None, Copysign(None, Var(t, loc), arg(), loc), loc,
            )]),
            StmtBlock([Assign(target, None, Var(t, loc), loc)]), loc,
        )

        # the comparison and the sign transfer are exact whatever context
        # encloses this statement; the rounding sets its own
        return ContextStmt(
            UnderscoreId(), ForeignVal(REAL, loc),
            StmtBlock([rounding, fixup]), loc,
        )

    def _rewrite(self, stmt: ContextStmt, src: _Source) -> list[Stmt]:
        """The block's rounds, with the sign of zero taken out of the context.
        Nothing rounds under the source context afterwards, so the block
        itself goes away."""
        stmts: list[Stmt] = []
        for s in stmt.body.stmts:
            # each emitted block gets its own context expression
            ctx_expr = _ctx_expr(stmt.ctx, src)
            if isinstance(s, Assign):
                assert isinstance(s.expr, Round) and isinstance(s.target, NamedId)
                stmts.append(self._unfold(s.expr, s.target, s.loc, src, ctx_expr))
            else:
                # a returned round lands in a temporary, which the return names
                assert isinstance(s, ReturnStmt) and isinstance(s.expr, Round)
                out = self.gensym.fresh('t')
                stmts.append(self._unfold(s.expr, out, s.loc, src, ctx_expr))
                stmts.append(ReturnStmt(Var(out, s.loc), s.loc))
        return stmts


class UnfoldNegZero:
    """
    Transformation pass to state a context's sign of zero as program text.
    """

    @staticmethod
    def apply(
        func: FuncDef, *,
        where: int | None = None,
        eval_info: PartialEvalInfo | None = None,
    ) -> FuncDef:
        """
        Takes the signed zero out of every qualifying rounding context in
        `func`, restoring the sign with `copysign` after the rounding.

        `where` selects a single candidate block by index, in visit order
        (outermost-first); candidates are the blocks this pass could rewrite.
        If `None`, every candidate is rewritten.
        """
        if not isinstance(func, FuncDef):
            raise TypeError(f'Expected \'FuncDef\', got {func}')
        check_where(where)

        if eval_info is None:
            eval_info = PartialEval.apply(func)

        return _UnfoldNegZeroInstance(func, eval_info, where).apply()
