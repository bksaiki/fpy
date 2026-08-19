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
from ..ast.visitor import DefaultTransformVisitor
from ..number import (
    REAL,
    Context,
    Float,
    MPBFixedContext,
    MPFixedContext,
    RealFloat,
)
from ..utils import CompareOp, Gensym
from .utils import (
    BlockRewriter,
    Cursor,
    Declined,
    EditLog,
    StmtCursor,
    agrees,
    check_where,
    fixed_probes,
    is_rounding_block,
    rounding_block,
    stmt_sites,
    try_round,
)

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


def _emitted(dropped: _FixedCtx, x: Float | RealFloat) -> Float | None:
    """What the generated code yields for `x`: the rounding under the
    format without the signed zero, then `copysign` from the operand where
    the result is zero."""
    t = try_round(dropped, x)
    if t is None:
        return None
    if not t.is_nar() and t.is_zero():
        return Float(c=0, s=x.s)
    return t


def _reproduced(ctx: _FixedCtx, dropped: _FixedCtx) -> bool:
    """Whether the emitted program agrees with `ctx` on every probe."""
    return all(
        agrees(try_round(ctx, x), _emitted(dropped, x)) for x in fixed_probes(ctx)
    )


def _ctx_expr(e: Expr, src: _Source) -> Expr:
    """
    The dropped context as an expression, in fresh nodes.  A constructor call
    keeps its written form with only the flag stated, so the rewritten
    program reads like the original; anything else — the rebuilt context
    itself.
    """
    if (
        isinstance(e, Call) and e.fn is type(src.ctx)
        and type(src.dropped) is type(src.ctx)
    ):
        # a structurally-fresh copy: each emitted block must occupy distinct
        # AST nodes, and the source expression stays in place under `where`
        call = DefaultTransformVisitor()._visit_expr(e, None)
        assert isinstance(call, Call)
        # the flag is keyword-only in both constructors
        kwargs = tuple(kv for kv in call.kwargs if kv[0] != 'enable_neg_zero')
        kwargs += (('enable_neg_zero', BoolVal(False, e.loc)),)
        return Call(call.func, call.fn, call.args, kwargs, call.loc)
    return ForeignVal(src.dropped, e.loc)


class _UnfoldNegZeroInstance(BlockRewriter):
    """Rewrites every qualifying context statement in a function."""

    _casts = False
    """whether a `fp.cast` block counts as a candidate"""

    func: FuncDef
    eval_info: PartialEvalInfo
    gensym: Gensym
    where: int | Cursor | None
    site_idx: int

    def __init__(
        self, func: FuncDef, eval_info: PartialEvalInfo,
        where: int | Cursor | None = None,
    ):
        self.func = func
        self.eval_info = eval_info
        self.gensym = Gensym(eval_info.def_use.names())
        self.where = where

    def apply(self) -> FuncDef:
        return self._visit_function(self.func, None)

    def _candidate(self, stmt: ContextStmt) -> list[Var] | None:
        """Only a rounding rounds onto zero; `Cast` asserts exactness, and
        an exact result never rounds to zero from anything but zero."""
        return rounding_block(stmt, casts=self._casts)

    def _verify(self, stmt: ContextStmt, args: list[Var]) -> _Source | Declined:
        """The block's format, if its sign of zero can be taken out of its
        context."""
        ctx = self.eval_info.by_expr.get(stmt.ctx)
        if not isinstance(ctx, Context):
            return Declined('the context is not statically known')
        if not isinstance(ctx, _FixedCtx):
            return Declined(
                'the context is not a fixed-point format '
                '(`MPFixedContext` or `MPBFixedContext`)'
            )
        if ctx.num_randbits != 0:
            return Declined(
                'stochastic rounding would have to draw its bits under the '
                'same format'
            )

        # only a format that keeps its signed zero has anything to unfold
        if not ctx.round(Float(c=0, s=True)).s:
            return Declined('the format has one zero; there is no sign to take out')
        dropped = _without_neg_zero(ctx)
        if dropped is None:
            return Declined('the format cannot be rebuilt without its signed zero')
        if not _reproduced(ctx, dropped):
            return Declined(
                'the rewrite would disagree with the format on an edge value '
                '(wrapping overflow is the common cause)'
            )
        return _Source(ctx, dropped)

    def _unfold(
        self, e: Round, target: NamedId, loc: Location | None, ctx_expr: Expr,
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
                stmts.append(self._unfold(s.expr, s.target, s.loc, ctx_expr))
            else:
                # a returned round lands in a temporary, which the return names
                assert isinstance(s, ReturnStmt) and isinstance(s.expr, Round)
                out = self.gensym.fresh('t')
                stmts.append(self._unfold(s.expr, out, s.loc, ctx_expr))
                stmts.append(ReturnStmt(Var(out, s.loc), s.loc))
        return stmts


class UnfoldNegZero:
    """
    Transformation pass to state a context's sign of zero as program text.
    """

    @staticmethod
    def sites(func: FuncDef, within: Cursor | None = None) -> list[StmtCursor]:
        """The candidate rounding blocks of `func`, in visit order --
        what a `where` index counts, whether or not each verifies.
        """
        casts = _UnfoldNegZeroInstance._casts
        return stmt_sites(func, lambda s: is_rounding_block(s, casts=casts), within)

    @staticmethod
    def apply(
        func: FuncDef, *,
        where: int | Cursor | None = None,
        eval_info: PartialEvalInfo | None = None,
    ) -> FuncDef:
        """
        Takes the signed zero out of every qualifying rounding context in
        `func`, restoring the sign with `copysign` after the rounding.

        `where` selects one structurally-matching rounding block by index
        (see :class:`.utils.BlockRewriter` for the numbering and errors);
        `None` rewrites every one that verifies.
        """
        return UnfoldNegZero.apply_with_edits(
            func,
            where=where,
            eval_info=eval_info,
        ).result

    @staticmethod
    def apply_with_edits(
        func: FuncDef, *,
        where: int | Cursor | None = None,
        eval_info: PartialEvalInfo | None = None,
    ) -> EditLog:
        """:meth:`apply`, with an :class:`EditLog` of what it replaced."""
        if not isinstance(func, FuncDef):
            raise TypeError(f'Expected \'FuncDef\', got {func}')
        check_where(where)

        if eval_info is None:
            eval_info = PartialEval.apply(func)

        vtor = _UnfoldNegZeroInstance(func, eval_info, where)
        out = vtor.apply()
        vtor.check_site('a candidate rounding block')
        return EditLog(func, out, tuple(vtor.edits), exprs_preserved=True)
