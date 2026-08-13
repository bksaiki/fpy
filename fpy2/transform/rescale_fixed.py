"""
Rescale a fixed-point context so that its digits land at position zero.

A ``FixedContext`` represents the format ``A(inf, scale, maxval)``: every
digit at or above position ``scale``, bounded by ``maxval``.  Scaling by a
power of two shifts the whole format,

.. math::

   2^k \\cdot A(\\infty, n, \\mathit{maxval})
       = A(\\infty, n + k, 2^k \\cdot \\mathit{maxval})

and rounding commutes with the shift, since multiplying by a power of two
is exact and order-preserving:

.. math::

   \\mathrm{round}_A(x) = 2^{-k} \\cdot \\mathrm{round}_{2^k A}(2^k \\cdot x)

Taking ``k = -scale`` lands the format at position zero, where its values
are integers.  This transform makes that identity structural: a rounding
under a ``FixedContext`` with a non-zero scale becomes a scale-in, a round
under the integer context, and a scale-out, both scalings exact under
``REAL``.

.. code-block:: python

   # Before
   with fp.FixedContext(True, -16, 32):     # A(inf, -16, MAX)
       aq = fp.round(a)

   # After
   with fp.FixedContext(True, 0, 32):       # A(inf, 0, MAX * 2**16)
       with fp.REAL:
           _t = 65536 * a
       _r = fp.round(_t)
       with fp.REAL:
           aq = fp.rational(1, 65536) * _r

The scale factors are emitted as constants, so :class:`fpy2.transform.ConstFold`
folds them into the surrounding expressions.

Only a block whose body is entirely ``x = fp.round(v)`` / ``x = fp.cast(v)``
over variables is rewritten.  Rounding commutes with the shift, but
arithmetic does not: a product of two shifted values is shifted by
``2**2k``, and an added constant would have to be shifted too.  Every other
block is left unchanged, including one that binds its context (``with C as
c:``), whose body could observe the rescaled context as a value.
"""

from ..analysis import PartialEval, PartialEvalInfo
from ..ast.fpyast import (
    Assign,
    Call,
    Cast,
    ContextStmt,
    Expr,
    ForeignVal,
    FuncDef,
    Integer,
    Location,
    Mul,
    NamedId,
    Rational,
    Round,
    Stmt,
    StmtBlock,
    UnderscoreId,
    Var,
)
from ..ast.visitor import DefaultTransformVisitor
from ..number import REAL, FixedContext
from ..number.context.fixed import FixedFormat
from ..utils import Gensym


def _pow2(k: int, loc: Location | None) -> Expr:
    """The constant `2 ** k`."""
    if k >= 0:
        return Integer(1 << k, loc)
    return Rational(None, 1, 1 << -k, loc)


def _rescale(ctx: FixedContext, scale: int) -> FixedContext:
    """`ctx` with its format shifted to `scale`, all else equal."""
    fmt = ctx.format()
    assert isinstance(fmt, FixedFormat)
    return FixedContext.from_format(
        FixedFormat(fmt.signed, scale, fmt.nbits),
        rm=ctx.rm, overflow=ctx.overflow,
        num_randbits=ctx.num_randbits, rng=ctx.rng,
    )


def _rescale_expr(e: Expr, ctx: FixedContext, scale: int) -> Expr:
    """
    `e`, which evaluates to `ctx`, rewritten to evaluate at `scale`.

    A ``FixedContext(...)`` call keeps its written form with only the scale
    argument replaced, so the rewritten program reads like the original.
    Any other expression is replaced by the rescaled context itself.
    """
    new_scale = Integer(scale, e.loc)
    if isinstance(e, Call) and e.fn is FixedContext:
        if len(e.args) >= 2:
            args = (e.args[0], new_scale, *e.args[2:])
            return Call(e.func, e.fn, args, e.kwargs, e.loc)
        if any(name == 'scale' for name, _ in e.kwargs):
            kwargs = tuple((n, new_scale if n == 'scale' else v) for n, v in e.kwargs)
            return Call(e.func, e.fn, e.args, kwargs, e.loc)

    return ForeignVal(_rescale(ctx, scale), e.loc)


class _RescaleFixedInstance(DefaultTransformVisitor):
    """Rewrites every qualifying context statement in a function."""

    func: FuncDef
    eval_info: PartialEvalInfo
    gensym: Gensym

    def __init__(self, func: FuncDef, eval_info: PartialEvalInfo):
        self.func = func
        self.eval_info = eval_info
        self.gensym = Gensym(eval_info.def_use.names())

    def apply(self) -> FuncDef:
        return self._visit_function(self.func, None)

    def _source_ctx(self, stmt: ContextStmt) -> FixedContext | None:
        """The block's context, if this block can be rescaled to position zero."""
        # a bound context is visible to the body as a value, which the rewrite changes
        if not isinstance(stmt.target, UnderscoreId):
            return None

        # the context must be statically known, fixed-point, and not already integral
        ctx = self.eval_info.by_expr.get(stmt.ctx)
        if not isinstance(ctx, FixedContext) or ctx.scale == 0:
            return None
        # a substituted NaN/Inf value would have to be shifted along with the format
        if ctx.nan_value is not None or ctx.inf_value is not None:
            return None

        # rounding commutes with the shift; arithmetic does not
        for s in stmt.body.stmts:
            if not isinstance(s, Assign) or not isinstance(s.target, NamedId):
                return None
            if not isinstance(s.expr, (Round, Cast)) or not isinstance(s.expr.arg, Var):
                return None

        return ctx

    def _rescale_round(self, stmt: Assign, scale: int) -> list[Stmt]:
        """`out = round(v)` scaled in, rounded at position zero, and scaled out."""
        assert isinstance(stmt.expr, (Round, Cast)) and isinstance(stmt.expr.arg, Var)
        loc = stmt.loc

        # scale in: the operand's digits move up to position zero
        scaled = self.gensym.fresh('_t')
        up = Assign(scaled, None, Mul(_pow2(-scale, loc), stmt.expr.arg, loc), loc)

        # round under the rescaled context
        rounded = self.gensym.fresh('_t')
        op = type(stmt.expr)(stmt.expr.func, Var(scaled, loc), loc)
        round_ = Assign(rounded, None, op, loc)

        # scale out: the result returns to its original magnitude, under the
        # original target name, so statements after the block are unaffected
        down = Assign(stmt.target, None, Mul(_pow2(scale, loc), Var(rounded, loc), loc), loc)

        real = ForeignVal(REAL, loc)
        return [
            ContextStmt(UnderscoreId(), real, StmtBlock([up]), loc),
            round_,
            ContextStmt(UnderscoreId(), real, StmtBlock([down]), loc),
        ]

    def _visit_context(self, stmt: ContextStmt, ctx: None):
        src = self._source_ctx(stmt)
        if src is None:
            return super()._visit_context(stmt, ctx)

        stmts: list[Stmt] = []
        for s in stmt.body.stmts:
            assert isinstance(s, Assign)
            stmts.extend(self._rescale_round(s, src.scale))

        dst = _rescale_expr(stmt.ctx, src, 0)
        return ContextStmt(stmt.target, dst, StmtBlock(stmts), stmt.loc), ctx


class RescaleFixed:
    """
    Transformation pass to rescale fixed-point rounding to position zero.
    """

    @staticmethod
    def apply(func: FuncDef, *, eval_info: PartialEvalInfo | None = None) -> FuncDef:
        if not isinstance(func, FuncDef):
            raise TypeError(f'Expected \'FuncDef\', got {func}')

        if eval_info is None:
            eval_info = PartialEval.apply(func)

        return _RescaleFixedInstance(func, eval_info).apply()
