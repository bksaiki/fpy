"""
Rescale a fixed-point context so that its digits land at position zero.

A fixed-point context represents the format ``A(inf, scale, maxval)``: every
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
under a fixed-point context with a non-zero scale becomes a scale-in, a
round under the integer context, and a scale-out, both scalings exact under
``REAL``.

Every fixed-point context shifts this way — :class:`fpy2.FixedContext` and
:class:`fpy2.SMFixedContext` by their ``scale``, :class:`fpy2.MPFixedContext`
and :class:`fpy2.MPBFixedContext` by their ``nmin``, which sits one position
below the scale.  A bounded format's ``maxval`` shifts with it, so the
integer range is unchanged.

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
    FixedContext,
    MPBFixedContext,
    MPFixedContext,
    RealFloat,
    SMFixedContext,
)
from ..number.format import (
    FixedFormat,
    Format,
    MPBFixedFormat,
    MPFixedFormat,
    SMFixedFormat,
)
from ..utils import Gensym

_FixedCtx = FixedContext | SMFixedContext | MPBFixedContext | MPFixedContext
"""the fixed-point contexts, whose formats shift under a power of two"""

_SCALE_ARG: dict[type, str] = {
    FixedContext: 'scale',
    SMFixedContext: 'scale',
    MPFixedContext: 'nmin',
}
"""
The constructor argument naming each context's digit position.

``MPBFixedContext`` is absent: its bounds are constructor arguments too, so
a rescaled call would have to rewrite those as well.
"""


def _pow2(k: int, loc: Location | None) -> Expr:
    """The constant `2 ** k`."""
    if k >= 0:
        return Integer(1 << k, loc)
    return Rational(None, 1, 1 << -k, loc)


def _scale_of(ctx: Context) -> int | None:
    """
    The position of `ctx`'s least significant digit, if it is fixed-point.

    ``nmin`` is the last *unrepresentable* position, one below the scale.
    """
    match ctx:
        case FixedContext() | SMFixedContext():
            return ctx.scale
        case MPBFixedContext() | MPFixedContext():
            return ctx.nmin + 1
        case _:
            return None


def _shift(x: RealFloat, k: int) -> RealFloat:
    """`x * 2**k`, exactly."""
    return RealFloat(s=x.s, exp=x.exp + k, c=x.c)


def _shift_format(fmt: Format, k: int) -> Format:
    """`fmt` scaled by `2**k`: every digit position moves, as does any bound."""
    match fmt:
        case FixedFormat():
            return FixedFormat(fmt.signed, fmt.scale + k, fmt.nbits)
        case SMFixedFormat():
            return SMFixedFormat(fmt.scale + k, fmt.nbits)
        case MPBFixedFormat():
            return MPBFixedFormat(
                fmt.nmin + k, _shift(fmt.pos_maxval, k), _shift(fmt.neg_maxval, k),
                fmt.enable_nan, fmt.enable_inf, fmt.enable_neg_zero,
            )
        case MPFixedFormat():
            return MPFixedFormat(
                fmt.nmin + k, fmt.enable_nan, fmt.enable_inf, fmt.enable_neg_zero,
            )
        case _:
            raise RuntimeError(f'unexpected fixed-point format {fmt}')


def _rescale(ctx: _FixedCtx, scale: int) -> _FixedCtx:
    """`ctx` with its format shifted to `scale`, all else equal."""
    k = scale - _scale_of(ctx)  # type: ignore[operator]
    fmt = _shift_format(ctx.format(), k)
    kwargs: dict = dict(rm=ctx.rm, num_randbits=ctx.num_randbits, rng=ctx.rng)
    if isinstance(ctx, MPBFixedContext):
        # only a bounded format can overflow
        kwargs['overflow'] = ctx.overflow
    return type(ctx).from_format(fmt, **kwargs)


def _rescale_expr(e: Expr, ctx: _FixedCtx, scale: int) -> Expr:
    """
    `e`, which evaluates to `ctx`, rewritten to evaluate at `scale`.

    A constructor call keeps its written form with only the digit-position
    argument replaced, so the rewritten program reads like the original.
    Any other expression is replaced by the rescaled context itself.
    """
    dst = _rescale(ctx, scale)
    name = _SCALE_ARG.get(type(ctx))
    if name is not None and isinstance(e, Call) and e.fn is type(ctx):
        pos = Integer(getattr(dst, name), e.loc)
        index = 1 if isinstance(ctx, FixedContext) else 0
        if len(e.args) > index:
            args = (*e.args[:index], pos, *e.args[index + 1:])
            return Call(e.func, e.fn, args, e.kwargs, e.loc)
        if any(n == name for n, _ in e.kwargs):
            kwargs = tuple((n, pos if n == name else v) for n, v in e.kwargs)
            return Call(e.func, e.fn, e.args, kwargs, e.loc)

    return ForeignVal(dst, e.loc)


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

    def _source_ctx(self, stmt: ContextStmt) -> _FixedCtx | None:
        """The block's context, if this block can be rescaled to position zero."""
        # a bound context is visible to the body as a value, which the rewrite changes
        if not isinstance(stmt.target, UnderscoreId):
            return None

        # the context must be statically known, fixed-point, and not already integral
        ctx = self.eval_info.by_expr.get(stmt.ctx)
        if not isinstance(ctx, _FixedCtx) or _scale_of(ctx) == 0:
            return None
        # a substituted NaN/Inf value would have to be shifted along with the format
        if ctx.nan_value is not None or ctx.inf_value is not None:
            return None

        # rounding commutes with the shift; arithmetic does not
        for s in stmt.body.stmts:
            match s:
                case Assign(target=NamedId()) | ReturnStmt():
                    if not isinstance(s.expr, (Round, Cast)) or not isinstance(s.expr.arg, Var):
                        return None
                case _:
                    return None

        return ctx

    def _rescale_round(
        self, e: Round | Cast, target: NamedId, loc: Location | None, scale: int
    ) -> list[Stmt]:
        """`target = round(v)` scaled in, rounded at position zero, and scaled out."""
        assert isinstance(e.arg, Var)

        # scale in: the operand's digits move up to position zero
        scaled = self.gensym.fresh('_t')
        up = Assign(scaled, None, Mul(_pow2(-scale, loc), e.arg, loc), loc)

        # round under the rescaled context
        rounded = self.gensym.fresh('_t')
        round_ = Assign(rounded, None, type(e)(e.func, Var(scaled, loc), loc), loc)

        # scale out: the result returns to its original magnitude, under the
        # original target name, so statements after the block are unaffected
        down = Assign(target, None, Mul(_pow2(scale, loc), Var(rounded, loc), loc), loc)

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

        scale = _scale_of(src)
        assert scale is not None

        stmts: list[Stmt] = []
        for s in stmt.body.stmts:
            assert isinstance(s.expr, (Round, Cast))
            if isinstance(s, Assign):
                assert isinstance(s.target, NamedId)
                stmts.extend(self._rescale_round(s.expr, s.target, s.loc, scale))
            else:
                # a returned round scales out into a temporary, then returns it
                assert isinstance(s, ReturnStmt)
                out = self.gensym.fresh('_t')
                stmts.extend(self._rescale_round(s.expr, out, s.loc, scale))
                stmts.append(ReturnStmt(Var(out, s.loc), s.loc))

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
