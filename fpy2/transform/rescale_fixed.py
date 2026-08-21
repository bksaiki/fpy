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

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

from ..analysis import PartialEval, PartialEvalInfo
from ..ast.fpyast import (
    Add,
    Assign,
    Call,
    Cast,
    ConstInf,
    ConstNan,
    ContextStmt,
    Expr,
    ForeignVal,
    FuncDef,
    Integer,
    Location,
    Mul,
    NamedId,
    Neg,
    Pow,
    Rational,
    RationalVal,
    ReturnStmt,
    Round,
    Stmt,
    StmtBlock,
    Sub,
    UnderscoreId,
    Var,
)
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
from .cursor import Cursor, EditLog, StmtCursor, stmt_sites
from .utils import (
    BlockRewriter,
    Declined,
    check_where,
    is_rounding_block,
    number_literal,
    rounding_block,
    shift,
)

_FixedCtx = FixedContext | SMFixedContext | MPBFixedContext | MPFixedContext
"""the fixed-point contexts, whose formats shift under a power of two"""

@dataclass(frozen=True)
class _CtorArgs:
    """Where a fixed-point constructor takes the arguments this pass rewrites."""

    position: str
    """name of the argument giving the digit position"""
    index: int
    """its index, when written positionally"""
    from_nmin: bool
    """whether the position is `nmin`, one below the scale"""
    bound: tuple[tuple[str, int | None], ...] = ()
    """
    the bound arguments, which shift with the format, each with its index
    when written positionally; `None` for a keyword-only one
    """


_CTOR_ARGS: dict[type, _CtorArgs] = {
    FixedContext: _CtorArgs('scale', 1, False),
    SMFixedContext: _CtorArgs('scale', 0, False),
    MPFixedContext: _CtorArgs('nmin', 0, True),
    MPBFixedContext: _CtorArgs('nmin', 0, True, (('maxval', 1), ('neg_maxval', None))),
}
"""How each fixed-point constructor is written."""


@dataclass
class _Shift:
    """
    Moving a block's format to digit position zero.

    `up` and `down` build the scale factors; each call returns fresh nodes,
    since a block may hold several roundings.
    """

    ctx: Callable[[], Expr]
    """builds the rescaled context expression; one node per block"""
    up: Callable[[], Expr]
    """`2 ** -scale`, applied to each operand"""
    down: Callable[[], Expr]
    """`2 ** scale`, applied to each result"""
    preamble: list[Stmt] = field(default_factory=list)
    """statements the block's context expression depends on"""


def _pow2(k: int, loc: Location | None) -> Expr:
    """The constant `2 ** k`."""
    if k >= 0:
        return Integer(1 << k, loc)
    return Rational(None, 1, 1 << -k, loc)


def _scale_of(ctx: _FixedCtx) -> int:
    """
    The position of `ctx`'s least significant digit.

    ``nmin`` is the last *unrepresentable* position, one below the scale.
    """
    if isinstance(ctx, (FixedContext, SMFixedContext)):
        return ctx.scale
    return ctx.nmin + 1


def _shift_format(fmt: Format, k: int) -> Format:
    """`fmt` scaled by `2**k`: every digit position moves, as does any bound."""
    match fmt:
        case FixedFormat():
            return FixedFormat(fmt.signed, fmt.scale + k, fmt.nbits)
        case SMFixedFormat():
            return SMFixedFormat(fmt.scale + k, fmt.nbits)
        case MPBFixedFormat():
            return MPBFixedFormat(
                fmt.nmin + k, shift(fmt.pos_maxval, k), shift(fmt.neg_maxval, k),
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
    k = scale - _scale_of(ctx)
    fmt = _shift_format(ctx.format(), k)
    # only a non-finite substitute gets this far, and it is scale-invariant
    kwargs: dict = {
        'rm': ctx.rm,
        'num_randbits': ctx.num_randbits,
        'rng': ctx.rng,
        'inf_value': ctx.inf_value,
        'nan_value': ctx.nan_value,
    }
    if isinstance(ctx, MPBFixedContext):
        # only a bounded format can overflow
        kwargs['overflow'] = ctx.overflow
    # the format follows the context's own type, which `type(ctx)` hides
    return type(ctx).from_format(cast(Any, fmt), **kwargs)


def _rescale_expr(e: Expr, ctx: _FixedCtx, scale: int) -> Expr:
    """
    `e`, which evaluates to `ctx`, rewritten to evaluate at `scale`.

    A constructor call keeps its written form with only the digit-position
    argument replaced, so the rewritten program reads like the original.
    Any other expression is replaced by the rescaled context itself.
    """
    dst = _rescale(ctx, scale)
    info = _CTOR_ARGS.get(type(ctx))
    if info is not None and isinstance(e, Call) and e.fn is type(ctx):
        pos = Integer(getattr(dst, info.position), e.loc)
        rewritten = _replace_arg(e, info.position, info.index, _const(pos))
        if rewritten is not None:
            # a stated bound shifts with the position; an unstated one is
            # derived from it and follows on its own
            for name, index in info.bound:
                if _arg_of(rewritten, name, index) is None:
                    continue
                attr = 'pos_maxval' if name == 'maxval' else name
                scaled = number_literal(getattr(dst, attr), e.loc)
                bound = _replace_arg(rewritten, name, index, _const(scaled))
                assert bound is not None
                rewritten = bound
            return rewritten

    return ForeignVal(dst, e.loc)


def _replace_arg(
    e: Call, name: str, index: int | None, rewrite: Callable[[Expr], Expr]
) -> Call | None:
    """`e` with the argument named `name` rewritten, however it is written."""
    if index is not None and len(e.args) > index:
        args = (*e.args[:index], rewrite(e.args[index]), *e.args[index + 1:])
        return Call(e.func, e.fn, args, e.kwargs, e.loc)
    if any(n == name for n, _ in e.kwargs):
        kwargs = tuple((n, rewrite(v) if n == name else v) for n, v in e.kwargs)
        return Call(e.func, e.fn, e.args, kwargs, e.loc)
    return None


def _const(value: Expr) -> Callable[[Expr], Expr]:
    """A rewrite that puts `value` in place of whatever was there."""
    return lambda _: value


def _arg_of(e: Call, name: str, index: int | None) -> Expr | None:
    """The argument named `name`, however it is written."""
    if index is not None and len(e.args) > index:
        return e.args[index]
    for n, v in e.kwargs:
        if n == name:
            return v
    return None


class _RescaleFixedInstance(BlockRewriter):
    """Rewrites every qualifying context statement in a function."""

    _casts = True
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
        """Rounding commutes with the shift; arithmetic does not."""
        return rounding_block(stmt, casts=self._casts)

    def _verify(self, stmt: ContextStmt, args: list[Var]) -> _Shift | Declined:
        """How to rescale this block, or why it must be left alone."""
        ctx = self.eval_info.by_expr.get(stmt.ctx)
        if isinstance(ctx, _FixedCtx):
            # a known format shifts by a constant, so the factors fold away
            scale = _scale_of(ctx)
            if scale == 0:
                return Declined(
                    'the format is already at digit position zero; there is '
                    'nothing to rescale'
                )
            # a *finite* substitute for NaN or infinity is a value in the
            # format, so it would have to shift along with it; a non-finite
            # one is the same at every scale.  `UnfoldSpecial` takes the
            # substitutes out of the context, after which nothing is left
            # here to shift.
            if any(
                v is not None and not v.is_nar()
                for v in (ctx.nan_value, ctx.inf_value)
            ):
                return Declined(
                    'a finite NaN or infinity substitute is a value in the '
                    'format and would have to shift with it; run '
                    '`unfold_special` first'
                )
            return _Shift(
                ctx=lambda: _rescale_expr(stmt.ctx, ctx, 0),
                up=lambda: _pow2(-scale, stmt.loc),
                down=lambda: _pow2(scale, stmt.loc),
            )

        if isinstance(stmt.ctx, Call):
            sym = self._symbolic_shift(stmt.ctx)
            if sym is None:
                return Declined(
                    'the constructor call is not a fixed-point context whose '
                    'position can be shifted symbolically'
                )
            return sym
        return Declined(
            'the context is neither a statically-known fixed-point format '
            'nor a constructor call to shift symbolically'
        )

    def _symbolic_shift(self, e: Call) -> _Shift | None:
        """
        Rescaling a context whose position is only known at run time.

        The format still shifts by a power of two, so the factors become
        `2 ** scale` expressions rather than constants.  The bound comes along:
        a constructible source format states a bound that is one of its own
        values, so the shifted bound is an exact integer.
        """
        info = _CTOR_ARGS.get(e.fn if isinstance(e.fn, type) else type(None))
        if info is None:
            return None

        # as in the known case, a finite substitute for NaN or infinity would
        # have to shift too; here only a written-out one can be ruled out
        if any(
            name in ('nan_value', 'inf_value') and not isinstance(v, (ConstNan, ConstInf))
            for name, v in e.kwargs
        ):
            return None

        loc = e.loc
        position = _arg_of(e, info.position, info.index)
        if position is None:
            return None

        # the scale is one above `nmin`; `nmin` is usually written as the
        # scale minus one, in which case the two cancel
        scale = position
        if info.from_nmin:
            if (isinstance(position, Sub) and isinstance(position.second, Integer)
                    and position.second.val == 1):
                scale = position.first
            else:
                scale = Add(position, Integer(1, loc), loc)
        # a format already at position zero has nothing to shift
        if isinstance(scale, Integer) and scale.val == 0:
            return None

        # a scale already held in a variable needs no binding of its own
        preamble: list[Stmt] = []
        if isinstance(scale, Var):
            k = scale.name
        else:
            k = self.gensym.fresh('_k')
            preamble.append(ContextStmt(
                UnderscoreId(), ForeignVal(REAL, loc),
                StmtBlock([Assign(k, None, scale, loc)]), loc,
            ))

        up = lambda: Pow(None, Integer(2, loc), Neg(Var(k, loc), loc), loc)
        down = lambda: Pow(None, Integer(2, loc), Var(k, loc), loc)

        def shift_bound(b: Expr) -> Expr:
            """`b * 2 ** -k`, folded when the two cancel.

            A bound stated as ``2 ** (k + c)`` shifts to the constant
            ``2 ** c``.  Multiplying instead leaves two powers whose exponents
            cancel only at run time, which no analysis of the result can see.
            """
            if isinstance(b, Pow) and b.func is None \
                    and isinstance(b.first, RationalVal) \
                    and b.first.as_rational() == 2 \
                    and isinstance(b.second, Add):
                # `k + c` and `c + k` are the same sum
                for var, const in ((b.second.first, b.second.second),
                                   (b.second.second, b.second.first)):
                    if isinstance(const, Integer) and var.is_equiv(Var(k, loc)):
                        return _pow2(const.val, loc)
            return Mul(b, up(), loc)

        def build_ctx() -> Expr:
            """The position becomes zero, so the values are integers; a stated
            bound shifts with it, while an unstated one follows on its own."""
            built = _replace_arg(
                e, info.position, info.index,
                _const(Integer(-1 if info.from_nmin else 0, loc)),
            )
            assert built is not None
            for name, index in info.bound:
                if _arg_of(built, name, index) is None:
                    continue
                scaled = _replace_arg(built, name, index, shift_bound)
                assert scaled is not None
                built = scaled
            return built

        if _replace_arg(e, info.position, info.index, lambda x: x) is None:
            # the position is written some other way
            return None
        return _Shift(ctx=build_ctx, up=up, down=down, preamble=preamble)

    def _rescale_round(
        self, e: Round | Cast, target: NamedId, loc: Location | None, shift: _Shift
    ) -> list[Stmt]:
        """`target = round(v)` scaled in, rounded at position zero, and scaled out."""
        assert isinstance(e.arg, Var)

        # scale in: the operand's digits move up to position zero
        scaled = self.gensym.fresh('_t')
        up = Assign(scaled, None, Mul(shift.up(), e.arg, loc), loc)

        # round under the rescaled context
        rounded = self.gensym.fresh('_t')
        round_ = Assign(rounded, None, type(e)(e.func, Var(scaled, loc), loc), loc)

        # scale out: the result returns to its original magnitude, under the
        # original target name, so statements after the block are unaffected
        down = Assign(target, None, Mul(shift.down(), Var(rounded, loc), loc), loc)

        return [
            ContextStmt(UnderscoreId(), ForeignVal(REAL, loc), StmtBlock([up]), loc),
            round_,
            ContextStmt(UnderscoreId(), ForeignVal(REAL, loc), StmtBlock([down]), loc),
        ]

    def _rewrite(self, stmt: ContextStmt, shift: _Shift) -> list[Stmt]:
        """The block, rescaled, after whatever its context expression needs."""
        stmts: list[Stmt] = []
        for s in stmt.body.stmts:
            if isinstance(s, Assign):
                assert isinstance(s.expr, (Round, Cast)) and isinstance(s.target, NamedId)
                stmts.extend(self._rescale_round(s.expr, s.target, s.loc, shift))
            else:
                # a returned round scales out into a temporary, then returns it
                assert isinstance(s, ReturnStmt) and isinstance(s.expr, (Round, Cast))
                out = self.gensym.fresh('_t')
                stmts.extend(self._rescale_round(s.expr, out, s.loc, shift))
                stmts.append(ReturnStmt(Var(out, s.loc), s.loc))

        block: Stmt = ContextStmt(stmt.target, shift.ctx(), StmtBlock(stmts), stmt.loc)
        return [*shift.preamble, block]


class RescaleFixed:
    """
    Transformation pass to rescale fixed-point rounding to position zero.
    """

    @staticmethod
    def sites(func: FuncDef, within: Cursor | None = None) -> list[StmtCursor]:
        """The sites of `func`, in visit order -- what a `where` index counts,
        and what `within` narrows.

        Runs the same decisions the rewrite does, so a listing reports exactly
        the blocks `where=None` would rewrite: no candidate that this pass
        refuses appears here or consumes an index.
        """
        eval_info = PartialEval.apply(func)
        return _RescaleFixedInstance(func, eval_info).list_sites(within)

    @staticmethod
    def refusals(
        func: FuncDef, within: Cursor | None = None
    ) -> list[tuple[Cursor, str]]:
        """Why each rounding block of `func` that is not a site was refused,
        in visit order.  A refusal takes no index, so this is how one is found.
        """
        eval_info = PartialEval.apply(func)
        return _RescaleFixedInstance(func, eval_info).list_refusals(within)

    @staticmethod
    def apply(
        func: FuncDef, *,
        where: int | Cursor | None = None,
        eval_info: PartialEvalInfo | None = None,
    ) -> FuncDef:
        """
        Rescales fixed-point rounding in `func` to digit position zero.

        `where` selects one structurally-matching rounding block by index
        (see :class:`.utils.BlockRewriter` for the numbering and errors);
        `None` rewrites every one that verifies.

        A format that substitutes a *finite* value for NaN or an infinity is
        declined: the substitute would have to shift along with the format.
        Run :class:`fpy2.transform.UnfoldSpecial` first, which takes those
        rules out of the context.
        """
        return RescaleFixed.apply_with_edits(
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

        vtor = _RescaleFixedInstance(func, eval_info, where)
        out = vtor.apply()
        vtor.check_site('a candidate rounding block')
        return EditLog(func, out, tuple(vtor.edits), exprs_preserved=True)
