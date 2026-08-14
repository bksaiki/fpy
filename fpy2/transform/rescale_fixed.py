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
    IfStmt,
    Integer,
    IsInf,
    IsNan,
    Location,
    Mul,
    NamedId,
    Neg,
    Pow,
    Rational,
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
    Float,
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
from .utils import (
    BlockRewriter,
    check_where,
    number_literal,
    shift,
    sign_choice,
    value_literal,
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
    specials: dict[str, Float] = field(default_factory=dict)
    """
    what the format makes of ``'nan'``, ``'+inf'``, and ``'-inf'``; an absent
    key is left to the rounding, which rejects it as it did before
    """


def _defined_specials(ctx: _FixedCtx) -> dict[str, Float]:
    """
    What NaN and the infinities round to under `ctx`, for whichever of them it
    defines.  A fixed-point format commonly defines none of them.
    """
    out: dict[str, Float] = {}
    for name, v in (
        ('nan', Float(isnan=True)),
        ('+inf', Float(isinf=True)),
        ('-inf', Float(isinf=True, s=True)),
    ):
        try:
            out[name] = ctx.round(v)
        except ValueError:
            # undefined under this format: leave it to the rounding
            pass
    return out


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


def _without_nan(fmt: Format) -> Format:
    """`fmt` with NaN turned off; a format that never had it is unchanged."""
    match fmt:
        # both derive from `MPBFixedFormat`, and neither has a NaN to turn off
        case FixedFormat() | SMFixedFormat():
            return fmt
        case MPBFixedFormat():
            return MPBFixedFormat(
                fmt.nmin, fmt.pos_maxval, fmt.neg_maxval,
                False, fmt.enable_inf, fmt.enable_neg_zero,
            )
        case MPFixedFormat():
            return MPFixedFormat(
                fmt.nmin, False, fmt.enable_inf, fmt.enable_neg_zero,
            )
        case _:
            raise RuntimeError(f'unexpected fixed-point format {fmt}')


def _needs_nan(ctx: _FixedCtx) -> bool:
    """Whether a rounding under `ctx` can produce a NaN of its own."""
    # an overflow becomes an infinity, which the format may substitute a NaN for
    return ctx.inf_value is not None and ctx.inf_value.isnan


def _rescale(ctx: _FixedCtx, scale: int, *, drop_nan: bool = False) -> _FixedCtx:
    """`ctx` with its format shifted to `scale`, all else equal."""
    k = scale - _scale_of(ctx)
    fmt = _shift_format(ctx.format(), k)
    if drop_nan and not _needs_nan(ctx):
        # a NaN reaches the rounding only as its operand, and the caller has
        # taken that case
        fmt = _without_nan(fmt)
    # only a non-finite substitute gets this far, and it is scale-invariant
    kwargs: dict = {
        'rm': ctx.rm,
        'num_randbits': ctx.num_randbits,
        'rng': ctx.rng,
        'inf_value': ctx.inf_value,
        'nan_value': None if drop_nan else ctx.nan_value,
    }
    if isinstance(ctx, MPBFixedContext):
        # only a bounded format can overflow
        kwargs['overflow'] = ctx.overflow
    # the format follows the context's own type, which `type(ctx)` hides
    return type(ctx).from_format(cast(Any, fmt), **kwargs)


def _rescale_expr(
    e: Expr, ctx: _FixedCtx, scale: int, *, drop_nan: bool = False
) -> Expr:
    """
    `e`, which evaluates to `ctx`, rewritten to evaluate at `scale`.

    A constructor call keeps its written form with only the digit-position
    argument replaced, so the rewritten program reads like the original.
    Any other expression is replaced by the rescaled context itself.

    With `drop_nan`, NaN is left out: the caller has taken that value into a
    branch of its own, so nothing reaches the rounding as one.
    """
    dst = _rescale(ctx, scale, drop_nan=drop_nan)
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
            if drop_nan and not _needs_nan(ctx):
                rewritten = Call(
                    rewritten.func, rewritten.fn, rewritten.args,
                    tuple(
                        (n, v) for n, v in rewritten.kwargs
                        if n not in ('nan_value', 'enable_nan')
                    ),
                    rewritten.loc,
                )
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

    func: FuncDef
    eval_info: PartialEvalInfo
    gensym: Gensym
    where: int | None
    fold_specials: bool
    site_idx: int

    def __init__(
        self, func: FuncDef, eval_info: PartialEvalInfo,
        where: int | None = None, fold_specials: bool = False,
    ):
        self.func = func
        self.eval_info = eval_info
        self.gensym = Gensym(eval_info.def_use.names())
        self.where = where
        self.fold_specials = fold_specials
        # Counts *candidate* blocks (those the rewrite could rescale) in
        # visit order, outermost-first.  `where` selects one by this index.
        self.site_idx = 0

    def apply(self) -> FuncDef:
        return self._visit_function(self.func, None)

    def _candidate(self, stmt: ContextStmt) -> _Shift | None:
        """How to rescale this block, or `None` if it must be left alone."""
        # a bound context is visible to the body as a value, which the rewrite changes
        if not isinstance(stmt.target, UnderscoreId):
            return None

        # rounding commutes with the shift; arithmetic does not
        for s in stmt.body.stmts:
            match s:
                case Assign(target=NamedId()) | ReturnStmt():
                    if not isinstance(s.expr, (Round, Cast)) or not isinstance(s.expr.arg, Var):
                        return None
                case _:
                    return None

        ctx = self.eval_info.by_expr.get(stmt.ctx)
        if isinstance(ctx, _FixedCtx):
            # a known format shifts by a constant, so the factors fold away
            if _scale_of(ctx) == 0:
                return None
            specials = _defined_specials(ctx) if self.fold_specials else {}
            # a *finite* substitute for NaN or infinity is a value in the
            # format, so it would have to shift along with it; a non-finite
            # one is the same at every scale.  Folding takes the specials out
            # of the block, which leaves nothing for a substitute to do.
            if not specials and any(
                v is not None and not v.is_nar()
                for v in (ctx.nan_value, ctx.inf_value)
            ):
                return None
            scale = _scale_of(ctx)
            drop_nan = 'nan' in specials
            return _Shift(
                ctx=lambda: _rescale_expr(stmt.ctx, ctx, 0, drop_nan=drop_nan),
                up=lambda: _pow2(-scale, stmt.loc),
                down=lambda: _pow2(scale, stmt.loc),
                specials=specials,
            )

        if isinstance(stmt.ctx, Call):
            return self._symbolic_shift(stmt.ctx)
        return None

    def _symbolic_shift(self, e: Call) -> _Shift | None:
        """
        Rescaling a context whose position is only known at run time.

        The format still shifts by a power of two, so the factors become
        `2 ** scale` expressions rather than constants.  The bound comes along:
        a constructible source format states a bound on its own grid, so the
        shifted bound lands exactly on the integer grid.
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

            A bound stated as ``2 ** (k + c)`` -- what a caller writes when the
            bound follows the position by a fixed number of digits -- shifts to
            the constant ``2 ** c``.  Multiplying instead would leave two
            powers whose exponents cancel only at run time, which no analysis
            of the result can see.
            """
            if (isinstance(b, Pow) and b.func is None
                    and isinstance(b.first, Integer) and b.first.val == 2
                    and isinstance(b.second, Add)
                    and isinstance(b.second.second, Integer)
                    and b.second.first.is_equiv(Var(k, loc))):
                return Integer(1 << b.second.second.val, loc)
            return Mul(b, up(), loc)

        def build_ctx() -> Expr:
            """The position becomes the integer grid; a stated bound shifts
            with it, while an unstated one is derived and follows on its own."""
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

    def _fold_specials(
        self, e: Round | Cast, target: NamedId, loc: Location | None,
        shift: _Shift, rescaled: list[Stmt],
    ) -> Stmt:
        """
        `rescaled`, behind branches assigning the folded special values.

        Only the values the format defines get a branch; anything else falls
        through to the rounding, which treats it exactly as it did before.
        """
        assert isinstance(e.arg, Var)
        name = e.arg.name

        def arg() -> Var:
            return Var(name, loc)

        def assign(v: Float) -> StmtBlock:
            return StmtBlock([Assign(target, None, value_literal(v, loc), loc)])

        rest: StmtBlock = StmtBlock(rescaled)
        pos_inf, neg_inf = shift.specials.get('+inf'), shift.specials.get('-inf')
        if pos_inf is not None and neg_inf is not None:
            # an infinity keeps its sign, so which one it was still matters
            body = StmtBlock([Assign(
                target, None, sign_choice(pos_inf, neg_inf, arg(), loc), loc,
            )])
            rest = StmtBlock([IfStmt(IsInf(None, arg(), loc), body, rest, loc)])

        nan_v = shift.specials.get('nan')
        if nan_v is not None:
            rest = StmtBlock([IfStmt(IsNan(None, arg(), loc), assign(nan_v), rest, loc)])

        # the folded values are exact, and stay so whatever context encloses
        # this statement; the rounding inside sets its own
        return ContextStmt(UnderscoreId(), ForeignVal(REAL, loc), rest, loc)

    def _rewrite(self, stmt: ContextStmt, shift: _Shift) -> list[Stmt]:
        """The block, rescaled, after whatever its context expression needs."""
        if shift.specials:
            # each rounding gets branches of its own, since each has its own
            # operand to test, so the block splits into one per rounding
            folded: list[Stmt] = []
            for s in stmt.body.stmts:
                assert isinstance(s, (Assign, ReturnStmt))
                assert isinstance(s.expr, (Round, Cast))
                out = s.target if isinstance(s, Assign) else self.gensym.fresh('_t')
                assert isinstance(out, NamedId)
                one = ContextStmt(
                    stmt.target, shift.ctx(),
                    StmtBlock(self._rescale_round(s.expr, out, s.loc, shift)),
                    stmt.loc,
                )
                folded.append(self._fold_specials(s.expr, out, s.loc, shift, [one]))
                if isinstance(s, ReturnStmt):
                    folded.append(ReturnStmt(Var(out, s.loc), s.loc))
            return [*shift.preamble, *folded]

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
    def apply(
        func: FuncDef, *,
        where: int | None = None,
        fold_specials: bool = False,
        eval_info: PartialEvalInfo | None = None,
    ) -> FuncDef:
        """
        Rescales fixed-point rounding in `func` to digit position zero.

        `where` selects a single candidate block by index, in visit order
        (outermost-first); candidates are the blocks this pass could
        rescale.  If `None`, every candidate is rescaled.

        With `fold_specials`, a rounding is preceded by branches assigning
        what the format makes of NaN and the infinities, for whichever of
        them it defines.  That takes them out of the rounding, which is what
        lets a format that substitutes a finite value for them be rescaled
        at all.
        """
        if not isinstance(func, FuncDef):
            raise TypeError(f'Expected \'FuncDef\', got {func}')
        check_where(where)

        if eval_info is None:
            eval_info = PartialEval.apply(func)

        return _RescaleFixedInstance(func, eval_info, where, fold_specials).apply()
