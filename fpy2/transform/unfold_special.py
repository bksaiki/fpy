"""
Unfold a rounding context's special values into program text.

A fixed-point format states what NaN and the infinities become — a
representable value of their own (``enable_nan``/``enable_inf``), a
substituted constant (``nan_value``/``inf_value``), or a refusal.  Each rule
that names a value can be stated as a branch on the operand instead, since a
special operand is the only way a special reaches the rounding:

.. code-block:: python

   # Before
   with C:                       # nan_value=v, inf_value=w
       r = fp.round(x)

   # After
   with fp.REAL:
       if fp.isnan(x):
           r = v
       elif fp.isinf(x):
           r = w                 # sign-dependent where the format says so
       elif x == 0:
           r = -0.0 if fp.signbit(x) else 0
       else:
           with C_:              # C with the stated rules removed
               r = fp.round(x)

The zero branch removes nothing from the format — a zero is always
representable — but with it the surviving rounding's operand is finite *and*
non-zero, which is what a value-class analysis needs to discharge the
format's remaining guards.

The two sides come out independently: dropping ``enable_inf`` from a format
whose overflow *produces* an infinity changes what finite operands past the
bound become, which the branches never see — so that side stays in the format
and only the NaN side is stated.  Which sides can come out is checked against
the source rather than assumed, over the values where the rewrite could
disagree; a format where neither side survives the check is left unchanged.
A refusal also stays: a branch can only assign a value, not refuse one.

Only a block whose body is entirely ``x = fp.round(v)`` or ``x = fp.cast(v)``
(or a returned round) over variables is rewritten.  A cast substitutes a
special exactly as a round does — the substitution happens before the
exactness check — so it sheds the same rules.  Stochastic rounding sheds them
too: a special never reaches the random draw, so the branches are
deterministic and the surviving context keeps its random bits; the agreement
probes run with the randomness turned off, where the two formats coincide.

`SMFixedContext` and `FixedContext` state no NaN or infinity of their own, so
what they shed is a substituted *value* — which comes off in-class, keeping
the source's format and the written form of its constructor.
"""

from dataclasses import dataclass, replace
from typing import Any

from ..analysis import PartialEval, PartialEvalInfo
from ..ast.fpyast import (
    Assign,
    Call,
    Cast,
    Compare,
    ContextStmt,
    Expr,
    ForeignVal,
    FuncDef,
    IfStmt,
    Integer,
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
from ..ast.visitor import DefaultTransformVisitor
from ..number import (
    REAL,
    Float,
    MPBFixedContext,
    MPFixedContext,
    RealFloat,
)
from ..utils import CompareOp, Gensym
from .utils import (
    BlockRewriter,
    agrees,
    check_where,
    fixed_probes,
    sign_choice,
    try_round,
)

_FixedCtx = MPFixedContext | MPBFixedContext
"""
the fixed-point contexts that state their special values as parameters.

`SMFixedContext` and `FixedContext` derive from `MPBFixedContext`, so the
second member covers every bounded fixed-point format.
"""

_Pair = tuple[Float, Float]
"""what a format makes of a special, as a `(positive, negative)` pair"""


@dataclass(frozen=True)
class _Source:
    """A format with stated special values, in the terms the rewrite needs."""

    ctx: _FixedCtx
    """the source format"""
    dropped: _FixedCtx
    """the same format with the shed rules removed, which the block rounds
    under instead"""
    nan: _Pair | None
    """what NaN becomes, where that rule comes out of the format"""
    inf: _Pair | None
    """what an infinity becomes, where that rule comes out of the format"""
    zero: _Pair
    """what each zero rounds to; stated so the surviving operand is non-zero"""


def _special_pair(ctx: _FixedCtx, positive: Float) -> _Pair | None:
    """What `ctx` makes of `positive` and its negative, or `None` where it
    refuses either — a branch can only assign a value, not refuse one."""
    pos = try_round(ctx, positive)
    neg = try_round(ctx, Float(x=positive, s=True))
    if pos is None or neg is None:
        return None
    return pos, neg


def _without_specials(
    ctx: _FixedCtx, *, drop_nan: bool, drop_inf: bool
) -> _FixedCtx | None:
    """`ctx` with the selected special-value rules removed, its class kept.
    `None` if the result will not construct."""
    # only the parameters that change are passed, so a subclass that fixes a
    # flag by construction (`SMFixedContext`, `FixedContext` state no NaN or
    # infinity) still sheds a substituted *value* in-class
    kwargs: dict[str, Any] = {}
    if drop_nan:
        kwargs |= {'nan_value': None} | ({'enable_nan': False} if ctx.enable_nan else {})
    if drop_inf:
        kwargs |= {'inf_value': None} | ({'enable_inf': False} if ctx.enable_inf else {})
    try:
        return ctx.with_params(**kwargs)
    except (TypeError, ValueError):
        # a subclass whose `with_params` rejects the flag keywords cannot
        # shed that rule without changing class; decline instead
        return None


def _emitted(src: _Source, x: Float | RealFloat) -> Float | None:
    """What the generated code yields for `x`: the branches in emission
    order, then the rounding under the format with the rules removed."""
    isnan = isinstance(x, Float) and x.isnan
    isinf = isinstance(x, Float) and x.isinf
    if src.nan is not None and isnan:
        return src.nan[1] if x.s else src.nan[0]
    if src.inf is not None and isinf:
        return src.inf[1] if x.s else src.inf[0]
    if not isnan and not isinf and x.is_zero():
        return src.zero[1] if x.s else src.zero[0]
    return try_round(src.dropped, x)


def _deterministic(ctx: _FixedCtx) -> _FixedCtx:
    """`ctx` with its randomness off, for probing.  The shed rules touch only
    NaN, the infinities, and zero, none of which reach the random draw — so
    agreement of the deterministic twins carries over."""
    if ctx.num_randbits == 0:
        return ctx
    return ctx.with_params(num_randbits=0)


def _describe(ctx: _FixedCtx) -> _Source | None:
    """
    `ctx` with as many of its special-value rules shed as the probes allow,
    most first; `None` where neither side comes out.
    """
    nan = _special_pair(ctx, Float(isnan=True)) if (
        ctx.enable_nan or ctx.nan_value is not None) else None
    inf = _special_pair(ctx, Float(isinf=True)) if (
        ctx.enable_inf or ctx.inf_value is not None) else None
    zero = _special_pair(ctx, Float(c=0))
    assert zero is not None  # a zero is always representable

    # the probes and the source's answers are the same for every attempt;
    # only the dropped side varies
    det = _deterministic(ctx)
    probes = fixed_probes(ctx)
    want = [try_round(det, x) for x in probes]

    for drop_nan, drop_inf in ((True, True), (True, False), (False, True)):
        if drop_nan and nan is None or drop_inf and inf is None:
            continue
        dropped = _without_specials(ctx, drop_nan=drop_nan, drop_inf=drop_inf)
        if dropped is None:
            continue
        src = _Source(
            ctx, dropped,
            nan=nan if drop_nan else None,
            inf=inf if drop_inf else None,
            zero=zero,
        )
        probed = replace(src, dropped=_deterministic(dropped))
        if all(agrees(w, _emitted(probed, x)) for w, x in zip(want, probes)):
            return src
    return None


def _ctx_expr(e: Expr, src: _Source) -> Expr:
    """
    The dropped context as an expression, in fresh nodes.  A constructor call
    keeps its written form with only the shed rules removed, so the rewritten
    program reads like the original; anything else — the rebuilt context
    itself.
    """
    if (
        isinstance(e, Call) and e.fn is type(src.ctx)
        and type(src.dropped) is type(src.ctx)
    ):
        shed = set()
        if src.nan is not None:
            shed |= {'enable_nan', 'nan_value'}
        if src.inf is not None:
            shed |= {'enable_inf', 'inf_value'}
        # a structurally-fresh copy: each emitted block must occupy distinct
        # AST nodes, and the source expression stays in place under `where`
        call = DefaultTransformVisitor()._visit_expr(e, None)
        assert isinstance(call, Call)
        # the rules are keyword-only in both constructors, and their defaults
        # are exactly "no rule", so shedding one is dropping its keyword
        kwargs = tuple(kv for kv in call.kwargs if kv[0] not in shed)
        return Call(call.func, call.fn, call.args, kwargs, call.loc)
    return ForeignVal(src.dropped, e.loc)


class _UnfoldSpecialInstance(BlockRewriter):
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
        """The block's format, if a special-value rule can be taken out of
        its context."""
        # a bound context is visible to the body as a value, which the rewrite changes
        if not isinstance(stmt.target, UnderscoreId):
            return None

        ctx = self.eval_info.by_expr.get(stmt.ctx)
        if not isinstance(ctx, _FixedCtx):
            return None

        # a cast substitutes a special exactly as a round does: the
        # substitution happens before the exactness check
        for s in stmt.body.stmts:
            match s:
                case Assign(target=NamedId()) | ReturnStmt():
                    if not isinstance(s.expr, (Round, Cast)) or not isinstance(s.expr.arg, Var):
                        return None
                case _:
                    return None

        return _describe(ctx)

    def _unfold(
        self, e: Round | Cast, target: NamedId, loc: Location | None,
        src: _Source, ctx_expr: Expr,
    ) -> Stmt:
        """`target = round(v)` as branches on the operand's class plus a
        rounding that sees only a finite, non-zero value."""
        assert isinstance(e.arg, Var)
        name = e.arg.name

        def arg() -> Var:
            return Var(name, loc)

        def assign(v: Expr) -> StmtBlock:
            return StmtBlock([Assign(target, None, v, loc)])

        # the rounding, under the format the rules came out of
        body = StmtBlock([ContextStmt(
            UnderscoreId(), ctx_expr,
            StmtBlock([Assign(target, None, type(e)(e.func, arg(), loc), loc)]), loc,
        )])

        # a zero is a constant of the format, and taking it out leaves the
        # rounding a non-zero operand for an analysis to rely on
        body = StmtBlock([IfStmt(
            Compare([CompareOp.EQ], [arg(), Integer(0, loc)], loc),
            assign(sign_choice(src.zero[0], src.zero[1], arg(), loc)),
            body, loc,
        )])
        for test, pair in ((IsInf, src.inf), (IsNan, src.nan)):
            if pair is None:
                continue
            body = StmtBlock([IfStmt(
                test(None, arg(), loc),
                assign(sign_choice(pair[0], pair[1], arg(), loc)),
                body, loc,
            )])

        # the branches compare and assign constants, so they are exact
        # whatever context encloses this statement; the rounding sets its own
        return ContextStmt(UnderscoreId(), ForeignVal(REAL, loc), body, loc)

    def _rewrite(self, stmt: ContextStmt, src: _Source) -> list[Stmt]:
        """The block's rounds, with the stated rules taken out of the
        context.  Nothing rounds under the source context afterwards, so the
        block itself goes away."""
        stmts: list[Stmt] = []
        for s in stmt.body.stmts:
            # each emitted block gets its own context expression
            ctx_expr = _ctx_expr(stmt.ctx, src)
            if isinstance(s, Assign):
                assert isinstance(s.expr, (Round, Cast)) and isinstance(s.target, NamedId)
                stmts.append(self._unfold(s.expr, s.target, s.loc, src, ctx_expr))
            else:
                # a returned round lands in a temporary, which the return names
                assert isinstance(s, ReturnStmt) and isinstance(s.expr, (Round, Cast))
                out = self.gensym.fresh('t')
                stmts.append(self._unfold(s.expr, out, s.loc, src, ctx_expr))
                stmts.append(ReturnStmt(Var(out, s.loc), s.loc))
        return stmts


class UnfoldSpecial:
    """
    Transformation pass to state a context's special values as program text.
    """

    @staticmethod
    def apply(
        func: FuncDef, *,
        where: int | None = None,
        eval_info: PartialEvalInfo | None = None,
    ) -> FuncDef:
        """
        Takes the special-value rules out of every qualifying rounding
        context in `func`, stating each as a branch on the operand; the
        surviving rounding sees only a finite, non-zero value.

        `where` selects a single candidate block by index, in visit order
        (outermost-first); candidates are the blocks this pass could rewrite.
        If `None`, every candidate is rewritten.
        """
        if not isinstance(func, FuncDef):
            raise TypeError(f'Expected \'FuncDef\', got {func}')
        check_where(where)

        if eval_info is None:
            eval_info = PartialEval.apply(func)

        return _UnfoldSpecialInstance(func, eval_info, where).apply()
