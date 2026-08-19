"""
Unfold a rounding context's special values into program text.

A format's answer for NaN, an infinity and a zero is a constant, since a special
operand is the only way a special reaches the rounding.  So each can be stated as
a branch on the operand instead:

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
           with C_:              # C, less any rule the branches took over
               r = fp.round(x)

That leaves the surviving rounding an operand that is finite *and* non-zero,
which is what a value-class analysis reads to discharge the guards below it.

**Stating a special and shedding its rule are separate.**  Stating one needs only
that the context be statically known: the branch assigns exactly what the
rounding would have returned, the format is untouched, and no check is required.
Shedding the rule from the format on top of that changes what the surviving
rounding does, so it is checked against the source over the values where the two
could disagree — and only a format that states the rule as a parameter
(``enable_nan``/``enable_inf``, ``nan_value``/``inf_value``) can do it at all.

The two come apart in both directions.  Dropping ``enable_inf`` from a format
whose overflow *produces* an infinity changes what finite operands past the bound
become, which the branches never see, so that rule stays while its branch is
still emitted.  And no *float* format states a rule this way — an encoded float
always has a NaN by construction — so a float context is stated and never shed.
A refusal is neither: a branch assigns a value and cannot refuse one, and leaving
the value to the rounding refuses it identically.

Which branches appear is decided per operand by
:class:`~fpy2.analysis.ValueClassInfer`: a class the operand cannot hold takes a
branch nothing reaches.  That is also what makes the rewrite idempotent — after
one pass the surviving operand is finite and non-zero, so a second pass states
nothing.  Stating a zero alone is not worth a rewrite, so a format that refuses
both specials is left unchanged.

Only a block whose body is entirely ``x = fp.round(v)`` or ``x = fp.cast(v)``
(or a returned round) over variables is rewritten.  A cast substitutes a special
exactly as a round does — the substitution happens before the exactness check —
so it takes the same branches.  Stochastic rounding takes them too: a special
never reaches the random draw, so the branches are deterministic and the
surviving context keeps its random bits; the agreement probes run with the
randomness turned off, where the two formats coincide.  ``REAL`` is declined:
it rounds exactly, so its specials pass through and the branches would say
nothing.

`SMFixedContext` and `FixedContext` state no NaN or infinity of their own, so
what they shed is a substituted *value* — which comes off in-class, keeping
the source's format and the written form of its constructor.
"""

from dataclasses import dataclass, replace
from typing import Any

from ..analysis import (
    PartialEval,
    PartialEvalInfo,
    ValueClass,
    ValueClassAnalysis,
    ValueClassInfer,
)
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
    Context,
    Float,
    MPBFixedContext,
    MPFixedContext,
    RealFloat,
)
from ..utils import CompareOp, Gensym
from .utils import (
    BlockRewriter,
    Declined,
    EditLog,
    agrees,
    check_site,
    check_where,
    fixed_probes,
    rounding_block,
    sign_choice,
    try_round,
)

_Shedable = MPFixedContext | MPBFixedContext
"""
the contexts that state their special values as parameters, and so can have one
removed from the format rather than only stated alongside it.

`SMFixedContext` and `FixedContext` derive from `MPBFixedContext`, so the second
member covers every bounded fixed-point format.  No float context is one.
"""

_Pair = tuple[Float, Float]
"""what a format makes of a special, as a `(positive, negative)` pair"""


@dataclass(frozen=True)
class _Source:
    """A format with stated special values, in the terms the rewrite needs."""

    ctx: Context
    """the source format"""
    dropped: Context
    """the same format with the shed rules removed -- `ctx` itself where nothing
    could be shed, in which case the block keeps its context verbatim"""
    nan: _Pair | None
    """what NaN becomes, or `None` where the format has no result for one"""
    inf: _Pair | None
    """what an infinity becomes, or `None` where the format has no result"""
    zero: _Pair
    """what each zero rounds to; stated so the surviving operand is non-zero"""
    shed: ValueClass
    """the sides whose rule `dropped` no longer states.  Their branch is what
    supplies the value, so it is emitted whatever the operand's class -- where a
    side is *not* shed, the format still answers and the branch is only a
    shortcut."""


def _special_pair(ctx: Context, positive: Float) -> _Pair | None:
    """What `ctx` makes of `positive` and its negative, or `None` where it
    refuses either — a branch can only assign a value, not refuse one."""
    pos = try_round(ctx, positive)
    neg = try_round(ctx, Float(x=positive, s=True))
    if pos is None or neg is None:
        return None
    return pos, neg


def _without_specials(ctx: _Shedable, shed: ValueClass) -> _Shedable | None:
    """`ctx` with the *shed* special-value rules removed, its class kept.
    `None` if the result will not construct."""
    # only the parameters that change are passed, so a subclass that fixes a
    # flag by construction (`SMFixedContext`, `FixedContext` state no NaN or
    # infinity) still sheds a substituted *value* in-class
    kwargs: dict[str, Any] = {}
    if ValueClass.NAN & shed:
        kwargs |= {'nan_value': None} | ({'enable_nan': False} if ctx.enable_nan else {})
    if ValueClass.INF & shed:
        kwargs |= {'inf_value': None} | ({'enable_inf': False} if ctx.enable_inf else {})
    try:
        return ctx.with_params(**kwargs)
    except (TypeError, ValueError):
        # a subclass whose `with_params` rejects the flag keywords cannot
        # shed that rule without changing class; decline instead
        return None


def _emitted(src: _Source, x: Float | RealFloat) -> Float | None:
    """What the generated code yields for `x`: the branches in emission order,
    then the rounding under the format with the rules removed.

    A special is modelled only where its rule was *shed*.  A side that stays in
    the format answers the same whether the branch or the rounding handles it --
    as the zero row always does -- so modelling it either way gives the same
    comparison, and leaving it out keeps the probe independent of anything the
    class analysis had to say.
    """
    isnan = isinstance(x, Float) and x.isnan
    isinf = isinstance(x, Float) and x.isinf
    if src.nan is not None and isnan and ValueClass.NAN & src.shed:
        return src.nan[1] if x.s else src.nan[0]
    if src.inf is not None and isinf and ValueClass.INF & src.shed:
        return src.inf[1] if x.s else src.inf[0]
    if not isnan and not isinf and x.is_zero():
        return src.zero[1] if x.s else src.zero[0]
    return try_round(src.dropped, x)


def _deterministic(ctx: _Shedable) -> _Shedable:
    """`ctx` with its randomness off, for probing.  The shed rules touch only
    NaN, the infinities, and zero, none of which reach the random draw — so
    agreement of the deterministic twins carries over."""
    if ctx.num_randbits == 0:
        return ctx
    return ctx.with_params(num_randbits=0)


def _describe(ctx: Context) -> _Source:
    """
    What `ctx` makes of each special, and as many of its stated rules shed from
    the format as the probes allow.

    The two jobs are separate.  **Hoisting** a special into a branch needs only
    that `ctx` be statically known, since the branch then assigns exactly what
    the rounding would have returned -- the format is untouched, so no probe is
    needed and every concrete context qualifies.  **Shedding** the rule from the
    format on top of that changes what the surviving rounding does, so it is
    checked against the source over the values where the two could disagree, and
    only a format that states the rule as a parameter can do it at all.
    """
    nan = _special_pair(ctx, Float(isnan=True))
    inf = _special_pair(ctx, Float(isinf=True))
    zero = _special_pair(ctx, Float(c=0))
    assert zero is not None  # a zero is always representable

    hoisted = _Source(ctx, ctx, nan=nan, inf=inf, zero=zero, shed=ValueClass(0))
    if not isinstance(ctx, _Shedable):
        return hoisted

    # the probes and the source's answers are the same for every attempt;
    # only the shed side varies
    det = _deterministic(ctx)
    probes = fixed_probes(ctx)
    want = [try_round(det, x) for x in probes]

    # most first, so a format that can lose both does
    for shed in (ValueClass.NAN | ValueClass.INF, ValueClass.NAN, ValueClass.INF):
        if (ValueClass.NAN & shed and nan is None
                or ValueClass.INF & shed and inf is None):
            continue        # a refusal has no value for the branch to take over
        dropped = _without_specials(ctx, shed)
        if dropped is None:
            continue
        src = replace(hoisted, dropped=dropped, shed=shed)
        probed = replace(src, dropped=_deterministic(dropped))
        if all(agrees(w, _emitted(probed, x)) for w, x in zip(want, probes)):
            return src
    return hoisted


def _hoisted(src: _Source, cls: ValueClass) -> ValueClass:
    """The sides to state as branches for an operand of class *cls*.

    A side the format has no value for cannot be stated at all -- a branch
    assigns a value, it cannot refuse one -- and leaving the value to fall
    through to the rounding refuses it identically.

    Otherwise a side is stated where the operand can *be* that kind of value, or
    where the format no longer states the rule and the branch is the only thing
    that can supply it.  Skipping a class the operand cannot hold is also what
    makes the rewrite idempotent: after one pass the surviving operand is finite
    and non-zero, so a second pass states nothing and declines.
    """
    out = ValueClass(0)
    for atom, pair in ((ValueClass.NAN, src.nan), (ValueClass.INF, src.inf),
                       (ValueClass.ZERO, src.zero)):
        if pair is not None and atom & (cls | src.shed):
            out |= atom
    return out


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
        if ValueClass.NAN & src.shed:
            shed |= {'enable_nan', 'nan_value'}
        if ValueClass.INF & src.shed:
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
    class_info: ValueClassAnalysis
    gensym: Gensym
    where: int | None
    site_idx: int

    def __init__(
        self, func: FuncDef, eval_info: PartialEvalInfo,
        class_info: ValueClassAnalysis, where: int | None = None,
    ):
        self.func = func
        self.eval_info = eval_info
        self.class_info = class_info
        self.gensym = Gensym(eval_info.def_use.names())
        self.where = where
        self.site_idx = 0

    def apply(self) -> FuncDef:
        return self._visit_function(self.func, None)

    def _candidate(self, stmt: ContextStmt) -> list[Var] | None:
        """A cast substitutes a special exactly as a round does: the
        substitution happens before the exactness check."""
        return rounding_block(stmt, casts=True)

    def _verify(self, stmt: ContextStmt, args: list[Var]) -> _Source | Declined:
        """The block's format, if any of its special values can be stated as
        a branch or shed from it."""
        # the branch values are the context's own answers, so it has to be known here
        ctx = self.eval_info.by_expr.get(stmt.ctx)
        if not isinstance(ctx, Context):
            return Declined(
                'the context is not statically known, so the branch values '
                'cannot be computed'
            )
        if ctx is REAL:
            return Declined('`REAL` rounds exactly; it has no special-value rules to state')

        # a zero rides along wherever the rewrite already happens, but stating
        # it alone buys nothing: the guards a class analysis discharges are about
        # the specials, and a format that refuses both has none to state
        src = _describe(ctx)
        specials = ValueClass.NAN | ValueClass.INF
        if src.shed or any(self._hoist(src, arg) & specials for arg in args):
            return src
        return Declined(
            'nothing to state: no special-value rule can be shed from the '
            'format and no operand can be a special value'
        )

    def _hoist(self, src: _Source, arg: Var) -> ValueClass:
        return _hoisted(src, self.class_info.classify(arg))

    def _unfold(
        self, e: Round | Cast, target: NamedId, loc: Location | None,
        src: _Source, ctx_expr: Expr,
    ) -> Stmt:
        """`target = round(v)` as branches on the operand's class plus a
        rounding that sees only a finite, non-zero value."""
        assert isinstance(e.arg, Var)
        name = e.arg.name
        hoist = self._hoist(src, e.arg)

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
        if ValueClass.ZERO & hoist:
            body = StmtBlock([IfStmt(
                Compare([CompareOp.EQ], [arg(), Integer(0, loc)], loc),
                assign(sign_choice(src.zero[0], src.zero[1], arg(), loc)),
                body, loc,
            )])
        for atom, test, pair in ((ValueClass.INF, IsInf, src.inf),
                                 (ValueClass.NAN, IsNan, src.nan)):
            if not (atom & hoist):
                continue
            assert pair is not None
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
        class_info: ValueClassAnalysis | None = None,
    ) -> FuncDef:
        """
        Takes the special-value rules out of every qualifying rounding
        context in `func`, stating each as a branch on the operand; the
        surviving rounding sees only a finite, non-zero value.

        `where` selects one structurally-matching rounding block by index
        (see :class:`.utils.BlockRewriter` for the numbering and errors);
        `None` rewrites every one that verifies.
        """
        return UnfoldSpecial.apply_with_edits(
            func,
            where=where,
            eval_info=eval_info,
            class_info=class_info,
        ).result

    @staticmethod
    def apply_with_edits(
        func: FuncDef, *,
        where: int | None = None,
        eval_info: PartialEvalInfo | None = None,
        class_info: ValueClassAnalysis | None = None,
    ) -> EditLog:
        """:meth:`apply`, with the record of what it replaced; the
        rewritten program is the log's `result`."""
        if not isinstance(func, FuncDef):
            raise TypeError(f'Expected \'FuncDef\', got {func}')
        check_where(where)

        if eval_info is None:
            eval_info = PartialEval.apply(func)
        if class_info is None:
            class_info = ValueClassInfer.analyze(func)

        vtor = _UnfoldSpecialInstance(func, eval_info, class_info, where)
        out = vtor.apply()
        check_site(where, vtor.site_idx, 'a candidate rounding block')
        return EditLog(func, out, tuple(vtor.edits))
