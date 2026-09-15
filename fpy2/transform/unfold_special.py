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
rounding does, so it is allowed only where no *finite* operand depends on the
rule — a branch never covers one — and only a format that states the rule as a
parameter (``enable_nan``/``enable_inf``, ``nan_value``/``inf_value``) can do it
at all.  Rounding a finite value reaches a special in exactly one way, an
overflow, so that is the whole of the question.

The two come apart in both directions.  Dropping ``enable_inf`` from a format
whose overflow *produces* an infinity changes what finite operands past the bound
become, which the branches never see, so that rule stays while its branch is
still emitted — which is what keeps a bounded float from shedding its infinity
while it still sheds its NaN.  An *encoded* float sheds neither: it spells NaN as
a ``nan_kind`` rather than a flag, and an encoding always has a NaN by
construction.  A refusal is neither stating nor shedding: a branch assigns a
value and cannot refuse one, and leaving the value to the rounding refuses it
identically.

Which branches appear is decided per operand by
:class:`~fpy2.analysis.ValueClassInfer`: a class the operand cannot hold takes a
branch nothing reaches.  That is also what makes the rewrite idempotent — after
one pass the surviving operand is finite and non-zero, so a second pass states
nothing.  Stating a zero alone is not worth a rewrite, so a format that refuses
both specials is left unchanged.

A site is the rounding itself, wherever the active context is one this rewrite
can restate; see :class:`~fpy2.transform.utils.ScopedRoundingRewriter`.  A cast
is one too: it substitutes a special exactly as a round does, the substitution
happening before the exactness check.  So is a stochastic rounding — a special
never reaches the random draw, so the branches are deterministic and the
surviving context keeps its random bits.  ``REAL`` is declined: it rounds
exactly, so its specials pass through and the branches would say nothing.

`SMFixedContext` and `FixedContext` state no NaN or infinity of their own, so
what they shed is a substituted *value* — which comes off in-class, keeping
the source's format and the written form of its constructor.
"""

from dataclasses import dataclass, replace
from typing import Any

from ..analysis import (
    ValueClass,
    ValueClassAnalysis,
    ValueClassInfer,
)
from ..ast.fpyast import (
    Assign,
    BoolVal,
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
    Round,
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
    MPBFloatContext,
    MPFixedContext,
    MPFloatContext,
    MPSFloatContext,
    OverflowMode,
)
from ..utils import CompareOp, Gensym
from .cursor import Cursor, EditLog
from .utils import (
    Declined,
    RoundingScopes,
    ScopedRoundingRewriter,
    check_where,
    sign_choice,
    try_round,
)

_FloatShedable = MPFloatContext | MPSFloatContext | MPBFloatContext
"""
the float contexts that state their special values as parameters.

Their flags default to *enabled*, the opposite of the fixed-point ones, so
shedding a rule from one is written by adding a keyword rather than dropping it.
"""

_Shedable = MPFixedContext | MPBFixedContext | _FloatShedable
"""
the contexts that state their special values as parameters, and so can have one
removed from the format rather than only stated alongside it.

`SMFixedContext` and `FixedContext` derive from `MPBFixedContext`, so the second
member covers every bounded fixed-point format.  An *encoded* float --
`EFloatContext` and `IEEEContext` -- is not one: it spells NaN as a `nan_kind`
rather than a flag, and an encoding always has a NaN by construction.
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


def _shedable(ctx: _Shedable) -> ValueClass:
    """
    The rules no *finite* operand depends on, and so may be shed.

    A branch only ever covers a special operand, so a rule that a finite one can
    still reach has to stay behind for it.  Rounding a finite value produces a
    special in exactly one way: an overflow.
    """
    both = ValueClass.NAN | ValueClass.INF
    if not isinstance(ctx, MPBFloatContext | MPBFixedContext):
        return both     # no bound, so nothing to overflow
    if ctx.overflow is not OverflowMode.OVERFLOW:
        return both     # saturating and wrapping stay finite; asserting raises
    if not any(ctx._overflow_to_infinity(s) for s in (False, True)):
        return both     # the overflow saturates whichever way it rounds
    # a finite overflow lands wherever an infinite operand does, so that rule
    # stays -- unless the format refuses it, which shedding leaves refused
    if ctx.enable_inf or ctx.inf_value is not None:
        return ValueClass.NAN
    return both


def _describe(ctx: Context) -> _Source:
    """
    What `ctx` makes of each special, and as many of its stated rules shed from
    the format as `_shedable` allows.

    The two jobs are separate.  **Hoisting** a special into a branch needs only
    that `ctx` be statically known, since the branch then assigns exactly what
    the rounding would have returned -- the format is untouched, so nothing has
    to be checked and every concrete context qualifies.  **Shedding** the rule from the
    format on top of that changes what the surviving rounding does, so it is
    allowed only where `_shedable` finds no finite operand depending on it, and
    only a format that states the rule as a parameter can do it at all.
    """
    nan = _special_pair(ctx, Float(isnan=True))
    inf = _special_pair(ctx, Float(isinf=True))
    zero = _special_pair(ctx, Float(c=0))
    assert zero is not None  # a zero is always representable

    hoisted = _Source(ctx, ctx, nan=nan, inf=inf, zero=zero, shed=ValueClass(0))
    if not isinstance(ctx, _Shedable):
        return hoisted

    safe = _shedable(ctx)

    # most first, so a format that can lose both does
    for shed in (ValueClass.NAN | ValueClass.INF, ValueClass.NAN, ValueClass.INF):
        if shed & ~safe:
            continue        # a finite operand still depends on this rule
        if (ValueClass.NAN & shed and nan is None
                or ValueClass.INF & shed and inf is None):
            continue        # a refusal has no value for the branch to take over
        dropped = _without_specials(ctx, shed)
        if dropped is None:
            continue
        return replace(hoisted, dropped=dropped, shed=shed)
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


def _ctx_expr(e: Expr | None, src: _Source, loc: Location | None) -> Expr:
    """
    The dropped context as an expression, in fresh nodes.  A constructor call
    keeps its written form with only the shed rules removed, so the rewritten
    program reads like the original; anything else — including a scope that is
    the function's own annotation, which states no expression in the body —
    the rebuilt context itself.
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
        # the rules are keyword-only in every constructor, so shedding one
        # starts by dropping its keyword
        kwargs = tuple(kv for kv in call.kwargs if kv[0] not in shed)
        if isinstance(src.ctx, _FloatShedable):
            # ... but a float constructor defaults its flags to *enabled*, so
            # dropping alone would leave the rule in place: say so outright
            kwargs += tuple(
                (name, BoolVal(False, e.loc))
                for name in ('enable_nan', 'enable_inf') if name in shed
            )
        return Call(call.func, call.fn, call.args, kwargs, call.loc)
    return ForeignVal(src.dropped, loc)


class _UnfoldSpecialInstance(ScopedRoundingRewriter):
    """States the special values of every qualifying rounding in a function."""

    _casts = True
    """a cast substitutes a special exactly as a round does: the substitution
    happens before the exactness check"""

    func: FuncDef
    scopes: RoundingScopes
    class_info: ValueClassAnalysis
    gensym: Gensym
    where: int | Cursor | None
    site_idx: int

    def __init__(
        self, func: FuncDef, scopes: RoundingScopes,
        class_info: ValueClassAnalysis, where: int | Cursor | None = None,
    ):
        self.func = func
        self.scopes = scopes
        self.class_info = class_info
        self.gensym = Gensym(scopes.def_use.names())
        self.where = where

    def apply(self) -> FuncDef:
        return self._visit_function(self.func, None)

    def _verify(self, e: Expr, ctx: Context | None) -> _Source | Declined:
        """The rounding's format, if any of its special values can be stated
        as a branch or shed from it."""
        # the branch values are the context's own answers, so it has to be known here
        if ctx is None:
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
        assert isinstance(e, (Round, Cast))
        if src.shed or self._hoist(src, e.arg) & specials:
            return src
        return Declined(
            'nothing to state: no special-value rule can be shed from the '
            'format and no operand can be a special value'
        )

    def _hoist(self, src: _Source, arg: Expr) -> ValueClass:
        return _hoisted(src, self.class_info.classify(arg))

    def _ladder(self, e: Expr, target: NamedId, out: list, src: _Source) -> None:
        """`target = round(v)` as branches on the operand's class plus a
        rounding that sees only a finite, non-zero value."""
        assert isinstance(e, (Round, Cast))
        loc = e.loc
        # the operand as written: a name the bind may mint has no class
        hoist = self._hoist(src, e.arg)
        ctx_expr = _ctx_expr(self.scopes.scope_ctx_expr(e), src, loc)
        name = self._arg_name(e, out)

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
        out.append(ContextStmt(UnderscoreId(), ForeignVal(REAL, loc), body, loc))


class UnfoldSpecial:
    """
    Transformation pass to state a context's special values as program text.
    """

    @staticmethod
    def sites(func: FuncDef, within: Cursor | None = None) -> list[Cursor]:
        """The sites of `func`, in visit order -- what a `where` index counts,
        and what `within` narrows.

        Runs the same decisions the rewrite does, so a listing reports exactly
        the roundings `where=None` would rewrite: no candidate that this pass
        refuses appears here or consumes an index.
        """
        class_info = ValueClassInfer.analyze(func)
        return _UnfoldSpecialInstance(
            func, RoundingScopes(func), class_info,
        ).list_sites(within)

    @staticmethod
    def refusals(
        func: FuncDef, within: Cursor | None = None
    ) -> list[tuple[Cursor, str]]:
        """Why each rounding of `func` that is not a site was refused, in visit
        order.  A refusal takes no index, so this is how one is found.
        """
        class_info = ValueClassInfer.analyze(func)
        return _UnfoldSpecialInstance(
            func, RoundingScopes(func), class_info,
        ).list_refusals(within)

    @staticmethod
    def apply(
        func: FuncDef, *,
        where: int | Cursor | None = None,
        class_info: ValueClassAnalysis | None = None,
    ) -> FuncDef:
        """
        Takes the special-value rules out of every qualifying rounding
        context in `func`, stating each as a branch on the operand; the
        surviving rounding sees only a finite, non-zero value.

        `where` selects one rounding by index (see
        :class:`.utils.ScopedRoundingRewriter` for the numbering and errors);
        `None` rewrites every one that verifies.
        """
        return UnfoldSpecial.apply_with_edits(
            func,
            where=where,
            class_info=class_info,
        ).result

    @staticmethod
    def apply_with_edits(
        func: FuncDef, *,
        where: int | Cursor | None = None,
        class_info: ValueClassAnalysis | None = None,
    ) -> EditLog:
        """:meth:`apply`, with an :class:`EditLog` of what it replaced."""
        if not isinstance(func, FuncDef):
            raise TypeError(f'Expected \'FuncDef\', got {func}')
        check_where(where)

        if class_info is None:
            class_info = ValueClassInfer.analyze(func)

        vtor = _UnfoldSpecialInstance(func, RoundingScopes(func), class_info, where)
        out = vtor.apply()
        vtor.check_site('a candidate rounding')
        return EditLog(func, out, tuple(vtor.edits), exprs_preserved=True)
