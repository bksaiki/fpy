"""
Rounding insertion: the inverse of :class:`.RoundElim`.

A correctly-rounded operation is a real computation followed by a rounding,
``f_F,rm = rnd_F,rm . f_R``.  Where format inference proves the real result
already lies in ``F``, the rounding does nothing, so the two are the same
function and either may replace the other.  :class:`.RoundElim` reads that
identity left to right, dropping a rounding that cannot be observed; this pass
reads it right to left, giving an exact operation a format.

The point is not to add work.  An operation under ``fp.REAL`` names no format,
so no environment's arithmetic implements it; the same operation under
``fp.FP64`` is a hardware multiply.  Inserting the rounding is what makes a
real-valued specification implementable.

The candidates are *operations*, not blocks, and the rewrite mirrors
:meth:`.RoundElim._hoist` exactly.  A context applies to every operation in its
block, so an operation can only be given a format of its own by being alone in
one.  Each operand that is not already a ``Var`` is therefore bound to a fresh
temporary under the *original* scope, and the operation itself is emitted alone
under the target:

.. code-block:: python

    # before, with FP32 arguments
    with fp.REAL:
        t = (x * x) + (y * y)

    # after, aimed at `x * x` with a target of FP64
    with fp.REAL:
        with fp.FP64:
            _t = (x * x)
        t = _t + (y * y)

The new block holds one ``Var``-argumented operation, so exactly one rounding is
inserted and the rest of the statement stays exact.  Because the inserted
rounding is verified to be an *identity*, it changes no value: a later operation
reading the temporary sees what it would have seen, so operations may be given
formats one at a time and in any order.

The bind preserves whatever the operand already did -- an operand that is itself
a rounded operation fires at its original scope before the new block sees the
value -- and idempotence falls out, since a second pass finds only
``Var``-argumented operations already under a format.

Declined, with a reason, when the operation's scope does not round exactly
(there is no rounding to insert), when the target is stochastic, when format
inference cannot bound the operation, and when the bound is not contained in the
target format -- including containment of the special values, which the
magnitude conditions cannot see.
"""

from typing import Any

from ..analysis import (
    ContextUse,
    ContextUseAnalysis,
    ContextUseSite,
    DefineUse,
    DefineUseAnalysis,
)
from ..analysis.format_infer import (
    AbstractableFormat,
    AbstractFormat,
    FormatAnalysis,
    FormatInfer,
    SetFormat,
    round_is_identity,
)
from ..ast.fpyast import (
    Abs,
    Add,
    Assign,
    Cast,
    ContextStmt,
    Expr,
    ForeignVal,
    FuncDef,
    IfExpr,
    ListComp,
    Location,
    Mul,
    Neg,
    Round,
    StmtBlock,
    Sub,
    UnderscoreId,
    Var,
)
from ..number import REAL, Context
from ..utils import Gensym
from .cursor import Cursor, EditLog, ExprCursor, expr_sites
from .error import TransformDeclined
from .utils import Declined, SiteRewriter, check_where

_ROUNDABLE = (Add, Sub, Mul, Abs, Neg, Round, Cast)
"""The operations that carry a context-driven rounding.

The set :class:`.RoundElim` eliminates, so the two operators mirror each
other's sites.
"""


def _operands(e: Expr) -> list[Expr]:
    """The operands of a roundable operation."""
    match e:
        case Add() | Sub() | Mul():
            return [e.first, e.second]
        case Abs() | Neg() | Round() | Cast():
            return [e.arg]
        case _:
            raise RuntimeError(f'not a roundable operation: {e}')


def _rebuild(e: Expr, operands: list[Expr]) -> Expr:
    """*e* with its operands replaced."""
    match e:
        case Add() | Sub() | Mul():
            return type(e)(operands[0], operands[1], e.loc)
        case Abs() | Neg():
            return type(e)(operands[0], e.loc)
        case Round() | Cast():
            return type(e)(e.func, operands[0], e.loc)
        case _:
            raise RuntimeError(f'not a roundable operation: {e}')


class _RoundInsertInstance(SiteRewriter):
    """Gives selected exact operations in a function a format."""

    _expr_sited = True   # the candidates are roundable operations

    func: FuncDef
    ctx: Context
    scopes: 'ExactScopes'
    format_info: FormatAnalysis
    gensym: Gensym
    where: int | Cursor | None

    def __init__(
        self,
        func: FuncDef,
        ctx: Context,
        scopes: 'ExactScopes',
        format_info: FormatAnalysis,
        def_use: DefineUseAnalysis,
        where: int | Cursor | None = None,
    ):
        self.func = func
        self.ctx = ctx
        self.scopes = scopes
        self.format_info = format_info
        self.gensym = Gensym(reserved=def_use.names())
        self.where = where

    def apply(self) -> FuncDef:
        return self._visit_function(self.func, None)

    def _verify(self, e: Expr) -> None | Declined:
        """`None` where *e* may be given the target format, else why not."""
        if self.ctx.is_stochastic():
            return Declined(
                'the target rounds stochastically, so it is not an identity on '
                'a value it represents'
            )
        stored = self.format_info.by_expr.get(e)
        # a stored bound may be a `Format`; `round_is_identity` wants the lift
        bound = (
            AbstractFormat.from_format(stored)
            if isinstance(stored, AbstractableFormat)
            else stored
        )
        if not isinstance(bound, (AbstractFormat, SetFormat)):
            return Declined(
                'format inference could not bound the operation, so the '
                'inserted rounding cannot be proven an identity'
            )
        if not round_is_identity(bound, self.ctx):
            return Declined(
                'the operation is not representable in the target format, so '
                'rounding to it would change the result'
            )
        # containment compares magnitudes; a special the target lacks is invisible there
        target = self.ctx.format()
        if (
            isinstance(bound, AbstractFormat)
            and isinstance(target, AbstractableFormat)
            and not bound.specials_contained_in(AbstractFormat.from_format(target))
        ):
            return Declined(
                'the operation can produce a special value the target format '
                'does not represent'
            )
        return None

    def _hoist(self, e: Expr, out: list) -> Expr:
        """Compute *e* alone under the target and return a `Var` for its site.

        The mirror of :meth:`.RoundElim._hoist`: each operand that survives the
        visit as a non-``Var`` is bound under the *original* scope first, so the
        emitted block rounds this operation and nothing else.
        """
        loc: Location | None = e.loc
        operands: list[Expr] = []
        for operand in _operands(e):
            new = self._visit_expr(operand, out)
            if isinstance(new, Var):
                # a name lookup rounds nothing, so the bind would be a pure copy
                operands.append(new)
                continue
            t = self.gensym.fresh('_t')
            out.append(Assign(t, None, new, loc))
            operands.append(Var(t, loc))

        result = self.gensym.fresh('_t')
        block = StmtBlock([Assign(result, None, _rebuild(e, operands), loc)])
        out.append(ContextStmt(
            UnderscoreId(), ForeignVal(self.ctx, loc), block, loc,
        ))
        return Var(result, loc)

    def _visit_expr(self, e: Expr, ctx: Any) -> Expr:
        if not isinstance(e, _ROUNDABLE) or not self.scopes.is_exact(e):
            return super()._visit_expr(e, ctx)

        idx = self.site_idx
        self.site_idx += 1
        # `ctx` is `None` in positions a statement-level preamble cannot reach
        if ctx is None or not self._selects_expr(e, idx):
            return super()._visit_expr(e, ctx)

        self._matched += 1
        declined = self._verify(e)
        if declined is not None:
            self.declined.append(declined.reason)
            if isinstance(self.where, int):
                raise TransformDeclined(f'where={idx}: {declined.reason}')
            return super()._visit_expr(e, ctx)

        hoisted = self._hoist(e, ctx)
        self._replaced = True
        return hoisted

    def _visit_list_comp(self, e: ListComp, ctx: Any) -> ListComp:
        # the element sees the loop targets and later iterables see earlier
        # ones, so no statement-level preamble reaches inside a comprehension
        targets = [self._visit_binding(t, ctx) for t in e.targets]
        iterables = [self._visit_expr(i, None) for i in e.iterables]
        elt = self._visit_expr(e.elt, None)
        return ListComp(targets, iterables, elt, e.loc)

    def _visit_if_expr(self, e: IfExpr, ctx: Any) -> IfExpr:
        # the condition is evaluated unconditionally; the branches are not, so
        # hoisting one of them out would evaluate it either way
        cond = self._visit_expr(e.cond, ctx)
        ift = self._visit_expr(e.ift, None)
        iff = self._visit_expr(e.iff, None)
        return IfExpr(cond, ift, iff, e.loc)


class ExactScopes:
    """Which operations of a function sit under a scope that rounds exactly.

    Shared by :meth:`RoundInsert.sites` and the rewrite so the two agree on
    what a `where` index counts.
    """

    def __init__(
        self,
        func: FuncDef,
        ctx_use: ContextUseAnalysis,
        format_info: FormatAnalysis,
    ):
        self.ctx_use = ctx_use
        # symbolic scopes resolve against the caller's pin, as they do for
        # `FormatInfer` itself; without one they stay unresolvable
        self.outer = None if format_info.fn_fmt is None else format_info.fn_fmt.ctx

    def _resolved(self, e: ContextUseSite) -> Context | None:
        scope = self.ctx_use.find_scope_from_use(e)
        if isinstance(scope.ctx, Context):
            return scope.ctx
        return self.outer

    def is_exact(self, e: Expr) -> bool:
        """Whether *e*'s active scope rounds exactly, so it has no rounding yet."""
        try:
            return self._resolved(e) is REAL   # type: ignore[arg-type]
        except KeyError:
            # a node built without going through scope analysis: not a site
            return False


class RoundInsert:
    """
    Transformation pass to give an exact operation a format, where the
    rounding is provably an identity.
    """

    @staticmethod
    def sites(func: FuncDef, within: Cursor | None = None) -> list[ExprCursor]:
        """The candidate operations of `func`, in visit order -- what a `where`
        index counts, whether or not each verifies.

        A roundable operation whose scope rounds exactly; an operation that
        already has a format is not a candidate.
        """
        scopes = _scopes_of(func)
        return expr_sites(
            func,
            lambda e: isinstance(e, _ROUNDABLE) and scopes.is_exact(e),
            within,
        )

    @staticmethod
    def apply(
        func: FuncDef,
        ctx: Context,
        *,
        where: int | Cursor | None = None,
    ) -> FuncDef:
        """
        Gives every qualifying exact operation of `func` the format `ctx`,
        where the rounding is provably an identity.

        `where` selects one candidate operation by index in visit order;
        `None` rewrites every one that verifies.
        """
        return RoundInsert.apply_with_edits(func, ctx, where=where).result

    @staticmethod
    def apply_with_edits(
        func: FuncDef,
        ctx: Context,
        *,
        where: int | Cursor | None = None,
    ) -> EditLog:
        """:meth:`apply`, with an :class:`EditLog` of what it replaced."""
        if not isinstance(func, FuncDef):
            raise TypeError(f'Expected \'FuncDef\', got {func}')
        if not isinstance(ctx, Context):
            raise TypeError(f'Expected a \'Context\', got {ctx}')
        check_where(where)

        def_use = DefineUse.analyze(func)
        ctx_use = ContextUse.analyze(func, def_use=def_use)
        format_info = FormatInfer.analyze(func, def_use=def_use, ctx_use=ctx_use)
        scopes = ExactScopes(func, ctx_use, format_info)

        vtor = _RoundInsertInstance(
            func, ctx, scopes, format_info, def_use, where,
        )
        out = vtor.apply()
        vtor.check_site('a candidate operation')
        return EditLog(func, out, tuple(vtor.edits))


def _scopes_of(func: FuncDef) -> ExactScopes:
    """The scope map of `func`, for a listing that runs on its own."""
    def_use = DefineUse.analyze(func)
    ctx_use = ContextUse.analyze(func, def_use=def_use)
    format_info = FormatInfer.analyze(func, def_use=def_use, ctx_use=ctx_use)
    return ExactScopes(func, ctx_use, format_info)
