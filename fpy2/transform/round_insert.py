"""
Rounding insertion: the inverse of :class:`.RoundElim`.

A correctly-rounded operation is a real computation followed by a rounding,
``f_F,rm = rnd_F,rm . f_R``.  Where format inference proves the real result
already lies in ``F``, the rounding does nothing, so the two are the same
function and either may replace the other.  :class:`.RoundElim` reads that
identity left to right, dropping a rounding that cannot be observed; this pass
reads it right to left, giving an exact operation a format.

The candidates are individual operations, and the rewrite mirrors
:class:`.RoundElim`'s hoist.  A context applies to every operation in its block,
so an operation is given a format of its own by being lifted into a block alone:
each operand that is not already a ``Var`` is bound to a fresh temporary under
the *original* scope, which both keeps whatever rounding the operand already did
and leaves the new block one operation to round.

.. code-block:: python

    # before, with FP32 arguments
    with fp.REAL:
        t = (x * x) + (y * y)

    # after, aimed at `x * x` with a target of FP64
    with fp.REAL:
        with fp.FP64:
            _t = (x * x)
        t = _t + (y * y)

Because the inserted rounding is verified an identity it changes no value, so
operations may be given formats one at a time and in any order -- including one
whose result a later exact operation reads.  Idempotence falls out: a second
pass finds only ``Var``-argumented operations already under a format.

The refusals, and why each one, are at :meth:`_RoundInsertInstance._verify`.
"""

from typing import Any

from ..analysis import ContextUse, DefineUse, SyntaxCheck
from ..analysis.format_infer import (
    AbstractableFormat,
    AbstractFormat,
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
    ForStmt,
    FuncDef,
    If1Stmt,
    IfExpr,
    IfStmt,
    ListComp,
    Mul,
    Neg,
    Round,
    StmtBlock,
    Sub,
    UnderscoreId,
    Var,
    WhileStmt,
)
from ..number import REAL, Context
from ..utils import Gensym
from .cursor import Cursor, EditLog, ExprCursor
from .utils import Declined, SiteRewriter, check_where, operands, rebuild

_ROUNDABLE = (Add, Sub, Mul, Abs, Neg, Round, Cast)
"""The operations that carry a context-driven rounding."""


class _RoundInsertInstance(SiteRewriter):
    """Gives selected exact operations in a function a format."""

    _expr_sited = True   # the candidates are roundable operations

    func: FuncDef
    ctx: Context
    scopes: 'ExactScopes'
    gensym: Gensym
    where: int | Cursor | None

    def __init__(
        self,
        func: FuncDef,
        ctx: Context,
        scopes: 'ExactScopes',
        where: int | Cursor | None = None,
    ):
        self.func = func
        self.ctx = ctx
        self.scopes = scopes
        self.gensym = Gensym(reserved=scopes.def_use.names())
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
        stored = self.scopes.format_info.by_expr.get(e)
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
        loc = e.loc
        args: list[Expr] = []
        for operand in operands(e):
            new = self._visit_expr(operand, out)
            if isinstance(new, Var):
                # a name lookup rounds nothing, so the bind would be a pure copy
                args.append(new)
                continue
            t = self.gensym.fresh('_t')
            out.append(Assign(t, None, new, loc))
            args.append(Var(t, loc))

        result = self.gensym.fresh('_t')
        block = StmtBlock([Assign(result, None, rebuild(e, args), loc)])
        out.append(ContextStmt(
            UnderscoreId(), ForeignVal(self.ctx, loc), block, loc,
        ))
        return Var(result, loc)

    def _visit_expr(self, e: Expr, ctx: Any) -> Expr:
        if not isinstance(e, _ROUNDABLE) or not self.scopes.is_exact(e):
            return super()._visit_expr(e, ctx)

        # a refusal is not a site, so it is decided before an index is spent:
        # `ctx` is `None` where no statement-level preamble reaches
        declined = (
            Declined(
                'the operation has no statement-level position for the block '
                'the rewrite emits'
            )
            if ctx is None
            else self._verify(e)
        )
        if declined is not None:
            self.refused.append((e, declined.reason))
            if self._named_by_cursor(e):
                # a cursor named it: say why, rather than that it named nothing
                self.declined.append(declined.reason)
            return super()._visit_expr(e, ctx)

        idx = self.site_idx
        self.site_idx += 1
        if not self._selects_expr(e, idx):
            return super()._visit_expr(e, ctx)

        self._matched += 1
        if self.listing:
            self.found_exprs.append(e)
            return super()._visit_expr(e, ctx)

        hoisted = self._hoist(e, ctx)
        self._replaced = True
        return hoisted

    # A compound statement's own sub-expression cannot carry a preamble.  For a
    # `while` condition that is soundness: the condition is re-evaluated every
    # iteration, and a preamble before the loop computes it once, which does not
    # terminate.  For the rest it is the edit log: `SiteRewriter._visit_block`
    # resets `_replaced` per statement, so a rewrite recorded while visiting the
    # sub-expression is lost once the nested block is visited, and every later
    # statement in the block mis-forwards.
    def _visit_if1(self, stmt: If1Stmt, ctx: Any):
        return super()._visit_if1(stmt, None)[0], ctx

    def _visit_if(self, stmt: IfStmt, ctx: Any):
        return super()._visit_if(stmt, None)[0], ctx

    def _visit_while(self, stmt: WhileStmt, ctx: Any):
        return super()._visit_while(stmt, None)[0], ctx

    def _visit_for(self, stmt: ForStmt, ctx: Any):
        return super()._visit_for(stmt, None)[0], ctx

    def _visit_context(self, stmt: ContextStmt, ctx: Any):
        return super()._visit_context(stmt, None)[0], ctx

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

    def __init__(self, func: FuncDef):
        self.def_use = DefineUse.analyze(func)
        self.ctx_use = ContextUse.analyze(func, def_use=self.def_use)
        self.format_info = FormatInfer.analyze(
            func, def_use=self.def_use, ctx_use=self.ctx_use,
        )
        # symbolic scopes resolve against the caller's pin, as they do for
        # `FormatInfer` itself; without one they stay unresolvable
        fn_fmt = self.format_info.fn_fmt
        self.outer = None if fn_fmt is None else fn_fmt.ctx

    def is_exact(self, e: Expr) -> bool:
        """Whether *e*'s active scope rounds exactly, so it has no rounding yet."""
        scope = self.ctx_use.find_scope_from_use(e)   # type: ignore[arg-type]
        ctx = scope.ctx if isinstance(scope.ctx, Context) else self.outer
        return ctx is REAL


class RoundInsert:
    """
    Transformation pass to give an exact operation a format, where the
    rounding is provably an identity.
    """

    @staticmethod
    def sites(
        func: FuncDef, within: Cursor | None = None, *, ctx: Context
    ) -> list[Cursor]:
        """The operations of `func` that would be given the format `ctx`, in
        visit order -- what a `where` index counts, and what `within` narrows.

        `ctx` is required because it decides the answer: whether an operation is
        a site is whether rounding *it* to *this* format is an identity.
        """
        if not isinstance(ctx, Context):
            raise TypeError(f'Expected a \'Context\', got {ctx}')
        scopes = ExactScopes(func)
        return _RoundInsertInstance(func, ctx, scopes).list_sites(within)

    @staticmethod
    def refusals(
        func: FuncDef, within: Cursor | None = None, *, ctx: Context
    ) -> list[tuple[Cursor, str]]:
        """Why each operation of `func` that is not a site for `ctx` was
        refused, in visit order.  A refusal takes no index, so this is how one
        is found.
        """
        if not isinstance(ctx, Context):
            raise TypeError(f'Expected a \'Context\', got {ctx}')
        return _RoundInsertInstance(func, ctx, ExactScopes(func)).list_refusals(within)

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

        scopes = ExactScopes(func)
        vtor = _RoundInsertInstance(func, ctx, scopes, where)
        out = vtor.apply()
        vtor.check_site('a candidate operation')
        SyntaxCheck.check(out, ignore_unknown=True)
        return EditLog(func, out, tuple(vtor.edits), exprs_preserved=True)
