"""
Rounding splitting: one correctly-rounded operation into two roundings.

A correctly-rounded operation is a real computation followed by a rounding,
``f_F1,rm1 = rnd_F1,rm1 . f_R``.  Where the pair of formats and modes satisfies
Figure 8 of *When Double Rounding is Correct*, that single rounding equals a
rounding to an intermediate followed by a rounding to the target, so either may
replace the other.  This pass reads that left to right, computing under the
intermediate and re-rounding to the target.

The candidates are the operations that *already* have a format, which is the
complement of :class:`.RoundInsert`'s.  An assignment does not round in FPy, so
the second rounding is emitted as an explicit ``round`` in the enclosing block,
where it picks up the target's own mode.

.. code-block:: python

    # before                       # after, split through an RTO intermediate
    with fp.FP32:                  with fp.FP32:
        t = x * y                      with <intermediate>:
                                           _t = x * y
                                       t = round(_t)

Which pairs are admissible is decided by
:func:`fpy2.analysis.format_infer.double_round_ok`; the intermediate is the
caller's, and :func:`fpy2.analysis.format_infer.derive_intermediate` computes a
suitable one.  Explicit ``Round`` / ``Cast`` nodes are deliberately not
candidates: splitting a rounding is :class:`.RoundMerge`'s inverse, and
admitting them makes a second application grow the tree twice as fast.

The refusals, and why each one, are at :meth:`_SplitRoundInstance._verify`.
"""

from typing import Any

from ..analysis import SyntaxCheck
from ..analysis.format_infer import (
    AbstractableFormat,
    AbstractFormat,
    double_round_ok,
)
from ..ast.fpyast import (
    Abs,
    Add,
    Assign,
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
from .cursor import Cursor, EditLog
from .utils import (
    Declined,
    RoundingScopes,
    SiteRewriter,
    check_where,
    operands,
    rebuild,
)

_SPLITTABLE = (Add, Sub, Mul, Abs, Neg)
"""The operations whose rounding may be split.

:class:`.RoundInsert`'s set minus ``Round`` and ``Cast``: an explicit rounding
is not an arithmetic operation with a rounding attached, and splitting one is
the inverse rewrite rather than this one.
"""


class _SplitRoundInstance(SiteRewriter):
    """Splits selected rounded operations through an intermediate."""

    _expr_sited = True   # the candidates are rounded operations

    func: FuncDef
    ctx: Context
    scopes: RoundingScopes
    gensym: Gensym
    where: int | Cursor | None

    def __init__(
        self,
        func: FuncDef,
        ctx: Context,
        scopes: RoundingScopes,
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
        """`None` where *e*'s rounding may be split, else why not."""
        target = self.scopes.scope_ctx(e)
        if target is None:
            return Declined(
                'the operation\'s scope is symbolic, so the rounding it '
                'performs is unknown'
            )
        if target is REAL:
            return Declined(
                'the operation rounds exactly, so there is no rounding to '
                'split; `insert_round` gives it one first'
            )
        if target.is_stochastic() or self.ctx.is_stochastic():
            return Declined(
                'a stochastic rounding is not a function of its input, so the '
                'composition cannot be checked'
            )

        rm1, rm2 = target.rounding_mode(), self.ctx.rounding_mode()
        if rm1 is None or rm2 is None:
            return Declined('a context without a rounding mode has no rule')

        f1, f2 = target.format(), self.ctx.format()
        if not isinstance(f1, AbstractableFormat) or not isinstance(f2, AbstractableFormat):
            return Declined(
                'one of the formats has no abstract form, so the premise '
                'cannot be checked'
            )
        if not double_round_ok(
            AbstractFormat.from_format(f1), rm1,
            AbstractFormat.from_format(f2), rm2,
        ):
            return Declined(
                f'rounding to {rm2.name} and then {rm1.name} is not the same '
                f'as rounding to {rm1.name} for these formats'
            )
        return None

    def _split(self, e: Expr, out: list) -> Expr:
        """Compute *e* under the intermediate; return the re-rounding of it.

        The operands are bound exactly as :meth:`.RoundInsert._hoist` binds
        them, so the emitted block computes this operation and nothing else.
        The returned ``round`` sits in the *enclosing* block, which is what
        applies the target's rounding.
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

        inner = self.gensym.fresh('_t')
        block = StmtBlock([Assign(inner, None, rebuild(e, args), loc)])
        out.append(ContextStmt(
            UnderscoreId(), ForeignVal(self.ctx, loc), block, loc,
        ))
        return Round(None, Var(inner, loc), loc)

    def _visit_expr(self, e: Expr, ctx: Any) -> Expr:
        if not isinstance(e, _SPLITTABLE):
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

        split = self._split(e, ctx)
        self._replaced = True
        return split


    # A compound statement's own sub-expression carries no preamble, exactly as
    # for `RoundInsert`.  For a `while` condition that is soundness: the
    # condition is re-evaluated every iteration, and a preamble before the loop
    # computes it once, which does not terminate.  The rest is scope.
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


class SplitRound:
    """
    Transformation pass to split one rounding into two that compose to the
    same answer.
    """

    @staticmethod
    def sites(
        func: FuncDef, within: Cursor | None = None, *, ctx: Context
    ) -> list[Cursor]:
        """The operations of `func` whose rounding would be split through
        `ctx`, in visit order -- what a `where` index counts, and what `within`
        narrows.

        `ctx` is required because it decides the answer: whether an operation is
        a site is whether *its* rounding composes with a rounding to *this*
        intermediate.
        """
        if not isinstance(ctx, Context):
            raise TypeError(f'Expected a \'Context\', got {ctx}')
        scopes = RoundingScopes(func)
        return _SplitRoundInstance(func, ctx, scopes).list_sites(within)

    @staticmethod
    def refusals(
        func: FuncDef, within: Cursor | None = None, *, ctx: Context
    ) -> list[tuple[Cursor, str]]:
        """Why each operation of `func` that is not a site was left alone."""
        if not isinstance(ctx, Context):
            raise TypeError(f'Expected a \'Context\', got {ctx}')
        scopes = RoundingScopes(func)
        return _SplitRoundInstance(func, ctx, scopes).list_refusals(within)

    @staticmethod
    def apply(
        func: FuncDef, ctx: Context, *, where: int | Cursor | None = None
    ) -> FuncDef:
        """
        Splits every rounding of `func` that composes with a rounding to `ctx`,
        and leaves the rest alone.

        `where` selects one operation by index in visit order; `None` takes
        every one it can split.
        """
        return SplitRound.apply_with_edits(func, ctx, where=where).result

    @staticmethod
    def apply_with_edits(
        func: FuncDef, ctx: Context, *, where: int | Cursor | None = None
    ) -> EditLog:
        """:meth:`apply`, with an :class:`EditLog` of what it replaced."""
        if not isinstance(func, FuncDef):
            raise TypeError(f'Expected \'FuncDef\', got {func}')
        if not isinstance(ctx, Context):
            raise TypeError(f'Expected a \'Context\', got {ctx}')
        check_where(where)

        scopes = RoundingScopes(func)
        vtor = _SplitRoundInstance(func, ctx, scopes, where)
        out = vtor.apply()
        vtor.check_site('a rounded operation')
        SyntaxCheck.check(out, ignore_unknown=True)
        return EditLog(func, out, tuple(vtor.edits), exprs_preserved=True)
