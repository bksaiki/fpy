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
candidates: splitting a rounding is the inverse of merging two, and admitting
them makes a second application grow the tree twice as fast.
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
    NamedId,
    Neg,
    Round,
    StmtBlock,
    Sub,
    UnderscoreId,
    Var,
    WhileStmt,
)
from ..number import REAL, Context, RealFloat
from ..utils import Gensym
from .cursor import Cursor, EditLog
from .utils import (
    Declined,
    RoundingRewriter,
    RoundingScopes,
    SiteRewriter,
    check_where,
    operands,
    rebuild,
)

_SPLITTABLE = (Add, Sub, Mul, Abs, Neg)
""":class:`.RoundInsert`'s set, minus the explicit roundings."""


class _SplitRoundInstance(RoundingRewriter):
    """Splits selected rounded operations through an intermediate."""

    def _candidate(self, e: Expr) -> bool:
        return isinstance(e, _SPLITTABLE)


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

        # The premises constrain what is representable, not what happens
        # beyond it, so a bounded intermediate has an overflow they do not
        # cover -- and no single behaviour suits every target.  See gap 4 of
        # `docs/todos/rounding-axes.md`; `derive_intermediate` sidesteps it.
        if isinstance(AbstractFormat.from_format(f2).bound, RealFloat):
            return Declined(
                'the intermediate has a finite range, so it would overflow '
                'where the single rounding did not; use an unbounded one'
            )
        return None

    def _wrap(self, t: NamedId, loc) -> Expr:
        # an assignment rounds nothing in FPy, so the second rounding is
        # explicit -- and sits in the enclosing block, which applies `rm1`
        return Round(None, Var(t, loc), loc)

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
