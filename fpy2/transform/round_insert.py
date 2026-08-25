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
    NamedId,
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
from .utils import (
    Declined,
    RoundingRewriter,
    RoundingScopes,
    SiteRewriter,
    check_where,
    operands,
    rebuild,
)

_ROUNDABLE = (Add, Sub, Mul, Abs, Neg, Round, Cast)
"""The operations that carry a context-driven rounding."""


class _RoundInsertInstance(RoundingRewriter):
    """Gives selected exact operations in a function a format."""

    def _candidate(self, e: Expr) -> bool:
        # an operation whose scope already rounds is not a candidate at all, so
        # the listing carries no sites that always refuse
        return isinstance(e, _ROUNDABLE) and self.scopes.is_exact(e)


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

    def _wrap(self, t: NamedId, loc) -> Expr:
        # the block's own rounding is the whole rewrite; the site just reads it
        return Var(t, loc)

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
        scopes = RoundingScopes(func)
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
        return _RoundInsertInstance(func, ctx, RoundingScopes(func)).list_refusals(within)

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

        scopes = RoundingScopes(func)
        vtor = _RoundInsertInstance(func, ctx, scopes, where)
        out = vtor.apply()
        vtor.check_site('a candidate operation')
        SyntaxCheck.check(out, ignore_unknown=True)
        return EditLog(func, out, tuple(vtor.edits), exprs_preserved=True)
