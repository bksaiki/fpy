"""
Rounding insertion: the inverse of :class:`.RoundElim`.

A correctly-rounded operation is a real computation followed by a rounding,
``f_F,rm = rnd_F,rm . f_R``.  Where format inference proves the real result
already lies in ``F``, the rounding does nothing, so the two are the same
function and either may replace the other.  :class:`.RoundElim` reads that
identity left to right, dropping a rounding that cannot be observed;
this pass reads it right to left, giving an exact operation a format.

The point is not to add work.  An operation under ``fp.REAL`` names no format,
so it cannot be lowered onto an environment's arithmetic; the same operation
under ``fp.FP64`` can.  Inserting the rounding is what makes a real-valued
specification implementable.

.. code-block:: python

    # before
    with fp.REAL:
        _t = (x * y)

    # after, given FP32 arguments and a target of FP64
    with fp.FP64:
        _t = (x * y)

The product of two FP32 values needs 48 digits, which FP64's 53 hold exactly,
so the inserted rounding is the identity and the rewrite changes no result.
Targeting FP32 instead is declined: 48 digits do not fit in 24.

Declined, with a reason, when the block's context is not ``REAL`` (there is no
rounding to insert), when the block holds more than one assignment (rounding the
first changes the second's operand, which needs a sequential analysis this pass
does not do), when the target is stochastic, when format inference cannot bound
the operation, and when the bound is not contained in the target format --
including containment of the special values, which the magnitude conditions
cannot see.
"""

from ..analysis import (
    ContextUse,
    ContextUseAnalysis,
    DefineUse,
    DefineUseAnalysis,
    PartialEval,
    PartialEvalInfo,
)
from ..analysis.format_infer import (
    AbstractableFormat,
    AbstractFormat,
    FormatAnalysis,
    FormatInfer,
    SetFormat,
    round_is_identity,
)
from ..ast.fpyast import Assign, ContextStmt, ForeignVal, FuncDef, Stmt
from ..number import REAL, Context
from .cursor import Cursor, EditLog, StmtCursor, stmt_sites
from .utils import BlockRewriter, Declined, check_where, exact_block, is_exact_block


class _RoundInsertInstance(BlockRewriter):
    """Gives every qualifying exact block in a function a format."""

    func: FuncDef
    ctx: Context
    eval_info: PartialEvalInfo
    format_info: FormatAnalysis
    where: int | Cursor | None

    def __init__(
        self,
        func: FuncDef,
        ctx: Context,
        eval_info: PartialEvalInfo,
        format_info: FormatAnalysis,
        where: int | Cursor | None = None,
    ):
        self.func = func
        self.ctx = ctx
        self.eval_info = eval_info
        self.format_info = format_info
        self.where = where

    def apply(self) -> FuncDef:
        return self._visit_function(self.func, None)

    def _candidate(self, stmt: ContextStmt) -> list[Assign] | None:
        """An exact block: whether its context is `REAL` is for `_verify`."""
        return exact_block(stmt)

    def _verify(self, stmt: ContextStmt, assigns: list[Assign]) -> Assign | Declined:
        """The assignment whose operation gains a format, or why it cannot."""
        if self.eval_info.by_expr.get(stmt.ctx) is not REAL:
            return Declined(
                'the block does not round exactly, so there is no rounding to '
                'insert; only a `fp.REAL` block is a target'
            )
        if len(assigns) > 1:
            return Declined(
                'the block holds more than one assignment, so rounding the first '
                'would change the operand of the next; split the block first'
            )
        if self.ctx.is_stochastic():
            return Declined(
                'the target rounds stochastically, so it is not an identity on a '
                'value it represents'
            )

        assign = assigns[0]
        stored = self.format_info.by_expr.get(assign.expr)
        # a stored bound may be a `Format`; `round_is_identity` wants the lift
        bound = (
            AbstractFormat.from_format(stored)
            if isinstance(stored, AbstractableFormat)
            else stored
        )
        if not isinstance(bound, (AbstractFormat, SetFormat)):
            return Declined(
                'format inference could not bound the operation, so the inserted '
                'rounding cannot be proven an identity'
            )
        if not round_is_identity(bound, self.ctx):
            return Declined(
                'the operation is not representable in the target format, so '
                'rounding to it would change the result'
            )
        # containment compares magnitudes; a special the target lacks is invisible there
        target_fmt = self.ctx.format()
        if (
            isinstance(bound, AbstractFormat)
            and isinstance(target_fmt, AbstractableFormat)
            and not bound.specials_contained_in(AbstractFormat.from_format(target_fmt))
        ):
            return Declined(
                'the operation can produce a special value the target format does '
                'not represent'
            )
        return assign

    def _rewrite(self, stmt: ContextStmt, assign: Assign) -> list[Stmt]:
        """The same body, under the target format."""
        return [
            ContextStmt(
                stmt.target, ForeignVal(self.ctx, stmt.loc), stmt.body, stmt.loc,
            )
        ]


class RoundInsert:
    """
    Transformation pass to give an exact operation a format, where the
    rounding is provably an identity.
    """

    @staticmethod
    def sites(func: FuncDef, within: Cursor | None = None) -> list[StmtCursor]:
        """The candidate exact blocks of `func`, in visit order -- what a
        `where` index counts, whether or not each verifies.
        """
        return stmt_sites(func, is_exact_block, within)

    @staticmethod
    def apply(
        func: FuncDef,
        ctx: Context,
        *,
        where: int | Cursor | None = None,
        def_use: DefineUseAnalysis | None = None,
        ctx_use: ContextUseAnalysis | None = None,
        eval_info: PartialEvalInfo | None = None,
        format_info: FormatAnalysis | None = None,
    ) -> FuncDef:
        """
        Rounds every qualifying exact block of `func` to `ctx`, where the
        rounding is provably an identity.

        `where` selects one structurally-matching exact block by index
        (see :class:`.utils.BlockRewriter` for the numbering and errors);
        `None` rewrites every one that verifies.
        """
        return RoundInsert.apply_with_edits(
            func,
            ctx,
            where=where,
            def_use=def_use,
            ctx_use=ctx_use,
            eval_info=eval_info,
            format_info=format_info,
        ).result

    @staticmethod
    def apply_with_edits(
        func: FuncDef,
        ctx: Context,
        *,
        where: int | Cursor | None = None,
        def_use: DefineUseAnalysis | None = None,
        ctx_use: ContextUseAnalysis | None = None,
        eval_info: PartialEvalInfo | None = None,
        format_info: FormatAnalysis | None = None,
    ) -> EditLog:
        """:meth:`apply`, with an :class:`EditLog` of what it replaced."""
        if not isinstance(func, FuncDef):
            raise TypeError(f'Expected \'FuncDef\', got {func}')
        if not isinstance(ctx, Context):
            raise TypeError(f'Expected a \'Context\', got {ctx}')
        check_where(where)

        if eval_info is None:
            eval_info = PartialEval.apply(func)
        if format_info is None:
            if def_use is None:
                def_use = DefineUse.analyze(func)
            if ctx_use is None:
                ctx_use = ContextUse.analyze(func, def_use=def_use)
            format_info = FormatInfer.analyze(func, def_use=def_use, ctx_use=ctx_use)

        vtor = _RoundInsertInstance(func, ctx, eval_info, format_info, where)
        out = vtor.apply()
        vtor.check_site('a candidate exact block')
        return EditLog(func, out, tuple(vtor.edits), exprs_preserved=True)
