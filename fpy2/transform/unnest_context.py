"""Unnesting context statements."""

from ..analysis import SyntaxCheck
from ..ast import *


class _Unnester(DefaultTransformVisitor):
    """
    Context statement unnester.
    """

    changed: bool

    def __init__(self):
        self.changed = False

    def _visit_block(self, block: StmtBlock, ctx: None):
        # a context statement may rewrite to several siblings
        stmts: list[Stmt] = []
        for stmt in block.stmts:
            s, _ = self._visit_statement(stmt, ctx)
            if isinstance(s, StmtBlock):
                stmts.extend(s.stmts)
            else:
                stmts.append(s)
        return StmtBlock(stmts), ctx

    def _visit_context(self, stmt: ContextStmt, ctx: None):
        body, _ = self._visit_block(stmt.body, ctx)
        stmts = list(body.stmts)

        # Peel every trailing `with`.  Stopping at one statement leaves
        # `with e1: (with e2: B)` alone: `DeadCodeEliminate` drops the outer
        # block outright, which is better than making siblings of the two.
        hoisted: list[Stmt] = []
        while len(stmts) > 1 and isinstance(stmts[-1], ContextStmt):
            hoisted.append(stmts.pop())

        outer = ContextStmt(stmt.target, stmt.ctx, StmtBlock(stmts), stmt.loc)
        if not hoisted:
            return outer, ctx

        self.changed = True
        return StmtBlock([outer, *reversed(hoisted)]), ctx

    def apply(self, func: FuncDef) -> tuple[FuncDef, bool]:
        return self._visit_function(func, None), self.changed


class UnnestContext:
    """Unnesting of context statements.

    A `with` block nested directly inside another does not refine it: entering
    one replaces the active context outright rather than merging with the
    context in force, so the parent contributes nothing to the nested block.
    Where that block sits at the *end* of its parent's body, the two can be
    written as siblings::

        with e1:                    with e1:
            X                           X
            with e2:        ->      with e2:
                B                       B

    which costs no statements and drops a level of nesting.  Nothing else
    moves: every statement stays under the same context, in the same order.

    A nested block that is its parent's *only* statement is left alone, since
    :class:`DeadCodeEliminate` drops the parent outright instead.

    This pass is for legibility.  Nothing downstream needs the flattened form
    -- `ContextUse` and `FormatInfer` key on scopes rather than on nesting.
    """

    @staticmethod
    def apply(func: FuncDef) -> FuncDef:
        """Unnest the context statements of *func*."""
        func, _ = UnnestContext.apply_with_status(func)
        return func

    @staticmethod
    def apply_with_status(func: FuncDef) -> tuple[FuncDef, bool]:
        """Same as :meth:`apply` but also returns a ``changed`` flag
        — ``True`` iff at least one block was unnested."""
        if not isinstance(func, FuncDef):
            raise TypeError(f'Expected `FuncDef`, got {type(func)} for {func}')
        func, changed = _Unnester().apply(func)
        SyntaxCheck.check(func)
        return func, changed
