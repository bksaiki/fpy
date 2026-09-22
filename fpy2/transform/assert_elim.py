"""Assertion elimination."""

from ..analysis import SyntaxCheck
from ..ast import *

__all__ = ['AssertElim']


class _Eliminator(DefaultTransformVisitor):
    """Drops every `assert`, and any statement left with an empty body."""

    def _visit_block(self, block: StmtBlock, ctx):
        stmts: list[Stmt] = []
        for stmt in block.stmts:
            if isinstance(stmt, AssertStmt):
                continue
            s, ctx = self._visit_statement(stmt, ctx)
            # a statement whose body held only asserts has nothing left to do
            if s is not None:
                stmts.append(s)
        return StmtBlock(stmts), ctx

    def _drop_if_empty(self, stmt, body: StmtBlock):
        return None if not body.stmts else stmt

    def _visit_if1(self, stmt: If1Stmt, ctx):
        body, _ = self._visit_block(stmt.body, ctx)
        return self._drop_if_empty(
            If1Stmt(self._visit_expr(stmt.cond, ctx), body, stmt.loc), body,
        ), ctx

    def _visit_for(self, stmt: ForStmt, ctx):
        body, _ = self._visit_block(stmt.body, ctx)
        return self._drop_if_empty(ForStmt(
            stmt.target, self._visit_expr(stmt.iterable, ctx), body, stmt.loc,
        ), body), ctx

    def _visit_while(self, stmt: WhileStmt, ctx):
        body, _ = self._visit_block(stmt.body, ctx)
        return self._drop_if_empty(WhileStmt(
            self._visit_expr(stmt.cond, ctx), body, stmt.loc,
        ), body), ctx

    def _visit_context(self, stmt: ContextStmt, ctx):
        body, _ = self._visit_block(stmt.body, ctx)
        return self._drop_if_empty(ContextStmt(
            stmt.target, self._visit_expr(stmt.ctx, ctx), body, stmt.loc,
        ), body), ctx

    def _visit_if(self, stmt: IfStmt, ctx):
        """An arm emptied by this becomes the other arm, negated.

        FPy has no empty block, so a two-armed `if` that loses one arm has to
        become a one-armed one rather than keep a hole.
        """
        cond = self._visit_expr(stmt.cond, ctx)
        ift, _ = self._visit_block(stmt.ift, ctx)
        iff, _ = self._visit_block(stmt.iff, ctx)
        if ift.stmts and iff.stmts:
            return IfStmt(cond, ift, iff, stmt.loc), ctx
        if ift.stmts:
            return If1Stmt(cond, ift, stmt.loc), ctx
        if iff.stmts:
            return If1Stmt(Not(cond, stmt.loc), iff, stmt.loc), ctx
        return None, ctx


class AssertElim:
    """Removes every `assert` from a function.

    A **semantic** change, not a cleanup: the program said to abort and the
    result will not.  So it is applied where a caller has asked for it, not
    as part of any normal form.

    A backend that cannot spell an assertion -- a Triton kernel cannot raise
    -- wants this *before* its passes rather than at emission, or a pass
    declines on a statement the backend was already told to discard.
    """

    @staticmethod
    def apply(func: FuncDef) -> FuncDef:
        if not isinstance(func, FuncDef):
            raise TypeError(f"Expected a 'FuncDef', got {func}")
        ast = _Eliminator()._visit_function(func, None)
        SyntaxCheck.check(ast, ignore_unknown=True)
        return ast
