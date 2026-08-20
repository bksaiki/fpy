"""
Unroller for `while` loops.
"""

from ..analysis import SyntaxCheck
from ..ast.fpyast import *
from .cursor import Cursor, EditLog, StmtCursor, stmt_sites
from .utils import SiteRewriter, check_where


class _WhileUnroll(SiteRewriter):
    """
    Unroll visitor.

    A single unroll rewrites
    ```
    while <cond>:
        <body>
    ```
    to
    ```
    if <cond>:
        <body>
        while <cond>:
            <body>
    ```
    """

    func: FuncDef
    where: int | Cursor | None
    times: int

    def __init__(
        self, func: FuncDef, where: int | Cursor | None, times: int
    ) -> None:
        super().__init__()
        self.func = func
        self.where = where
        self.times = times

    def _visit_while(self, stmt: WhileStmt, ctx: None):
        block, pos = self._site
        idx = self.site_idx
        self.site_idx += 1
        if self._selects(block, pos, idx):
            self._matched += 1
            # original loop
            cond = self._visit_expr(stmt.cond, ctx)
            body, _ = self._visit_block(stmt.body, ctx)
            ret_stmt: Stmt = WhileStmt(cond, body, stmt.loc)
            # unroll n times
            for _ in range(self.times):
                cond = self._visit_expr(stmt.cond, ctx)
                body, _ = self._visit_block(stmt.body, ctx)
                unrolled = StmtBlock(body.stmts + [ret_stmt])
                ret_stmt = If1Stmt(cond, unrolled, stmt.loc)

            # a zero-times unroll rebuilds the loop and changes nothing, so
            # cursors inside its body still name what they named
            self._replaced = self.times > 0
            return ret_stmt, None
        else:
            return super()._visit_while(stmt, ctx)

    def apply(self):
        return self._visit_function(self.func, None)


class WhileUnroll:
    """
    Unrolling for `while` loops.
    """

    @staticmethod
    def sites(func: FuncDef, within: Cursor | None = None) -> list[StmtCursor]:
        """The `while` loops of `func`, in visit order: what a `where`
        index counts."""
        return stmt_sites(func, lambda s: isinstance(s, WhileStmt), within)

    @staticmethod
    def apply(
        func: FuncDef, where: int | Cursor | None = None, times: int = 1
    ) -> FuncDef:
        """
        Apply the transformation.

        Parameters
        ----------
        where : int | Cursor | None
            Which `while` loop to unroll: an index counting loops in visit
            order, or a cursor or region naming a program point, which takes
            every loop at or beneath it. If `None`, unroll every `while` loop.
        times : int
            The number of times to unroll the loop.
        """
        return WhileUnroll.apply_with_edits(func, where, times).result

    @staticmethod
    def apply_with_edits(
        func: FuncDef, where: int | Cursor | None = None, times: int = 1
    ) -> EditLog:
        """:meth:`apply`, with an :class:`EditLog` of what it replaced."""
        if not isinstance(func, FuncDef):
            raise TypeError(f"Expected a \'FuncDef\', got {func}")
        check_where(where)
        if not isinstance(times, int):
            raise TypeError(f"Expected an \'int\' for times, got {times}")
        if times < 0:
            raise ValueError(f"Expected a non-negative integer for times, got {times}")

        unroller = _WhileUnroll(func, where, times)
        out = unroller.apply()
        # An in-range `where` matches, so `site_idx` exceeds it even though
        # re-visiting a matched loop's body inflates the counter
        unroller.check_site('a `while` loop')
        SyntaxCheck.check(out, ignore_unknown=True)
        return EditLog(func, out, tuple(unroller.edits), exprs_preserved=True)
