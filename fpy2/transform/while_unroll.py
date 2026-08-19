"""
Unroller for `while` loops.
"""

from ..analysis import SyntaxCheck
from ..ast.fpyast import *
from .utils import Cursor, EditLog, SiteRewriter, check_where


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
        self.site_idx = 0

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

    def _visit_block(self, block: StmtBlock, ctx: None):
        new_stmts: list[Stmt] = []
        for pos, stmt in enumerate(block.stmts):
            self._site = (block, pos)
            self._replaced = False
            s, _ = self._visit_statement(stmt, ctx)
            new_stmts.append(s)
            if self._replaced:
                # the loop became one statement: an `if` guarding the rest
                self._record(block, pos, 1)
                self._replaced = False
        return StmtBlock(new_stmts), None

    def apply(self):
        return self._visit_function(self.func, None)


class WhileUnroll:
    """
    Unrolling for `while` loops.
    """

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
        """:meth:`apply`, with the record of what it replaced; the rewritten
        program is the log's `result`."""
        if not isinstance(func, FuncDef):
            raise TypeError(f"Expected a \'FuncDef\', got {func}")
        check_where(where)
        if not isinstance(times, int):
            raise TypeError(f"Expected an \'int\' for times, got {times}")
        if times < 0:
            raise ValueError(f"Expected a non-negative integer for times, got {times}")

        unroller = _WhileUnroll(func, where, times)
        out = unroller.apply()
        # A `where` that named no loop leaves the function unchanged; fail
        # rather than silently no-op.  When `where` is out of range no loop
        # matches, so no body is re-visited and `site_idx` is the true loop
        # count; an in-range `where` matches (so `site_idx` exceeds it, even
        # though re-visiting a matched loop's body can inflate the counter).
        unroller.check_site('a `while` loop')
        SyntaxCheck.check(out, ignore_unknown=True)
        return EditLog(func, out, tuple(unroller.edits), exprs_preserved=True)
