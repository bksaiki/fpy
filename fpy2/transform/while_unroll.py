"""
Unroller for `while` loops.
"""

from ..analysis import SyntaxCheck
from ..ast.fpyast import *
from ..ast.visitor import DefaultTransformVisitor

class _WhileUnroll(DefaultTransformVisitor):
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
    where: int | None
    times: int
    index: int

    def __init__(self, func: FuncDef, where: int | None, times: int) -> None:
        super().__init__()
        self.func = func
        self.where = where
        self.times = times
        self.index = 0

    def _visit_while(self, stmt: WhileStmt, ctx: None):
        if self.where is None or self.index == self.where:
            self.index += 1
            # original loop
            cond = self._visit_expr(stmt.cond, ctx)
            body, _ = self._visit_block(stmt.body, ctx)
            ret_stmt: Stmt = WhileStmt(cond, body, stmt.loc)
            # unroll n times
            for _ in range(self.times):
                cond = self._visit_expr(stmt.cond, ctx)
                body, _ = self._visit_block(stmt.body, ctx)
                block = StmtBlock(body.stmts + [ret_stmt])
                ret_stmt = If1Stmt(cond, block, stmt.loc)

            return ret_stmt, None
        else:
            self.index += 1
            return super()._visit_while(stmt, ctx)

    def _visit_block(self, block: StmtBlock, ctx: None):
        new_stmts = []
        for stmt in block.stmts:
            stmt, _ = self._visit_statement(stmt, ctx)
            new_stmts.append(stmt)
        return StmtBlock(new_stmts), None

    def apply(self):
        return self._visit_function(self.func, None)


class WhileUnroll:
    """
    Unrolling for `while` loops.
    """

    @staticmethod
    def apply(func: FuncDef, where: int | None = None, times: int = 1):
        """
        Apply the transformation.

        Parameters
        ----------
        where : int | None
            The index of the `while` loop to unroll. If `None`, unroll all
            `while` loops.
        times : int
            The number of times to unroll the loop.
        """
        if not isinstance(func, FuncDef):
            raise TypeError(f"Expected a \'FuncDef\', got {func}")
        if not isinstance(times, int):
            raise TypeError(f"Expected an \'int\' for times, got {times}")
        if times < 0:
            raise ValueError(f"Expected a non-negative integer for times, got {times}")

        unroller = _WhileUnroll(func, where, times)
        func = unroller.apply()
        SyntaxCheck.check(func, ignore_unknown=True)
        return func
