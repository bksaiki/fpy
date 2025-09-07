"""
This module defines a reachability analysis.
"""

import dataclasses

from ..ast import *

@dataclasses.dataclass
class _ReachabilityCtx:
    is_reachable: bool

    @staticmethod
    def default():
        return _ReachabilityCtx(True)

@dataclasses.dataclass
class ReachabilityAnalysis:
    by_stmt: dict[Stmt, bool]
    has_fallthrough: bool


class _ReachabilityInstance(DefaultVisitor):
    """
    Reachability analyzer instance.

    For each `_visit_XXX` for statement nodes:
    - the visitor context has a field `is_reachable` indicating if
    there is a reachable path to the node
    - the method returns whether there is an executable path
    that requires additional computation.
    """

    func: FuncDef
    is_reachable: dict[Stmt, bool] = {}

    def __init__(self, func: FuncDef):
        self.func = func
        self.is_reachable = {}

    def analyze(self):
        has_fallthrough = self._visit_function(self.func, _ReachabilityCtx.default())
        return ReachabilityAnalysis(self.is_reachable, has_fallthrough)

    def _visit_assign(self, stmt: Assign, ctx: _ReachabilityCtx) -> bool:
        # OUT[s] = IN[s]
        return ctx.is_reachable

    def _visit_indexed_assign(self, stmt: IndexedAssign, ctx: _ReachabilityCtx) -> bool:
        # OUT[s] = IN[s]
        return ctx.is_reachable

    def _visit_if1(self, stmt: If1Stmt, ctx: _ReachabilityCtx) -> bool:
        # IN[body] = IN[s]
        # OUT[s] = IN[s] |_| OUT[body]
        body_is_reachable = self._visit_block(stmt.body, ctx)
        return ctx.is_reachable or body_is_reachable

    def _visit_if(self, stmt: IfStmt, ctx: _ReachabilityCtx) -> bool:
        # IN[ift] = IN[s]
        # IN[iff] = IN[s]
        # OUT[s] = OUT[ift] |_| OUT[iff]
        ift_is_reachable = self._visit_block(stmt.ift, ctx)
        iff_is_reachable = self._visit_block(stmt.ift, ctx)
        return ift_is_reachable or iff_is_reachable

    def _visit_while(self, stmt: WhileStmt, ctx: _ReachabilityCtx) -> bool:
        # IN[body] = IN[s]
        # OUT[s] = IN[s] |_| OUT[body]
        body_is_reachable = self._visit_block(stmt.body, ctx)
        return ctx.is_reachable or body_is_reachable

    def _visit_for(self, stmt: ForStmt, ctx: _ReachabilityCtx) -> bool:
        # IN[body] = IN[s]
        # OUT[s] = IN[s] |_| OUT[body]
        body_is_reachable = self._visit_block(stmt.body, ctx)
        return ctx.is_reachable or body_is_reachable

    def _visit_context(self, stmt: ContextStmt, ctx: _ReachabilityCtx) -> bool:
        # IN[body] = IN[s]
        # OUT[s] = OUT[body]
        body_is_reachable = self._visit_block(stmt.body, ctx)
        return body_is_reachable

    def _visit_assert(self, stmt: AssertStmt, ctx: _ReachabilityCtx) -> bool:
        # OUT[s] = IN[s]
        return ctx.is_reachable

    def _visit_effect(self, stmt: EffectStmt, ctx: _ReachabilityCtx) -> bool:
        # OUT[s] = IN[s]
        return ctx.is_reachable

    def _visit_return(self, stmt: ReturnStmt, ctx: _ReachabilityCtx) -> bool:
        return False

    def _visit_statement(self, stmt: Stmt, ctx: _ReachabilityCtx):
        self.is_reachable[stmt] = ctx.is_reachable
        return super()._visit_statement(stmt, ctx)

    def _visit_block(self, block: StmtBlock, ctx: _ReachabilityCtx):
        for stmt in block.stmts:
            is_reachable = self._visit_statement(stmt, ctx)
            ctx = _ReachabilityCtx(is_reachable)
        return ctx.is_reachable

class Reachability:
    """
    This class performs a reachability analysis on an FPy program.

    It classifies statements each statement as
    - "possibly reachable": there exists an execution path from the entry to the statement.
    - "unreachable": there is no execution path from the entry to the statement.

    The dataflow equation for reachability is:

    OUT[s] = |_| IN[s]

    """

    @staticmethod
    def analyze(func: FuncDef, check_all_reachable: bool = False, check_no_fallthrough: bool = False):
        # run the analysis
        if not isinstance(func, FuncDef):
            raise TypeError(f'Expected \'FuncDef\', got {type(func)} for {func}')
        analysis = _ReachabilityInstance(func).analyze()

        # optionally check that all statements are reachable
        if check_all_reachable:
            for stmt, is_reachable in analysis.by_stmt.items():
                if not is_reachable:
                    print(f'WARNING: for `{func.name}`, `{stmt.format()}` is not reachable')

        # optionally check that every path through the program ends
        # at a return statement
        if check_no_fallthrough and analysis.has_fallthrough:
            print(f'WARN: for `{func.name}`, not all paths have a return statement')

        return analysis
