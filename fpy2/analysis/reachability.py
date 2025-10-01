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


class ReachabilityError(Exception):
    """Assertion error from a `Reachability` analysis."""
    pass

@dataclasses.dataclass
class ReachabilityAnalysis:
    """Result type of a `Reachability` analysis."""

    has_entry: dict[Stmt, bool]
    """is the statement reachable?"""
    has_exit: dict[Stmt, bool]
    """is there a reachable path through the statement?"""
    ret_stmts: set[ReturnStmt]
    """all return statements in the program"""
    has_fallthrough: bool
    """is there a path that does not end in a return statement"""


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
    has_entry: dict[Stmt, bool]
    has_exit: dict[Stmt, bool]
    ret_stmts: set[ReturnStmt]

    def __init__(self, func: FuncDef):
        self.func = func
        self.has_entry = {}
        self.has_exit = {}
        self.ret_stmts = set()

    def analyze(self):
        has_fallthrough = self._visit_function(self.func, _ReachabilityCtx.default())
        return ReachabilityAnalysis(self.has_entry, self.has_exit, self.ret_stmts, has_fallthrough)

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
        iff_is_reachable = self._visit_block(stmt.iff, ctx)
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
        self.ret_stmts.add(stmt)
        return False
    
    def _visit_pass(self, stmt: PassStmt, ctx: _ReachabilityCtx) -> bool:
        # OUT[s] = IN[s]
        return ctx.is_reachable

    def _visit_statement(self, stmt: Stmt, ctx: _ReachabilityCtx):
        self.has_entry[stmt] = ctx.is_reachable
        is_reachable = super()._visit_statement(stmt, ctx)
        self.has_exit[stmt] = is_reachable
        return is_reachable

    def _visit_block(self, block: StmtBlock, ctx: _ReachabilityCtx):
        for stmt in block.stmts:
            is_reachable = self._visit_statement(stmt, ctx)
            ctx = _ReachabilityCtx(is_reachable)
        return ctx.is_reachable

    def _visit_function(self, func: FuncDef, ctx: _ReachabilityCtx):
        # TODO: add a reduce visitor?
        return self._visit_block(func.body, ctx)

class Reachability:
    """
    This class performs a reachability analysis on an FPy program.

    It classifies statements each statement as
    - "possibly reachable": there exists an execution path from the entry to the statement.
    - "unreachable": there is no execution path from the entry to the statement.

    The dataflow equation for reachability is:

    `OUT[s] = |_| IN[s]`

    The resulting `ReachabilityAnalysis` has four fields:
    - `has_entry`: is the statement reachable?
    - `has_exit`: is there a reachable path through the statement?
    - `ret_stmts: all return statements in the program
    - `has_fallthrough`: is there a path that does not end in a return statement

    The analysis may optionally assert three properties:
    - `check_all_reachable`: every statement in the program is reachable
    - `check_no_fallthrough`: every program path ends at a return statement
    - `check_single_exit`: the program has only one return statement
    """

    @staticmethod
    def analyze(func: FuncDef, check: bool = False):
        # run the analysis
        if not isinstance(func, FuncDef):
            raise TypeError(f'Expected \'FuncDef\', got {type(func)} for {func}')
        analysis = _ReachabilityInstance(func).analyze()

        if check:
            # optionally check that all statements are reachable
            unreachable: list[Stmt] = []
            for stmt, is_reachable in analysis.has_entry.items():
                if not is_reachable:
                    unreachable.append(stmt)

            if unreachable:
                fmt_str = [f'`{func.name}` has unreachable statements']
                fmt_str.extend(
                    f'at ???: `{stmt.format()}`' if stmt.loc is None else f'at {stmt.loc.format()}: `{stmt.format()}`'
                    for stmt in unreachable)
                raise ReachabilityError('\n  '.join(fmt_str))

            # optionally check that every path through the program ends
            # at a return statement
            if analysis.has_fallthrough:
                raise ReachabilityError(f'in `{func.name}`: not all paths have a return statement')

            # optionally check that there is exactly one return statement
            if len(analysis.ret_stmts) != 1:
                raise ReachabilityError(f'`{func.name}` must have a single return statement')

        return analysis
