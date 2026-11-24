"""
This module defines a reachability analysis.
"""

import dataclasses

from ..ast import *


class ReachabilityError(Exception):
    """Assertion error from a `Reachability` analysis."""


def _name_of(ast: FuncDef | StmtBlock) -> str:
    return str(ast.name) if isinstance(ast, FuncDef) else '<block>'

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

    def falls_through(self, block: StmtBlock) -> bool:
        """Does some path reach the end of *block*?"""
        return not block.stmts or self.has_exit[block.stmts[-1]]


class _ReachabilityInstance(DefaultVisitor):
    """
    Reachability analyzer instance.

    Each statement visitor takes whether the statement is reachable and
    returns whether execution can continue past it.
    """

    func: FuncDef | StmtBlock
    has_entry: dict[Stmt, bool]
    has_exit: dict[Stmt, bool]
    ret_stmts: set[ReturnStmt]

    def __init__(self, func: FuncDef | StmtBlock):
        self.func = func
        self.has_entry = {}
        self.has_exit = {}
        self.ret_stmts = set()

    def analyze(self):
        match self.func:
            case FuncDef():
                has_fallthrough = self._visit_function(self.func, True)
            case StmtBlock():
                has_fallthrough = self._visit_block(self.func, True)
            case _:
                raise RuntimeError(f'unexpected AST node {self.func}')
        return ReachabilityAnalysis(self.has_entry, self.has_exit, self.ret_stmts, has_fallthrough)

    def _visit_assign(self, stmt: Assign, reachable: bool) -> bool:
        # OUT[s] = IN[s]
        return reachable

    def _visit_indexed_assign(self, stmt: IndexedAssign, reachable: bool) -> bool:
        # OUT[s] = IN[s]
        return reachable

    def _visit_if1(self, stmt: If1Stmt, reachable: bool) -> bool:
        # IN[body] = IN[s]
        # OUT[s] = IN[s] |_| OUT[body]
        body_is_reachable = self._visit_block(stmt.body, reachable)
        return reachable or body_is_reachable

    def _visit_if(self, stmt: IfStmt, reachable: bool) -> bool:
        # IN[ift] = IN[s]
        # IN[iff] = IN[s]
        # OUT[s] = OUT[ift] |_| OUT[iff]
        ift_is_reachable = self._visit_block(stmt.ift, reachable)
        iff_is_reachable = self._visit_block(stmt.iff, reachable)
        return ift_is_reachable or iff_is_reachable

    def _visit_while(self, stmt: WhileStmt, reachable: bool) -> bool:
        # IN[body] = IN[s]
        # OUT[s] = IN[s] |_| OUT[body]
        body_is_reachable = self._visit_block(stmt.body, reachable)
        return reachable or body_is_reachable

    def _visit_for(self, stmt: ForStmt, reachable: bool) -> bool:
        # IN[body] = IN[s]
        # OUT[s] = IN[s] |_| OUT[body]
        body_is_reachable = self._visit_block(stmt.body, reachable)
        return reachable or body_is_reachable

    def _visit_context(self, stmt: ContextStmt, reachable: bool) -> bool:
        # IN[body] = IN[s]
        # OUT[s] = OUT[body]
        body_is_reachable = self._visit_block(stmt.body, reachable)
        return body_is_reachable

    def _visit_assert(self, stmt: AssertStmt, reachable: bool) -> bool:
        # OUT[s] = IN[s]
        return reachable

    def _visit_effect(self, stmt: EffectStmt, reachable: bool) -> bool:
        # OUT[s] = IN[s]
        return reachable

    def _visit_return(self, stmt: ReturnStmt, reachable: bool) -> bool:
        self.ret_stmts.add(stmt)
        return False
    
    def _visit_pass(self, stmt: PassStmt, reachable: bool) -> bool:
        # OUT[s] = IN[s]
        return reachable

    def _visit_statement(self, stmt: Stmt, reachable: bool):
        self.has_entry[stmt] = reachable
        is_reachable = super()._visit_statement(stmt, reachable)
        self.has_exit[stmt] = is_reachable
        return is_reachable

    def _visit_block(self, block: StmtBlock, reachable: bool):
        for stmt in block.stmts:
            reachable = self._visit_statement(stmt, reachable)
        return reachable

    def _visit_function(self, func: FuncDef, reachable: bool):
        return self._visit_block(func.body, reachable)

class Reachability:
    """
    This class performs a reachability analysis on an FPy program.

    It classifies each statement as
    - "possibly reachable": there exists an execution path from the entry to the statement.
    - "unreachable": there is no execution path from the entry to the statement.

    `IN[s]` is `OUT` of the preceding statement; each visitor gives `OUT[s]`.

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
    def analyze(
        func: FuncDef | StmtBlock,
        *,
        check_all_reachable: bool = False,
        check_no_fallthrough: bool = False,
        check_single_exit: bool = False,
    ):
        """Run the analysis.  Each ``check_*`` flag opts into one of
        the three optional assertions documented on the class."""
        if not isinstance(func, FuncDef | StmtBlock):
            raise TypeError(f'Expected \'FuncDef\' or \'StmtBlock\', got {type(func)} for {func}')
        analysis = _ReachabilityInstance(func).analyze()

        if check_all_reachable:
            unreachable: list[Stmt] = []
            for stmt, is_reachable in analysis.has_entry.items():
                if not is_reachable:
                    unreachable.append(stmt)

            if unreachable:
                fmt_str = [f'`{_name_of(func)}` has unreachable statements']
                fmt_str.extend(
                    f'at ???: `{stmt.format()}`' if stmt.loc is None else f'at {stmt.loc.format()}: `{stmt.format()}`'
                    for stmt in unreachable)
                raise ReachabilityError('\n  '.join(fmt_str))

        if check_no_fallthrough and analysis.has_fallthrough:
            raise ReachabilityError(f'in `{_name_of(func)}`: not all paths have a return statement')

        if check_single_exit and len(analysis.ret_stmts) != 1:
            raise ReachabilityError(f'`{_name_of(func)}` must have a single return statement')

        return analysis
