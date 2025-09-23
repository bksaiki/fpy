"""
Definition analysis.
"""

from typing import TypeAlias

from ..ast.fpyast import *
from ..ast.visitor import DefaultVisitor

_Defs: TypeAlias = set[NamedId]
_DefMap: TypeAlias = dict[StmtBlock, _Defs]

__all__ = [
    'DefAnalysis',
]

class _DefAnalysis(DefaultVisitor):
    """Visitor for definition analysis."""

    ast: FuncDef | StmtBlock
    blocks: _DefMap

    def __init__(self, ast: FuncDef | StmtBlock):
        self.ast = ast
        self.blocks = {}

    def _visit_assign(self, stmt: Assign, ctx: None):
        # introduces bindings
        defs: _Defs = set()
        for name in stmt.target.names():
            defs.add(name)
        return defs

    def _visit_indexed_assign(self, stmt: IndexedAssign, ctx: None):
        # does not introduce bindings
        return set()

    def _visit_if1(self, stmt: If1Stmt, ctx: None):
        # does not introduce bindings but body might
        return self._visit_block(stmt.body, ctx)

    def _visit_if(self, stmt: IfStmt, ctx: None):
        # does not introduce bindings but branches might
        ift_defs = self._visit_block(stmt.ift, ctx)
        iff_defs = self._visit_block(stmt.iff, ctx)
        return ift_defs | iff_defs

    def _visit_while(self, stmt: WhileStmt, ctx: None):
        # does not introduce bindings but body might
        return self._visit_block(stmt.body, ctx)

    def _visit_for(self, stmt: ForStmt, ctx: None):
        # introduces binding for loop variable
        defs: _Defs = set()
        for name in stmt.target.names():
            defs.add(name)
        defs |= self._visit_block(stmt.body, ctx)
        return defs

    def _visit_context(self, stmt: ContextStmt, ctx: None):
        # might introduce a binding
        defs: _Defs = set()
        if isinstance(stmt.target, NamedId):
            defs.add(stmt.target)
        return defs | self._visit_block(stmt.body, ctx)

    def _visit_assert(self, stmt: AssertStmt, ctx: None):
        return set()

    def _visit_effect(self, stmt: EffectStmt, ctx: None):
        return set()

    def _visit_return(self, stmt: ReturnStmt, ctx: None):
        return set()

    def _visit_pass(self, stmt: PassStmt, ctx: None):
        return set()

    def _visit_block(self, block: StmtBlock, ctx: None):
        ds: _Defs = set()
        for stmt in block.stmts:
            ds |= self._visit_statement(stmt, ctx)
        self.blocks[block] = ds
        return ds

    def analyze(self):
        match self.ast:
            case FuncDef():
                self._visit_function(self.ast, None)
            case StmtBlock():
                self._visit_block(self.ast, None)
            case _:
                raise RuntimeError(f'unexpected AST node {self.ast}')

        return self.blocks


class DefAnalysis:
    """
    Definition analysis.

    Computes the set of targets seen within each statement block.
    """

    @staticmethod
    def analyze(ast: FuncDef | StmtBlock):
        if not isinstance(ast, FuncDef | StmtBlock):
            raise TypeError(f'Expected \'FuncDef\' or \'StmtBlock\', got {type(ast)} for {ast}')
        return _DefAnalysis(ast).analyze()
