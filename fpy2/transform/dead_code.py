"""
Dead code elimination.
"""

from ..ast import *



class _DeadCodeEliminate(DefaultVisitor):
    """
    Dead code elimination analysis.
    """

    func: FuncDef

    def __init__(self, func: FuncDef):
        self.func = func

    def apply(self):
        self._visit_function(self.func, None)

    def _visit_assign(self, stmt: Assign, ctx):
        raise NotImplementedError

    def _visit_indexed_assign(self, stmt: IndexedAssign, ctx):
        raise NotImplementedError

    def _visit_if1(self, stmt: If1Stmt, ctx):
        raise NotImplementedError

    def _visit_if(self, stmt: IfStmt, ctx):
        raise NotImplementedError

    def _visit_while(self, stmt: WhileStmt, ctx):
        raise NotImplementedError

    def _visit_for(self, stmt: ForStmt, ctx):
        raise NotImplementedError

    def _visit_context(self, stmt: ContextStmt, ctx):
        raise NotImplementedError

    def _visit_assert(self, stmt: AssertStmt, ctx):
        raise NotImplementedError

    def _visit_effect(self, stmt: EffectStmt, ctx):
        raise NotImplementedError

    def _visit_return(self, stmt: ReturnStmt, ctx):
        raise NotImplementedError

    def _visit_block(self, block: StmtBlock, ctx):
        raise NotImplementedError

    def _visit_function(self, func: FuncDef, ctx):
        raise NotImplementedError



class DeadCodeEliminate:
    """
    Dead code elimination.

    Removes any unused statements.
    """

    @staticmethod
    def apply(ast: FuncDef) -> FuncDef:
        if not isinstance(ast, FuncDef):
            raise TypeError(f'Expected `FuncDef`, got {type(ast)} for {ast}')
        return _DeadCodeEliminate(ast).apply()
