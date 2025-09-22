"""
Transformation pass to rewrite in-place tuple mutation as functional updates.
"""

from ..analysis import DefineUse, SyntaxCheck
from ..ast import *

class _FuncUpdateInstance(DefaultTransformVisitor):
    """Single-use instance of the FuncUpdate pass."""
    func: FuncDef

    def __init__(self, func: FuncDef):
        self.func = func

    def apply(self) -> FuncDef:
        return self._visit_function(self.func, None)

    def _visit_indexed_assign(self, stmt: IndexedAssign, ctx: None):
        slices = [self._visit_expr(slice, ctx) for slice in stmt.indices]
        expr = self._visit_expr(stmt.expr, ctx)
        e = ListSet(Var(stmt.var, None), slices, expr, stmt.loc)
        s = Assign(stmt.var, None, e, stmt.loc)
        return s, None

class FuncUpdate:
    """
    Transformation pass to rewrite in-place tuple mutation as functional updates.

    This pass rewrites the IR to use functional updates instead of
    in-place tuple mutation. While variables may still be mutated by
    re-assignment, this transformation ensures that no tuple is mutated.
    """

    @staticmethod
    def apply(func: FuncDef) -> FuncDef:
        ast = _FuncUpdateInstance(func).apply()
        SyntaxCheck.check(ast, ignore_unknown=True)
        return ast
