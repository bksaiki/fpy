"""
Copy propagation.
"""

from ..analysis import DefineUse, SyntaxCheck
from ..ast import *


class _CopyPropagateInstance(DefaultAstVisitor):
    """Single-use instance of copy propagation."""
    func: FuncDef
    xform: DefaultAstTransformVisitor

    def __init__(self, func: FuncDef):
        self.func = func
        self.xform = DefaultAstTransformVisitor()

    def apply(self):
        """Applies copy propagation to the function."""
        # create a copy of the AST and run definition-use analysis
        func = self.xform._visit_function(self.func, None)
        def_use = DefineUse.analyze(func)

        # find direct assigments and substitute them
        remove: set[SimpleAssign] = set()
        for d, uses in def_use.uses.items():
            if isinstance(d, SimpleAssign) and isinstance(d.expr, Var):
                # direct assignment: x = y
                # substitute all occurences of this definition of `x` with `y`
                remove.add(d)
                for use in uses:
                    match use:
                        case Var():
                            use.name = d.expr.name
                        case IndexAssign():
                            use.var = d.expr.name
                        case _:
                            raise RuntimeError('unreachable', use)

        # eliminate the assignments
        self._visit_function(func, remove)
        return func

    def _visit_block(self, block: StmtBlock, ctx: set[SimpleAssign]):
        stmts: list[Stmt] = []
        for stmt in block.stmts:
            if not isinstance(stmt, SimpleAssign) or stmt not in ctx:
                self._visit_statement(stmt, ctx)
                stmts.append(stmt)
        block.stmts = stmts


class CopyPropagate:
    """
    Copy propagation.

    This transform replaces any variable that is assigned another variable.
    """

    @staticmethod
    def apply(func: FuncDef):
        """Applies copy propagation to the given AST."""
        if not isinstance(func, FuncDef):
            raise TypeError(f'Expected \'FuncDef\' for {func}, got {type(func)}')
        func = _CopyPropagateInstance(func).apply()
        SyntaxCheck.check(func)
        return func
