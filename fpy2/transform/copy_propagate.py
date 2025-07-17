"""
Copy propagation.
"""

from typing import Optional

from ..analysis import DefineUse, SyntaxCheck
from ..ast import *


class _CopyPropagateInstance(DefaultVisitor):
    """Single-use instance of copy propagation."""
    func: FuncDef
    names: Optional[set[NamedId]]
    xform: DefaultTransformVisitor

    def __init__(self, func: FuncDef, names: Optional[set[NamedId]]):
        self.func = func
        self.names = names
        self.xform = DefaultTransformVisitor()

    def apply(self):
        """Applies copy propagation to the function."""
        # create a copy of the AST and run definition-use analysis
        func = self.xform._visit_function(self.func, None)
        def_use = DefineUse.analyze(func)

        # find direct assigments and substitute them
        remove: set[Assign] = set()
        for name, defs in def_use.defs.items():
            # skip any names not matching the filter
            if self.names is not None and name not in self.names:
                continue

            for d in defs:
                if isinstance(d, Assign) and isinstance(d.expr, Var):
                    # direct assignment: x = y
                    # substitute all occurences of this definition of `x` with `y`
                    remove.add(d)
                    for use in def_use.uses[d]:
                        match use:
                            case Var():
                                use.name = d.expr.name
                            case IndexedAssign():
                                use.var = d.expr.name
                            case _:
                                raise RuntimeError('unreachable', use)

        # eliminate the assignments
        self._visit_function(func, remove)
        return func

    def _visit_block(self, block: StmtBlock, ctx: set[Assign]):
        stmts: list[Stmt] = []
        for stmt in block.stmts:
            if not isinstance(stmt, Assign) or stmt not in ctx:
                self._visit_statement(stmt, ctx)
                stmts.append(stmt)
        block.stmts = stmts


class CopyPropagate:
    """
    Copy propagation.

    This transform replaces any variable that is assigned another variable.
    """

    @staticmethod
    def apply(func: FuncDef, *, names: Optional[set[NamedId]] = None) -> FuncDef:
        """Applies copy propagation to the given AST."""
        if not isinstance(func, FuncDef):
            raise TypeError(f'Expected \'FuncDef\' for {func}, got {type(func)}')
        func = _CopyPropagateInstance(func, names).apply()
        SyntaxCheck.check(func, ignore_unknown=True)
        return func
