"""
Dead code elimination.
"""

from ..ast import *
from ..analysis import (
    DefineUse, DefineUseAnalysis, AssignDef, PhiDef,
    Purity
)

class _Eliminator(DefaultTransformVisitor):
    """
    Dead code eliminator.
    """

    func: FuncDef
    unused_assign: set[Assign]
    unused_fv: set[NamedId]

    def __init__(self, func: FuncDef, unused_assign: set[Assign], unused_fv: set[NamedId]):
        self.func = func
        self.unused_assign = unused_assign
        self.unused_fv = unused_fv

    def _visit_block(self, block: StmtBlock, ctx: None):
        stmts: list[Stmt] = []
        for stmt in block.stmts:
            if not isinstance(stmt, Assign) or stmt not in self.unused_assign:
                s, _ = self._visit_statement(stmt, ctx)
                stmts.append(s)

        if len(stmts) == 0:
            stmts.append(PassStmt(None))
        return StmtBlock(stmts), ctx

    def _visit_function(self, func: FuncDef, ctx: None):
        free_vars = set(v for v in func.free_vars if v not in self.unused_fv)
        body, _ = self._visit_block(func.body, ctx)
        return FuncDef(func.name, func.args, free_vars, func.ctx, body, func.spec, func.meta, loc=func.loc)

    def _apply(self):
        return self._visit_function(self.func, None)


class _DeadCodeEliminate:
    """
    Dead code elimination analysis.
    """

    func: FuncDef
    def_use: DefineUseAnalysis

    def __init__(self, func: FuncDef, def_use: DefineUseAnalysis):
        self.func = func
        self.def_use = def_use

    def apply(self):
        while True:
            # compute variables that are overriden by phi variables
            phi_args: set[AssignDef] = set()
            for phis in self.def_use.phis.values():
                for phi in phis:
                    if isinstance(phi.lhs, AssignDef):
                        phi_args.add(phi.lhs)
                    if isinstance(phi.rhs, AssignDef):
                        phi_args.add(phi.rhs)

            # process def-use analysis for definitions without uses
            # specifically interested in assignments, phi variables,
            # and free variables
            unused_assign: set[Assign] = set()
            unused_fv: set[NamedId] = set()
            for d, uses in self.def_use.uses.items():
                if len(uses) == 0:
                    if isinstance(d, AssignDef):
                        if isinstance(d.site, FuncDef):
                            # free variable
                            unused_fv.add(d.name)
                        elif (
                            isinstance(d.site, Assign)
                            and d not in phi_args
                            and Purity.analyze_expr(d.site.expr, self.def_use)
                        ):
                            # assignment
                            unused_assign.add(d.site)

            if len(unused_assign) == 0 and len(unused_fv) == 0:
                # nothing to do
                return self.func

            # need to eliminate dead code and try again
            self.func = _Eliminator(self.func, unused_assign, unused_fv)._apply()
            print(self.func.format())
            self.def_use = DefineUse.analyze(self.func)

class DeadCodeEliminate:
    """
    Dead code elimination.

    Removes any unused statements.
    """

    @staticmethod
    def apply(ast: FuncDef, def_use: DefineUseAnalysis | None = None) -> FuncDef:
        if not isinstance(ast, FuncDef):
            raise TypeError(f'Expected `FuncDef`, got {type(ast)} for {ast}')
        if def_use is None:
            def_use = DefineUse.analyze(ast)
        return _DeadCodeEliminate(ast, def_use).apply()
