"""
Dead code elimination.

TODO:
- rewrite `if True: ... else: ...` to just the `if` body
"""

from typing import cast

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
    def_use: DefineUseAnalysis
    unused_assign: set[Assign]
    unused_fv: set[NamedId]
    eliminated: bool

    def __init__(self, func: FuncDef, def_use: DefineUseAnalysis, unused_assign: set[Assign], unused_fv: set[NamedId]):
        self.func = func
        self.def_use = def_use
        self.unused_assign = unused_assign
        self.unused_fv = unused_fv
        self.eliminated = False

    def _is_empty_block(self, block: StmtBlock) -> bool:
        return len(block.stmts) == 1 and isinstance(block.stmts[0], PassStmt)

    def _visit_assign(self, assign: Assign, ctx: None):
        # remove any assignment marked for deletion
        if assign in self.unused_assign:
            # eliminate unused assignment
            self.eliminated = True
            return None, ctx
        else:
            return super()._visit_assign(assign, ctx)

    def _visit_if1(self, stmt: If1Stmt, ctx: None):
        if isinstance(stmt.cond, BoolVal):
            if stmt.cond.val:
                # if True: ... -> ...
                # return the block directly
                self.eliminated = True
                body = self._visit_block(stmt.body, ctx)
                return body, ctx
            else:
                # if False: ... -> (nothing)
                self.eliminated = True
                return None, ctx
        elif self._is_empty_block(stmt.body) and Purity.analyze_expr(stmt.cond, self.def_use):
            # if _: pass -> (nothing)
            self.eliminated = True
            return None, ctx
        else:
            # nothing to eliminate
            return super()._visit_if1(stmt, ctx)

    def _visit_if(self, stmt: IfStmt, ctx: None):
        if isinstance(stmt.cond, BoolVal):
            if stmt.cond.val:
                # if True: ... else: ... -> ...
                self.eliminated = True
                ift, _ = self._visit_block(stmt.ift, ctx)
                return ift, ctx
            else:
                # if False: ... else: ... -> ...
                self.eliminated = True
                iff, _ = self._visit_block(stmt.iff, ctx)
                return iff, ctx
        elif self._is_empty_block(stmt.ift) and self._is_empty_block(stmt.iff) and Purity.analyze_expr(stmt.cond, self.def_use):
            # if _: pass else: pass -> (nothing)
            self.eliminated = True
            return None, ctx
        elif self._is_empty_block(stmt.ift) and Purity.analyze_expr(stmt.cond, self.def_use):
            # if _: pass else: ... -> if not _: ...
            self.eliminated = True
            s = If1Stmt(Not(stmt.cond, stmt.loc), stmt.iff, loc=stmt.loc)
            return s, ctx
        elif self._is_empty_block(stmt.iff) and Purity.analyze_expr(stmt.cond, self.def_use):
            # if _: ... else: pass -> ...
            self.eliminated = True
            s = If1Stmt(stmt.cond, stmt.ift, loc=stmt.loc)
            return s, ctx
        else:
            # nothing to eliminate
            return super()._visit_if(stmt, ctx)

    def _visit_while(self, stmt: WhileStmt, ctx: None):
        # remove `while False: ...`
        # remove `while _: pass`
        if (isinstance(stmt.cond, BoolVal) and not stmt.cond.val) or self._is_empty_block(stmt.body):
            # eliminate unnecessary while statement
            self.eliminated = True
            return None, ctx
        else:
            return super()._visit_while(stmt, ctx)

    def _visit_pass(self, stmt: PassStmt, ctx: None):
        # unnecessary pass statement
        self.eliminated = True
        return None, ctx

    def _visit_block(self, block: StmtBlock, ctx: None):
        if self._is_empty_block(block):
            # do nothing
            return block, ctx
        else:
            # visit statements
            stmts: list[Stmt] = []
            for stmt in block.stmts:
                s, _ = self._visit_statement(stmt, ctx)
                # s = cast(Stmt | StmtBlock | None, s)
                match s:
                    case None:
                        pass
                    case Stmt():
                        stmts.append(s)
                    case StmtBlock():
                        stmts.extend(s.stmts)
                    case _:
                        raise RuntimeError(f'unexpected: {s}')

            # empty block -> add a pass statement
            if len(stmts) == 0:
                stmts.append(PassStmt(None))
            return StmtBlock(stmts), ctx

    def _visit_function(self, func: FuncDef, ctx: None):
        free_vars = set(v for v in func.free_vars if v not in self.unused_fv)
        body, _ = self._visit_block(func.body, ctx)
        return FuncDef(func.name, func.args, free_vars, func.ctx, body, func.spec, func.meta, loc=func.loc)

    def _apply(self):
        func = self._visit_function(self.func, None)
        return func, self.eliminated


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
            # phi variables are virtual definitions:
            # if a phi variable is used, so are its children
            # if a phi variable is not used, its children are also unused
            used_assigns: set[AssignDef] = set()
            for phis in self.def_use.phis.values():
                for phi in phis:
                    if len(self.def_use.uses[phi]) > 0:
                        # phi variable is used, so 
                        if isinstance(phi.lhs, AssignDef):
                            used_assigns.add(phi.lhs)
                        if isinstance(phi.rhs, AssignDef):
                            used_assigns.add(phi.rhs)

            # process def-use analysis for definitions without uses
            # specifically interested in assignments, phi variables,
            # and free variables
            unused_assign: set[Assign] = set()
            unused_fv: set[NamedId] = set()
            for d, uses in self.def_use.uses.items():
                if len(uses) == 0 and len(self.def_use.successors[d]) == 0:
                    # print(d.name, d.site.format())
                    if isinstance(d, AssignDef):
                        if isinstance(d.site, FuncDef):
                            # free variable
                            unused_fv.add(d.name)
                        elif (
                            isinstance(d.site, Assign)
                            # and d not in used_assigns
                            and Purity.analyze_expr(d.site.expr, self.def_use)
                        ):
                            # assignment
                            unused_assign.add(d.site)

            # print(unused_assign)

            # run code eliminator
            self.func, eliminated = _Eliminator(self.func, self.def_use, unused_assign, unused_fv)._apply()
            if not eliminated:
                return self.func

            # removed something so try again
            self.def_use = DefineUse.analyze(self.func)


class DeadCodeEliminate:
    """
    Dead code elimination.
    - removes any unused statements
    - removes any unused free variables
    - removes any never-executed branch
    - removes empty bodies
    """

    @staticmethod
    def apply(ast: FuncDef, def_use: DefineUseAnalysis | None = None) -> FuncDef:
        if not isinstance(ast, FuncDef):
            raise TypeError(f'Expected `FuncDef`, got {type(ast)} for {ast}')
        if def_use is None:
            def_use = DefineUse.analyze(ast)
        return _DeadCodeEliminate(ast, def_use).apply()
