"""
Dead code elimination.

TODO:
- rewrite `if True: ... else: ...` to just the `if` body
"""

from typing import cast

from ..ast import *
from ..analysis import (
    DefineUse, DefineUseAnalysis, AssignDef, PhiDef,
    Purity, SyntaxCheck
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

    def __init__(
        self,
        func: FuncDef,
        def_use: DefineUseAnalysis,
        unused_assign: set[Assign],
        unused_fv: set[NamedId]
    ):
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

    def _visit_if1(self, stmt: If1Stmt, ctx: None) -> tuple[Stmt | StmtBlock | None, None]:
        if isinstance(stmt.cond, BoolVal):
            if stmt.cond.val:
                # if True: ... -> ...
                # return the block directly
                self.eliminated = True
                body, _ = self._visit_block(stmt.body, ctx)
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

    def _visit_if(self, stmt: IfStmt, ctx: None) -> tuple[Stmt | StmtBlock | None, None]:
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

    def _visit_while(self, stmt: WhileStmt, ctx: None) -> tuple[Stmt | StmtBlock | None, None]:
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

    def _visit_block(self, block: StmtBlock, ctx: None) -> tuple[StmtBlock, None]:
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
        return FuncDef(func.name, func.args, free_vars, func.ctx, body, func.spec, func.meta, func.env, loc=func.loc)

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
            # process def-use analysis for definitions without uses
            # specifically interested in assignments, phi variables, and free variables
            unused_assign: set[Assign] = set()
            unused_fv: set[NamedId] = set()
            unused_phi: set[PhiDef] = set()
            for d, uses in self.def_use.uses.items():
                if len(uses) > 0 or any(isinstance(s, PhiDef) for s in self.def_use.successors[d]):
                    continue
                match d:
                    case AssignDef():
                        if isinstance(d.site, FuncDef):
                            # free variable
                            unused_fv.add(d.name)
                        elif (
                            isinstance(d.site, Assign)
                            and Purity.analyze_expr(d.site.expr, self.def_use)
                        ):
                            # assignment
                            unused_assign.add(d.site)
                    case PhiDef():
                        # phi variable with no uses
                        unused_phi.add(d)
                    case _:
                        raise RuntimeError(f'unexpected def: {d}')

            # if a phi variable is unused, then its arguments are also unused
            for phi in unused_phi:
                lhs = self.def_use.defs[phi.lhs]
                rhs = self.def_use.defs[phi.rhs]
                if isinstance(lhs, AssignDef) and isinstance(lhs.site, Assign):
                    unused_assign.add(lhs.site)
                if isinstance(rhs, AssignDef) and isinstance(rhs.site, Assign):
                    unused_assign.add(rhs.site)

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
    def apply(func: FuncDef, def_use: DefineUseAnalysis | None = None) -> FuncDef:
        if not isinstance(func, FuncDef):
            raise TypeError(f'Expected `FuncDef`, got {type(func)} for {func}')
        if def_use is None:
            def_use = DefineUse.analyze(func)
        func = _DeadCodeEliminate(func, def_use).apply()
        SyntaxCheck.check(func)
        return func
