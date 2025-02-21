"""Transformation pass to rewrite if statements to if expressions."""

from typing import Optional

from .define_use import DefineUse
from .verify import VerifyIR

from ..ir import *
from ..utils import Gensym

class _SimplifyIfInstance(DefaultTransformVisitor):
    """Single-use instance of the SimplifyIf pass."""
    func: FunctionDef
    gensym: Gensym

    def __init__(self, func: FunctionDef, names: set[NamedId]):
        self.func = func
        self.gensym = Gensym(reserved=names)

    def apply(self):
        return self._visit_function(self.func, None)

    def _visit_if1_stmt(self, stmt: If1Stmt, ctx: None):
        stmts: list[Stmt] = []
        # compile condition
        cond = self._visit_expr(stmt.cond, ctx)
        # generate temporary if needed
        if not isinstance(cond, Var):
            t = self.gensym.fresh('cond')
            stmts.append(VarAssign(t, BoolType(), cond))
            cond = Var(t)
        # inline body
        body, _ = self._visit_block(stmt.body, ctx)
        stmts.extend(body.stmts)
        # convert phi nodes into if expressions
        for phi in stmt.phis:
            ife = IfExpr(cond, Var(phi.rhs), Var(phi.lhs))
            stmts.append(VarAssign(phi.name, AnyType(), ife))
        return Block(stmts)

    def _visit_if_stmt(self, stmt: IfStmt, ctx: None):
        stmts: list[Stmt] = []
        # compile condition
        cond = self._visit_expr(stmt.cond, ctx)
        # generate temporary if needed
        if not isinstance(cond, Var):
            t = self.gensym.fresh('cond')
            stmts.append(VarAssign(t, BoolType(), cond))
            cond = Var(t)
        # inline if-true block
        ift, _ = self._visit_block(stmt.ift, ctx)
        stmts.extend(ift.stmts)
        # inline if-false block
        iff, _ = self._visit_block(stmt.iff, ctx)
        stmts.extend(iff.stmts)
        # convert phi nodes into if expressions
        for phi in stmt.phis:
            ife = IfExpr(cond, Var(phi.lhs), Var(phi.rhs))
            stmts.append(VarAssign(phi.name, AnyType(), ife))
        return Block(stmts)

    def _visit_block(self, block: Block, ctx: None):
        stmts: list[Stmt] = []
        for stmt in block.stmts:
            match stmt:
                case If1Stmt():
                    if1_block = self._visit_if1_stmt(stmt, ctx)
                    stmts.extend(if1_block.stmts)
                case IfStmt():
                    if_block = self._visit_if_stmt(stmt, ctx)
                    stmts.extend(if_block.stmts)
                case _:
                    stmt, _ = self._visit_statement(stmt, ctx)
                    stmts.append(stmt)
        return Block(stmts), None


class SimplifyIf:
    """
    Control flow simplifification: transform if statements to if expressions.

    This transformation rewrites a block of the form:
    ```
    if <cond>
        S1 ...
    else:
        S2 ...
    S3 ...
    ```
    to an equivalent block using if expressions:
    ```
    t = <cond>
    S1 ...
    S2 ...
    x_i = x_{i, S1} if t else x_{i, S2}
    S3 ...
    ```
    where `x_i = phi(x_{i, S1}, x_{i, S2})` is a phi node associated
    with the if-statement and `t` is a free variable.
    """

    @staticmethod
    def apply(func: FunctionDef, names: Optional[set[NamedId]] = None):
        if names is None:
            uses = DefineUse.analyze(func)
            names = (uses.keys())
        ir = _SimplifyIfInstance(func, names).apply()
        VerifyIR.check(ir)
        return ir
