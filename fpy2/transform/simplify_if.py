"""Transformation pass to rewrite if statements to if expressions."""

from ..analysis import DefineUse, DefineUseAnalysis, SyntaxCheck
from ..ast import *
from ..utils import Gensym

from .copy_propagate import CopyPropagate
from .rename_target import RenameTarget


class _SimplifyIfInstance(DefaultTransformVisitor):
    """Single-use instance of the SimplifyIf pass."""
    func: FuncDef
    def_use: DefineUseAnalysis
    gensym: Gensym

    def __init__(self, func: FuncDef, def_use: DefineUseAnalysis):
        self.func = func
        self.def_use = def_use
        self.gensym = Gensym(reserved=def_use.names())

    def apply(self):
        func = self._visit_function(self.func, None)
        return func, self.gensym.generated

    def _visit_if1(self, stmt: If1Stmt, ctx: None):
        stmts: list[Stmt] = []

        # compile condition
        cond = self._visit_expr(stmt.cond, ctx)

        # generate temporary if needed
        if not isinstance(cond, Var):
            t = self.gensym.fresh('cond')
            s = Assign(t, BoolTypeAnn(None), cond, None)
            stmts.append(s)
            cond = Var(t, None)

        # compile the body
        body, _ = self._visit_block(stmt.body, ctx)

        # identify variables that were mutated in the body
        mutated = self.def_use.mutated_in(stmt.body)

        # rename mutated variables in the body
        rename = { var: self.gensym.refresh(var) for var in mutated }
        body = RenameTarget.apply_block(body, rename)

        # generate assignments and inline the body
        for var in mutated:
            t = rename[var]
            s = Assign(t, None, Var(var, None), None)
            stmts.append(s)
        stmts.extend(body.stmts)

        # make if expressions for each mutated variable
        for var in mutated:
            e = IfExpr(cond, Var(rename[var], None), Var(var, None), None)
            s = Assign(var, None, e, None)
            stmts.append(s)

        return StmtBlock(stmts)


    def _visit_if(self, stmt: IfStmt, ctx: None):
        stmts: list[Stmt] = []

        # compile condition
        cond = self._visit_expr(stmt.cond, ctx)

        # generate temporary if needed
        if not isinstance(cond, Var):
            t = self.gensym.fresh('cond')
            s = Assign(t, BoolTypeAnn(None), cond, None)
            stmts.append(s)
            cond = Var(t, None)

        # compile the bodies
        ift, _ = self._visit_block(stmt.ift, ctx)
        iff, _ = self._visit_block(stmt.iff, ctx)

        # identify variables that were mutated in each body
        mutated_ift = self.def_use.mutated_in(stmt.ift)
        mutated_iff = self.def_use.mutated_in(stmt.iff)

        # identify variables that were introduced in the bodies
        # FPy semantics says they must be introduced in both branches
        intros_ift = self.def_use.introed_in(stmt.ift)
        intros_iff = self.def_use.introed_in(stmt.iff)
        intros = sorted(intros_ift & intros_iff) # intersection of fresh variables

        # combine sets
        mutated_or_new_ift = sorted(mutated_ift)
        mutated_or_new_iff = sorted(mutated_iff)
        mutated_or_new_ift.extend(intros)
        mutated_or_new_iff.extend(intros)

        # rename mutated variables in each body, generate assignments, and inline
        rename_ift = { var: self.gensym.refresh(var) for var in mutated_or_new_ift }
        rename_iff = { var: self.gensym.refresh(var) for var in mutated_or_new_iff }

        ift = RenameTarget.apply_block(ift, rename_ift)
        iff = RenameTarget.apply_block(iff, rename_iff)

        for var in mutated_ift:
            t = rename_ift[var]
            s = Assign(t, None, Var(var, None), None)
            stmts.append(s)
        stmts.extend(ift.stmts)

        for var in mutated_iff:
            t = rename_iff[var]
            s = Assign(t, None, Var(var, None), None)
            stmts.append(s)
        stmts.extend(iff.stmts)

        # make if expressions for each mutated or introduced variable
        unique: set[NamedId] = set()
        for var in mutated_or_new_ift:
            if var not in unique:
                ift_name = rename_ift.get(var, var)
                iff_name = rename_iff.get(var, var)
                e = IfExpr(cond, Var(ift_name, None), Var(iff_name, None), None)
                s = Assign(var, None, e, None)
                stmts.append(s)
                unique.add(var)

        return StmtBlock(stmts)


    def _visit_block(self, block: StmtBlock, ctx: None):
        stmts: list[Stmt] = []
        for stmt in block.stmts:
            match stmt:
                case If1Stmt():
                    if1_block = self._visit_if1(stmt, ctx)
                    stmts.extend(if1_block.stmts)
                case IfStmt():
                    if_block = self._visit_if(stmt, ctx)
                    stmts.extend(if_block.stmts)
                case _:
                    stmt, _ = self._visit_statement(stmt, ctx)
                    stmts.append(stmt)
        return StmtBlock(stmts), None


#
# This transformation rewrites a block of the form:
# ```
# if <cond>
#     S1 ...
# else:
#     S2 ...
# S3 ...
# ```
# to an equivalent block using if expressions:
# ```
# t = <cond>
# S1 ...
# S2 ...
# x_i = x_{i, S1} if t else x_{i, S2}
# S3 ...
# ```
# where `x_i` is a phi node merging `phi(x_{i, S1}` and `x_{i, S2})`
# that is associated with the if-statement and `t` is a free variable.

class SimplifyIf:
    """
    Control flow simplification:

    Transforms if statements into if expressions.
    The inner block is hoisted into the outer block and each
    phi variable is made explicit with an if expression.
    """

    @staticmethod
    def apply(func: FuncDef):
        def_use = DefineUse.analyze(func)
        ast, new_ids = _SimplifyIfInstance(func, def_use).apply()
        ast = CopyPropagate.apply(ast, names=new_ids)
        SyntaxCheck.check(ast, ignore_unknown=True)
        return ast
