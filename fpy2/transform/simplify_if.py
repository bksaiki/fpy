"""Transformation pass to rewrite if statements to if expressions."""

from ..analysis import DefineUse, DefineUseAnalysis
from ..ast import *
from ..transform import RenameTarget
from ..utils import Gensym

class _SimplifyIfInstance(DefaultAstTransformVisitor):
    """Single-use instance of the SimplifyIf pass."""
    func: FuncDef
    def_use: DefineUseAnalysis
    gensym: Gensym

    def __init__(self, func: FuncDef, def_use: DefineUseAnalysis):
        self.func = func
        self.def_use = def_use
        self.gensym = Gensym(reserved=set(def_use.defs.keys()))

    def apply(self):
        return self._visit_function(self.func, None)

    def _visit_if1(self, stmt: If1Stmt, ctx: None):
        stmts: list[Stmt] = []

        # compile condition
        cond = self._visit_expr(stmt.cond, ctx)

        # generate temporary if needed
        if not isinstance(cond, Var):
            t = self.gensym.fresh('cond')
            s = SimpleAssign(t, cond, BoolTypeAnn(None), None)
            stmts.append(s)
            cond = Var(t, None)

        # compile the body
        body, _ = self._visit_block(stmt.body, ctx)

        # identify variables that were mutated in the body
        defs_in, defs_out = self.def_use.blocks[stmt.body]
        mutated = defs_in.mutated_in(defs_out)

        # rename mutated variables in the body and inline it
        rename = { var: self.gensym.refresh(var) for var in mutated }
        body = RenameTarget.apply_block(body, rename)
        stmts.extend(body.stmts)

        # make if expressions for each mutated variable
        for var in mutated:
            e = IfExpr(cond, Var(rename[var], None), Var(var, None), None)
            s = SimpleAssign(var, e, None, None)
            stmts.append(s)

        return StmtBlock(stmts)


    def _visit_if(self, stmt: IfStmt, ctx: None):
        stmts: list[Stmt] = []

        # compile condition
        cond = self._visit_expr(stmt.cond, ctx)

        # generate temporary if needed
        if not isinstance(cond, Var):
            t = self.gensym.fresh('cond')
            s = SimpleAssign(t, cond, BoolTypeAnn(None), None)
            stmts.append(s)
            cond = Var(t, None)

        # compile the bodies
        ift, _ = self._visit_block(stmt.ift, ctx)
        iff, _ = self._visit_block(stmt.iff, ctx)

        # identify variables that were mutated in each body
        defs_in_ift, defs_out_ift = self.def_use.blocks[stmt.ift]
        defs_in_iff, defs_out_iff = self.def_use.blocks[stmt.iff]
        mutated_ift = defs_in_ift.mutated_in(defs_out_ift)
        mutated_iff = defs_in_iff.mutated_in(defs_out_iff)

        # identify variables that were introduced in the bodies
        # FPy semantics says they must be introduces in both branches
        intros_ift = defs_in_ift.fresh_in(defs_out_ift)
        intros_iff = defs_in_iff.fresh_in(defs_out_iff)
        intros = intros_ift & intros_iff

        # add to "mutated" set (bit of a misnomer)
        mutated_ift.extend(intros)
        mutated_iff.extend(intros)

        # rename mutated variables in each body and inline them
        rename_ift = { var: self.gensym.refresh(var) for var in mutated_ift }
        ift = RenameTarget.apply_block(ift, rename_ift)
        stmts.extend(ift.stmts)

        rename_iff = { var: self.gensym.refresh(var) for var in mutated_iff }
        iff = RenameTarget.apply_block(iff, rename_iff)
        stmts.extend(iff.stmts)

        # make if expressions for each mutated variable
        mutated_uniq: set[NamedId] = set()
        for var in mutated_ift:
            ift_name = rename_ift[var]
            iff_name = rename_iff.get(var, var)
            e = IfExpr(cond, Var(ift_name, None), Var(iff_name, None), None)
            s = SimpleAssign(var, e, None, None)
            stmts.append(s)
            mutated_uniq.add(var)

        for var in mutated_iff:
            if var not in mutated_uniq:
                ift_name = rename_ift.get(var, var)
                iff_name = rename_iff[var]
                e = IfExpr(cond, Var(ift_name, None), Var(iff_name, None), None)
                s = SimpleAssign(var, e, None, None)
                stmts.append(s)
                mutated_uniq.add(var)

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
        ast = _SimplifyIfInstance(func, def_use).apply()
        SyntaxCheck.check(ast)
        return ast
