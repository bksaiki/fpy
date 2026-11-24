"""
Transformation pass to bundle updated variables in for loops
into a single variable.
"""

from ..analysis import DefineUse, DefineUseAnalysis, SyntaxCheck
from ..ast import *
from ..utils import Gensym
from .rename_target import RenameTarget


class _ForBundlingInstance(DefaultTransformVisitor):
    """Single-use instance of the ForBundling pass."""
    func: FuncDef
    def_use: DefineUseAnalysis
    gensym: Gensym

    def __init__(self, func: FuncDef, def_use: DefineUseAnalysis) -> None:
        self.func = func
        self.def_use = def_use
        self.gensym = Gensym(reserved=def_use.names())

    def apply(self) -> FuncDef:
        return self._visit_function(self.func, None)

    def _visit_for(self, stmt: ForStmt, ctx: None) -> StmtBlock:
        # let x_0, ..., x_N be variables mutated in the for loop
        # let x_0', ..., x_N', t be fresh variables
        #
        # The transformation is as follows:
        # ```
        # for <target> in <iterable>:
        #    ...
        # ```
        # ==>
        # ```
        # t = (x_0, ..., x_N)
        # for <target> in <iterable>:
        #     x_0', ..., x_N' = t
        #     ...
        #     t = (x_0', ..., x_N')
        # x_0, ..., x_N = t
        # ```
        #
        # The target is rebound every iteration and is scoped to the loop, so
        # it is never carried, even where the body reassigns it.

        # sorted: the order decides the tuple's element order and each
        # name's index in it, so a set would make the emitted program
        # depend on hash order and differ run to run
        mutated = sorted(self.def_use.mutated_in(stmt.body) - stmt.target.names())
        if len(mutated) > 1:
            # need to apply the transformation
            stmts: list[Stmt] = []

            # fresh variable to hold tuple of mutated variables
            t = self.gensym.fresh('t')

            # fresh variables for each mutated variable
            rename = { var: self.gensym.refresh(var) for var in mutated }

            # create a tuple of mutated variables
            e = TupleExpr([Var(var, None) for var in mutated], None)
            s: Stmt = Assign(t, None, e, None)
            stmts.append(s)

            # compile iterable and body
            iterable = self._visit_expr(stmt.iterable, None)
            body, _ = self._visit_block(stmt.body, None)
            body = RenameTarget.apply_block(body, rename)

            # unpack the tuple at the start of the body
            binding = TupleBinding([rename[var] for var in mutated], None)
            s = Assign(binding, None, Var(t, None), None)
            body.stmts.insert(0, s)

            # repack the tuple at the end of the body
            s = Assign(t, None, TupleExpr([Var(rename[v], None) for v in mutated], None), None)
            body.stmts.append(s)

            # append the for statement
            s = ForStmt(stmt.target, iterable, body, None)
            stmts.append(s)

            # unpack the tuple after the loop
            s = Assign(TupleBinding(mutated, None), None, Var(t, None), None)
            stmts.append(s)

            return StmtBlock(stmts)
        else:
            # transformation is not needed
            s, _ = super()._visit_for(stmt, None)
            return StmtBlock([s])


    def _visit_block(self, block: StmtBlock, ctx: None) -> tuple[StmtBlock, None]:
        stmts: list[Stmt] = []
        for stmt in block.stmts:
            if isinstance(stmt, ForStmt):
                b = self._visit_for(stmt, None)
                stmts.extend(b.stmts)
            else:
                stmt, _ = self._visit_statement(stmt, None)
                stmts.append(stmt)
        return StmtBlock(stmts), None


class ForBundling:
    """
    Transformation pass to bundle updated variables in for loops.

    This pass rewrites the IR to bundle updated variables in for loops
    into a single variable. This transformation ensures there is only
    one phi node per while loop.
    """

    @staticmethod
    def apply(func: FuncDef) -> FuncDef:
        if not isinstance(func, FuncDef):
            raise TypeError(f'Expected \'FuncDef\', got {func}')

        def_use = DefineUse.analyze(func)
        ast = _ForBundlingInstance(func, def_use).apply()
        SyntaxCheck.check(ast, ignore_unknown=True)
        return ast
