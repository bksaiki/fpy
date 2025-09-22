"""
Transformation pass to bundle updated variables in while loops
into a single variable.
"""

from ..analysis import DefineUse, DefineUseAnalysis, SyntaxCheck
from ..ast import *
from ..utils import Gensym

from .rename_target import RenameTarget


_Ctx = dict[NamedId, Expr]

class _WhileBundlingInstance(DefaultTransformVisitor):
    """Single-use instance of the WhileBundling pass."""
    func: FuncDef
    def_use: DefineUseAnalysis
    gensym: Gensym

    def __init__(self, func: FuncDef, def_use: DefineUseAnalysis):
        self.func = func
        self.def_use = def_use
        self.gensym = Gensym(reserved=def_use.names())

    def apply(self) -> FuncDef:
        return self._visit_function(self.func, {})

    def _visit_var(self, e: Var, ctx: _Ctx):
        if e.name in ctx:
            return ctx[e.name]
        else:
            return Var(e.name, e.loc)

    def _visit_while(self, stmt: WhileStmt, ctx: _Ctx) -> StmtBlock:
        # let x_0, ..., x_N be variables mutated in the while loop
        # let x_0', ..., x_N', t be fresh variables
        #
        # The transformation is as follows:
        # ```
        # while <cond>:
        #    ...
        # ```
        # ==>
        # ```
        # t = (x_0, ..., x_N)
        # while <cond'>:
        #     x_0', ..., x_N' = t
        #     ...
        #     t = (x_0', ..., x_N')
        # x_0, ..., x_N = t
        # ```
        # where `<cond'> := [x_0 -> t[0], ..., x_N -> t[N]] <cond>`
        # subsitutes for `x_0, ..., x_N` in the condition.

        # identify variables that were mutated in the body
        mutated = self.def_use.mutated_in(stmt.body)
        if len(mutated) > 1:
            # need to apply the transformation
            stmts: list[Stmt] = []

            # fresh variable to hold tuple of mutated variables
            t = self.gensym.fresh('t')

            # fresh variables for each mutated variable
            rename = { var: self.gensym.refresh(var) for var in mutated }

            # create a tuple of mutated variables
            s: Stmt = Assign(t, None, TupleExpr([Var(var, None) for var in mutated], None), None)
            stmts.append(s)

            # apply substitution to the condition
            cond_ctx = { var: ListRef(Var(t, None), Integer(i, None), None) for i, var in enumerate(mutated) }
            cond = self._visit_expr(stmt.cond, cond_ctx)

            # compile the body and apply the renaming
            body, _ = self._visit_block(stmt.body, ctx)
            body = RenameTarget.apply_block(body, rename)

            # unpack the tuple at the start of the body
            s = Assign(TupleBinding([rename[v] for v in mutated], None), None, Var(t, None), None)
            body.stmts.insert(0, s)

            # repack the tuple at the end of the body
            s = Assign(t, None, TupleExpr([Var(rename[v], None) for v in mutated], None), None)
            body.stmts.append(s)

            # append the while statement
            s = WhileStmt(cond, body, None)
            stmts.append(s)

            # unpack the tuple after the loop
            s = Assign(TupleBinding(mutated, None), None, Var(t, None), None)
            stmts.append(s)

            return StmtBlock(stmts)
        else:
            # transformation is not needed
            cond = self._visit_expr(stmt.cond, ctx)
            body, _ = self._visit_block(stmt.body, ctx)
            s = WhileStmt(cond, body, None)
            return StmtBlock([s])


    def _visit_block(self, block: StmtBlock, ctx: _Ctx):
        stmts: list[Stmt] = []
        for stmt in block.stmts:
            if isinstance(stmt, WhileStmt):
                b = self._visit_while(stmt, ctx)
                stmts.extend(b.stmts)
            else:
                stmt, _ = self._visit_statement(stmt, ctx)
                stmts.append(stmt)
        return StmtBlock(stmts), ctx


class WhileBundling:
    """
    Transformation pass to bundle updated variables in while loops.

    This pass rewrites the IR to bundle updated variables in while loops
    into a single variable. This transformation ensures there is only
    one phi node per while loop.
    """

    @staticmethod
    def apply(func: FuncDef) -> FuncDef:
        if not isinstance(func, FuncDef):
            raise SyntaxCheck(f'Expected \'FuncDef\', got {func}')

        def_use = DefineUse.analyze(func)
        func = _WhileBundlingInstance(func, def_use).apply()
        SyntaxCheck.check(func, ignore_unknown=True)
        return func
