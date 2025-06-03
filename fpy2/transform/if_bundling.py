"""
Transformation pass to pack mutated variables in an if statement
into a single mutated variable.
"""

from ..analysis import DefineUse, DefineUseAnalysis, SyntaxCheck
from ..ast import *
from ..utils import Gensym

from .rename_target import RenameTarget


_Ctx = dict[NamedId, Expr]

class _IfBundlingInstance(DefaultTransformVisitor):
    """Single-use instance of the IfBundling pass."""
    func: FuncDef
    def_use: DefineUseAnalysis
    gensym: Gensym

    def __init__(self, func: FuncDef, def_use: DefineUseAnalysis):
        self.func = func
        self.def_use = def_use
        self.gensym = Gensym(reserved=def_use.names)

    def apply(self) -> FuncDef:
        return self._visit_function(self.func, {})

    def _visit_var(self, e: Var, ctx: _Ctx):
        if e.name in ctx:
            return ctx[e.name]
        else:
            return Var(e.name, e.loc)

    def _visit_if1(self, stmt: If1Stmt, ctx: _Ctx) -> StmtBlock:
        # let x_0, ..., x_N be variables mutated in the if statement body
        # let x_0', ..., x_N', t be fresh variables
        #
        # ```
        # if <cond>:
        #    <body>
        # ```
        # ==>
        # ```
        # t = (x_0, ..., x_N)
        # if <cond>:
        #     x_0', ..., x_N' = t
        #     <body>
        #     t = (x_0', ..., x_N')
        # x_0, ..., x_N = t
        # ```
        # where `<cond'> := [x_0 -> t[0], ..., x_N -> t[N]] <cond>`
        # subsitutes for `x_0, ..., x_N` in the condition.

        # identify variables that were mutated in the body
        defs_in, defs_out = self.def_use.blocks[stmt.body]
        mutated = defs_in.mutated_in(defs_out)

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
            cond_ctx = { var: TupleRef(Var(t, None), (Integer(i, None),), None) for i, var in enumerate(mutated) }
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

            # append the if statement
            s = If1Stmt(cond, body, None)
            stmts.append(s)

            # unpack the tuple after the if
            s = Assign(TupleBinding(mutated, None), None, Var(t, None), None)
            stmts.append(s)

            return StmtBlock(stmts)
        else:
            # no need to apply the transformation
            cond = self._visit_expr(stmt.cond, ctx)
            body, _ = self._visit_block(stmt.body, ctx)
            s = If1Stmt(cond, body, None)
            return StmtBlock([s])

    def _visit_if(self, stmt: IfStmt, ctx: _Ctx) -> StmtBlock:
        # let x_0, ..., x_N be variables mutated in the if statement bodies
        # let y_0, ..., y_N be variables introduced in the if statement bodies
        # let x_0', ..., x_N', t be fresh variables
        #
        # ```
        # if <cond>:
        #    <ift-body>
        # else:
        #    <iff-body>
        # ```
        # ==>
        # ```
        # t = (x_0, ..., x_N)
        # if <cond>:
        #     x_0', ..., x_N' = t
        #     <ift-body>
        #     t = (x_0', ..., x_N', y_0, ..., y_N)
        # else:
        #     x_0', ..., x_N' = t
        #     <iff-body>
        #     t = (x_0', ..., x_N', y_0, ..., y_N)
        # x_0, ..., x_N, y_0, ..., y_N = t
        # ```
        # where `<cond'> := [x_0 -> t[0], ..., x_N -> t[N]] <cond>`
        # subsitutes for `x_0, ..., x_N` in the condition.

        raise NotImplementedError

    def _visit_block(self, block: StmtBlock, ctx: _Ctx):
        stmts: list[Stmt] = []
        for stmt in block.stmts:
            match stmt:
                case If1Stmt():
                    b = self._visit_if1(stmt, ctx)
                    stmts.extend(b.stmts)
                case IfStmt():
                    b = self._visit_if(stmt, ctx)
                    stmts.extend(b.stmts)
                case _:
                    stmt, _ = self._visit_statement(stmt, ctx)
                    stmts.append(stmt)
        return StmtBlock(stmts), ctx


class IfBundling:
    """
    Transformation pass to pack mutated variables in if statements.

    This pass rewrites the AST to pack mutated variables in an if statement
    into a single mutated variable. This ensures a convenient translation
    into more functional languages.
    """

    @staticmethod
    def apply(func: FuncDef) -> FuncDef:
        if not isinstance(func, FuncDef):
            raise SyntaxCheck(f'Expected \'FuncDef\', got {func}')

        def_use = DefineUse.analyze(func)
        ast = _IfBundlingInstance(func, def_use).apply()
        SyntaxCheck.check(ast, ignore_unknown=True)
        return ast
