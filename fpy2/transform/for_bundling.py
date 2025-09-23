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

    def __init__(self, func: FuncDef, def_use: DefineUseAnalysis):
        self.func = func
        self.def_use = def_use
        self.gensym = Gensym(reserved=def_use.names())

    def apply(self) -> FuncDef:
        return self._visit_function(self.func, None)

    def _visit_tuple_binding(self, binding: TupleBinding, rename: dict[NamedId, NamedId] | None = None):
        new_vars: list[Id | TupleBinding] = []
        for var in binding:
            match var:
                case NamedId():
                    if rename is None:
                        new_vars.append(var)
                    else:
                        new_vars.append(rename.get(var, var))
                case UnderscoreId():
                    new_vars.append(var)
                case TupleBinding():
                    new_vars.append(self._visit_tuple_binding(var, rename))
                case _:
                    raise NotImplementedError(f'unreachable {var}')
        return TupleBinding(new_vars, binding.loc)


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
        # In the case that `x_i` is re-defined in `<target>`,
        # replace `x_i` with `x_i'` in the target and
        # generate `_` in place of `x_i'` in the body.
        # 

        # identify variables that were mutated in the body
        mutated = self.def_use.mutated_in(stmt.body)
        if len(mutated) > 1:
            # need to apply the transformation
            stmts: list[Stmt] = []

            # fresh variable to hold tuple of mutated variables
            t = self.gensym.fresh('t')

            # fresh variables for each mutated variable
            rename = { var: self.gensym.refresh(var) for var in mutated }

            # extract target names
            match stmt.target:
                case NamedId():
                    target_names = { stmt.target }
                case UnderscoreId():
                    target_names = set()
                case TupleBinding():
                    target_names = stmt.target.names()
                case _:
                    raise RuntimeError('unreachable', stmt.target)

            # create a tuple of mutated variables
            e = TupleExpr([Var(var, None) for var in mutated], None)
            s: Stmt = Assign(t, None, e, None)
            stmts.append(s)

            # transform target
            match stmt.target:
                case Id():
                    target = stmt.target
                case TupleBinding():
                    target = self._visit_tuple_binding(stmt.target, rename)
                case _:
                    raise RuntimeError('unreachable', stmt.target)

            # compile iterable and body
            iterable = self._visit_expr(stmt.iterable, None)
            body, _ = self._visit_block(stmt.body, None)
            body = RenameTarget.apply_block(body, rename)

            # unpack the tuple at the start of the body
            # replace any variable in the target with `_`
            binding = TupleBinding([UnderscoreId() if var in target_names else rename[var] for var in mutated], None)
            s = Assign(binding, None, Var(t, None), None)
            body.stmts.insert(0, s)

            # repack the tuple at the end of the body
            s = Assign(t, None, TupleExpr([Var(rename[v], None) for v in mutated], None), None)
            body.stmts.append(s)

            # append the for statement
            s = ForStmt(target, iterable, body, None)
            stmts.append(s)

            # unpack the tuple after the loop
            s = Assign(TupleBinding(mutated, None), None, Var(t, None), None)
            stmts.append(s)

            return StmtBlock(stmts)
        else:
            # transformation is not needed
            match stmt.target:
                case Id():
                    target = stmt.target
                case TupleBinding():
                    target = self._visit_tuple_binding(stmt.target, None)
                case _:
                    raise RuntimeError('unreachable', stmt.target)

            iterable = self._visit_expr(stmt.iterable, None)
            body, _ = self._visit_block(stmt.body, None)
            s = ForStmt(target, iterable, body, None)
            return StmtBlock([s])


    def _visit_block(self, block: StmtBlock, ctx: None):
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
            raise SyntaxCheck(f'Expected \'FuncDef\', got {func}')

        def_use = DefineUse.analyze(func)
        ast = _ForBundlingInstance(func, def_use).apply()
        SyntaxCheck.check(ast, ignore_unknown=True)
        return ast
