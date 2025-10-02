"""
Function inlining.
"""

from dataclasses import dataclass

from ..analysis import AssignDef, DefineUse, DefineUseAnalysis, ReachingDefs, SyntaxCheck
from ..ast.fpyast import *
from ..ast.visitor import DefaultTransformVisitor
from ..function import Function
from ..utils import Gensym

from .rename_target import RenameTarget

def _replace_ret(block: StmtBlock, new_var: NamedId):
    last_stmt = block.stmts[-1]
    match last_stmt:
        case ReturnStmt():
            new_stmt = Assign(new_var, None, last_stmt.expr, last_stmt.loc)
            block.stmts[-1] = new_stmt
        case ContextStmt():
            _replace_ret(last_stmt.body, new_var)
        case _:
            raise RuntimeError(f'expected a `return` or `with` statement, got `{last_stmt}`')


@dataclass
class _Ctx:
    stmts: list[Stmt]

    @staticmethod
    def default():
        return _Ctx(stmts=[])


class _FuncInline(DefaultTransformVisitor):
    """Function inline visitor."""

    func: FuncDef
    def_use: DefineUseAnalysis
    gensym: Gensym

    def __init__(self, func: FuncDef, def_use: DefineUseAnalysis):
        self.func = func
        self.def_use = def_use
        self.gensym = Gensym(self.def_use.names())

    def _visit_call(self, e: Call, ctx: _Ctx):
        if not isinstance(e.fn, Function):
            # not calling a function so no inlining
            return super()._visit_call(e, ctx)

        # candidate for inlining
        # ASSUME: single return statement at the end of the function body

        # first, rename all variables in the function body
        reachability = ReachingDefs.analyze(e.fn.ast)
        subst: dict[NamedId, NamedId] = {}
        for d in reachability.defs:
            if isinstance(d, AssignDef) and not d.is_free:
                subst[d.name] = self.gensym.refresh(d.name)
        ast = RenameTarget.apply(e.fn.ast, subst)

        # TODO: check that free variables agree

        # bind the return value to a fresh variable and splice into the current block
        t = self.gensym.fresh('t')
        _replace_ret(ast.body, t)
        if ast.ctx is None:
            # no overriding context
            ctx.stmts.extend(ast.body.stmts)
        else:
            # overriding context
            stmt = ContextStmt(UnderscoreId(), ForeignVal(ast.ctx, None), ast.body, ast.loc)
            ctx.stmts.append(stmt)

        return Var(t, e.loc)


    def _visit_block(self, block: StmtBlock, ctx: None):
        block_ctx = _Ctx.default()
        for stmt in block.stmts:
            stmt, _ = self._visit_statement(stmt, block_ctx)
            block_ctx.stmts.append(stmt)
        b = StmtBlock(block_ctx.stmts)
        return b, None

    def apply(self) -> FuncDef:
        return self._visit_function(self.func, None)


class FuncInline:
    """
    Function inlining.
    """

    @staticmethod
    def apply(
        func: FuncDef, *,
        def_use: DefineUseAnalysis | None = None
    ) -> FuncDef:
        """
        Applies function inlining to `func` returning the transformed function.
        """
        if not isinstance(func, FuncDef):
            raise TypeError(f'expected a \'FuncDef\', got `{func}`')
        if def_use is None:
            def_use = DefineUse.analyze(func)

        inst = _FuncInline(func, def_use)
        SyntaxCheck.check(func, ignore_unknown=True)
        return inst.apply()
