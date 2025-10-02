"""
Function inlining.
"""

from dataclasses import dataclass
from typing import Iterable

from ..analysis import AssignDef, DefineUse, DefineUseAnalysis, ReachingDefs, SyntaxCheck
from ..ast.fpyast import *
from ..ast.visitor import DefaultTransformVisitor
from ..env import ForeignEnv
from ..function import Function
from ..number import REAL
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
    is_ctx_expr: bool

    @staticmethod
    def default():
        return _Ctx(stmts=[], is_ctx_expr=False)


class _FuncInline(DefaultTransformVisitor):
    """Function inline visitor."""

    func: FuncDef
    def_use: DefineUseAnalysis
    funcs: set[FuncDef] | None

    gensym: Gensym
    free_vars: set[NamedId]
    env: ForeignEnv

    def __init__(self, func: FuncDef, def_use: DefineUseAnalysis, funcs: set[FuncDef] | None):
        self.func = func
        self.def_use = def_use
        self.funcs = funcs
        self.gensym = Gensym(self.def_use.names())
        self.free_vars = set(func.free_vars)
        self.env = func.env.copy()

    def _visit_call(self, e: Call, ctx: _Ctx):
        if not isinstance(e.fn, Function):
            # not calling a function so no inlining
            return super()._visit_call(e, ctx)
        if self.funcs is not None and e.fn not in self.funcs:
            # not a candidate for inlining
            return super()._visit_call(e, ctx)

        # ASSUME: single return statement at the end of the function body

        # first, rename all variables in the function body
        reachability = ReachingDefs.analyze(e.fn.ast)
        subst: dict[NamedId, NamedId] = {}
        for d in reachability.defs:
            if isinstance(d, AssignDef) and not d.is_free:
                subst[d.name] = self.gensym.refresh(d.name)
        ast = RenameTarget.apply(e.fn.ast, subst)

        # merge free variables
        for name in ast.free_vars:
            if str(name) in self.env:
                # already in the environment, check that it is the same
                val = self.env.get(str(name))
                if val != e.fn.env.get(str(name)):
                    raise RuntimeError(f'cannot inline function `{e.fn.name}` due to conflicting free variable `{name}`')
        self.env = self.env.merge(ast.env, keys=map(str, ast.free_vars))
        self.free_vars |= ast.free_vars

        # bind arguments to parameters
        for arg, param in zip(e.args, ast.args):
            arg = self._visit_expr(arg, ctx)
            name = subst.get(param.name, param.name)
            ctx.stmts.append(Assign(name, param.type, arg, e.loc))

        # bind the return value to a fresh variable and splice into the current block
        t = self.gensym.fresh('t')
        _replace_ret(ast.body, t)
        if ast.ctx is not None:
            # overriding context
            stmt = ContextStmt(UnderscoreId(), ForeignVal(ast.ctx, None), ast.body, ast.loc)
            ctx.stmts.append(stmt)
        elif ctx.is_ctx_expr:
            # overriding context must be `RealContext` since
            # we are in a context expression `with e: ...`
            stmt = ContextStmt(UnderscoreId(), ForeignVal(REAL, None), ast.body, ast.loc)
            ctx.stmts.append(stmt)
        else:
            # no overriding context
            ctx.stmts.extend(ast.body.stmts)

        # return the bound value
        return Var(t, e.loc)


    def _visit_context(self, stmt: ContextStmt, ctx: _Ctx):
        ctx_e = self._visit_expr(stmt.ctx, _Ctx(ctx.stmts, True))
        body, _ = self._visit_block(stmt.body, None)
        s = ContextStmt(stmt.target, ctx_e, body, stmt.loc)
        return s, None

    def _visit_block(self, block: StmtBlock, ctx: _Ctx | None):
        block_ctx = _Ctx.default()
        for stmt in block.stmts:
            stmt, _ = self._visit_statement(stmt, block_ctx)
            block_ctx.stmts.append(stmt)
        b = StmtBlock(block_ctx.stmts)
        return b, None

    def _visit_function(self, func: FuncDef, ctx: None):
        body, _ = self._visit_block(func.body, None)
        f = FuncDef(func.name, func.args, self.free_vars, func.ctx, body, func.spec, func.meta, self.env, loc=func.loc)
        return f

    def apply(self) -> FuncDef:
        return self._visit_function(self.func, None)


class FuncInline:
    """
    Function inlining.
    """

    @staticmethod
    def apply(
        func: FuncDef, *,
        def_use: DefineUseAnalysis | None = None,
        funcs: Iterable[FuncDef] | None = None
    ) -> FuncDef:
        """
        Applies function inlining to `func` returning the transformed function.
        """
        if not isinstance(func, FuncDef):
            raise TypeError(f'expected a \'FuncDef\', got `{func}`')
        if def_use is None:
            def_use = DefineUse.analyze(func)

        if funcs is not None:
            funcs = set(funcs)

        inst = _FuncInline(func, def_use, funcs)
        SyntaxCheck.check(func, ignore_unknown=True)
        return inst.apply()
