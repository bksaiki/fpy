"""
Transformation pass to bundle updated variables in while loops
into a single variable.
"""

from typing import Optional

from .define_use import DefineUse
from .verify import VerifyIR
from ..ir import *
from ..utils import Gensym

_CtxType = dict[str, Expr]
"""
Visitor method context type.
For statements, this context maps variables to a fresh variable.
For expressions, this context maps variables to an expression.
"""

class _WhileBundlingInstance(DefaultTransformVisitor):
    """Single-use instance of the WhileBundling pass."""
    func: FunctionDef
    gensym: Gensym

    def __init__(self, func: FunctionDef, names: set[str]):
        self.func = func
        self.gensym = Gensym(*names)

    def apply(self) -> FunctionDef:
        return self._visit(self.func, {})
    
    def _visit_var(self, e: Var, ctx: _CtxType):
        if e.name in ctx:
            return ctx[e.name]
        else:
            return Var(e.name)

    def _visit_phi(self, phi, ctx):
        if phi.lhs in ctx:
            lhs = ctx[phi.lhs]
            if not isinstance(lhs, Var):
                raise NotImplementedError('expected a Var', lhs)
            return PhiNode(phi.name, lhs.name, phi.rhs, phi.ty)
        else:
            return PhiNode(phi.name, phi.lhs, phi.rhs, phi.ty)

    def _visit_while_stmt(self, stmt: WhileStmt, ctx: _CtxType):
        if len(stmt.phis) <= 1:
            return super()._visit_while_stmt(stmt, set(ctx))
        else:
            # unified phi variable
            phi_init = self.gensym.fresh('t')
            phi_name = self.gensym.fresh('t')
            phi_update = self.gensym.fresh('t')
            phi_node = PhiNode(phi_name, phi_init, phi_update, AnyType())
            # decompose current phi variables
            phis = [self._visit_phi(phi, ctx) for phi in stmt.phis]
            phi_names = [phi.name for phi in phis]
            phi_inits = [Var(phi.lhs) for phi in phis]
            phi_updates = [Var(phi.rhs) for phi in phis]
            phi_renamed = [self.gensym.fresh('t') for _ in phis]
            # compile condition with new context
            cond_ctx = ctx.copy()
            for i, phi in enumerate(phis):
                cond_ctx[phi.name] = RefExpr(Var(phi_name), Integer(i))
            cond: Expr = self._visit(stmt.cond, cond_ctx)
            # compile body with new context
            body_ctx = ctx.copy()
            for phi, rename in zip(phis, phi_renamed):
                body_ctx[phi.name] = Var(rename)
            body: Block = self._visit(stmt.body, body_ctx)
            # construct the block
            init_stmt = VarAssign(phi_init, AnyType(), TupleExpr(*phi_inits))
            unpack_stmt = TupleAssign(TupleBinding(phi_renamed), AnyType(), Var(phi_name))
            update_stmt = VarAssign(phi_update, AnyType(), TupleExpr(*phi_updates))
            while_stmt = WhileStmt(cond, Block([unpack_stmt, *body.stmts, update_stmt]), [phi_node])
            unpack_stmt = TupleAssign(TupleBinding(phi_names), AnyType(), Var(phi_name))
            return Block([init_stmt, while_stmt, unpack_stmt])

    def _visit_block(self, block: Block, ctx: _CtxType):
        stmts: list[Stmt] = []
        for stmt in block.stmts:
            if isinstance(stmt, WhileStmt):
                stmt_or_block = self._visit_while_stmt(stmt, ctx.copy())
                if isinstance(stmt_or_block, Stmt):
                    stmts.append(stmt_or_block)
                elif isinstance(stmt_or_block, Block):
                    stmts.extend(stmt_or_block.stmts)
                else:
                    raise NotImplementedError('unexpected', stmt_or_block)
            else:
                stmts.append(self._visit(stmt, ctx.copy()))
        return Block(stmts)

class WhileBundling:
    """
    Transformation pass to bundle updated variables in while loops.

    This pass rewrites the IR to bundle updated variables in while loops
    into a single variable. This transformation ensures there is only
    one phi node per while loop.
    """

    @staticmethod
    def apply(func: FunctionDef, names: Optional[set[str]] = None) -> FunctionDef:
        if names is None:
            uses = DefineUse.analyze(func)
            names = set(uses.keys())
        ir = _WhileBundlingInstance(func, names).apply()
        VerifyIR.check(ir)
        return ir
