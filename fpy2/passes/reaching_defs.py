"""
This module implements a reaching definitions analysis pass for the IR.
"""

from dataclasses import dataclass

from ..ir import *

@dataclass
class Reach:
    reach_in: set[NamedId]
    reach_out: set[NamedId]
    kill_out: set[NamedId]

_BlockCtx = set[NamedId]
_StmtCtx = tuple[set[NamedId], set[NamedId]]
_RetType = tuple[set[NamedId], set[NamedId]]

class _ReachingDefsInstance(DefaultVisitor):
    """Single-use instance of the reaching definitions analysis."""
    func: FunctionDef
    reaches: dict[Block, Reach]

    def __init__(self, func: FunctionDef):
        self.func = func
        self.reaches = {}

    def analyze(self):
        self._visit_function(self.func, None)
        return self.reaches

    def _visit_var_assign(self, stmt: VarAssign, ctx: _StmtCtx) -> _RetType:
        defs_in, kill_in = ctx
        defs_out = { *defs_in, stmt.var } if isinstance(stmt.var, NamedId) else defs_in
        kill_out = kill_in | (defs_in & { stmt.var })
        return defs_out, kill_out

    def _visit_tuple_assign(self, stmt: TupleAssign, ctx: _StmtCtx) -> _RetType:
        defs_in, kill_in = ctx
        gen = set(stmt.binding.names())
        defs_out = defs_in | gen
        kill_out = kill_in | (defs_in & gen)
        return defs_out, kill_out

    def _visit_ref_assign(self, stmt: RefAssign, ctx: _StmtCtx) -> _RetType:
        return ctx

    def _visit_if1_stmt(self, stmt: If1Stmt, ctx: _StmtCtx) -> _RetType:
        defs_in, kill_in = ctx
        _, block_kill = self._visit_block(stmt.body, defs_in)

        defs_out = defs_in.copy()
        kill_out = kill_in | (defs_in & block_kill)
        return defs_out, kill_out

    def _visit_if_stmt(self, stmt: IfStmt, ctx: _StmtCtx) -> _RetType:
        defs_in, kill_in = ctx
        ift_defs, ift_kill = self._visit_block(stmt.ift, defs_in)
        iff_defs, iff_kill = self._visit_block(stmt.iff, defs_in)

        defs_out = ift_defs & iff_defs
        kill_out = kill_in | (defs_in & (ift_kill | iff_kill))
        return defs_out, kill_out

    def _visit_while_stmt(self, stmt: WhileStmt, ctx: _StmtCtx) -> _RetType:
        defs_in, kill_in = ctx
        _, block_kill = self._visit_block(stmt.body, defs_in)

        defs_out = defs_in.copy()
        kill_out = kill_in | (defs_in & block_kill)
        return defs_out, kill_out

    def _visit_for_stmt(self, stmt: ForStmt, ctx: _StmtCtx) -> _RetType:
        defs_in, kill_in = ctx
        defs = { *defs_in, stmt.var } if isinstance(stmt.var, NamedId) else defs_in
        _, block_kill = self._visit_block(stmt.body, defs)

        defs_out = defs.copy()
        kill_out = kill_in | (defs & block_kill)
        return defs_out, kill_out

    def _visit_context(self, stmt: ContextStmt, ctx: _StmtCtx) -> _RetType:
        # TODO: handle `stmt.name`
        defs_in, kill_in = ctx
        _, block_kill = self._visit_block(stmt.body, defs_in.copy())

        defs_out = defs_in.copy()
        kill_out = kill_in | (defs_in & block_kill)
        return defs_out, kill_out

    def _visit_assert(self, stmt: AssertStmt, ctx: _StmtCtx) -> _RetType:
        return ctx

    def _visit_return(self, stmt: Return, ctx: _StmtCtx) -> _RetType:
        return ctx

    def _visit_block(self, block: Block, ctx: _BlockCtx):
        reach_in = ctx.copy()

        defs: set[NamedId] = reach_in.copy()
        kill: set[NamedId] = set()
        for stmt in block.stmts:
            defs, kill = self._visit_statement(stmt, (defs, kill))

        reach_out = defs | (reach_in - kill)
        self.reaches[block] = Reach(reach_in, reach_out, kill)
        return defs, kill

    def _visit_function(self, func: FunctionDef, _: None):
        ctx: set[NamedId] = set()
        for arg in func.args:
            if isinstance(arg.name, NamedId):
                ctx.add(arg.name)
        self._visit_block(func.body, ctx)

    # override to get typing hint
    def _visit_statement(self, stmt, ctx: _StmtCtx) -> _RetType:
        return super()._visit_statement(stmt, ctx)

class ReachingDefs:
    """
    Reaching definitions analysis for the FPy IR.
    """

    analysis_name = 'reaching_defs'

    @staticmethod
    def analyze(func: FunctionDef) -> dict[Block, Reach]:
        return _ReachingDefsInstance(func).analyze()
