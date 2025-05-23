"""Live variable analysis on the FPy IR"""

from typing import TypeAlias

from ..ir import *

_RetType: TypeAlias = set[NamedId]

class _LiveVars(ReduceVisitor):
    """Single-instance of the live variable analysis."""
    e: FuncDef | StmtBlock | Stmt | Expr

    def __init__(self, e: FuncDef | StmtBlock | Stmt | Expr):
        self.e = e

    def analyze(self):
        """Runs live-variable analysis on `self.e`."""
        match self.e:
            case FuncDef():
                return self._visit_function(self.e, None)
            case StmtBlock():
                return self._visit_block(self.e, set())
            case Stmt():
                return self._visit_statement(self.e, set())
            case Expr():
                return self._visit_expr(self.e, None)
            case _:
                raise RuntimeError('unreachable')

    def _visit_var(self, e: Var, ctx: None) -> _RetType:
        return { e.name }

    def _visit_bool(self, e: BoolVal, ctx: None) -> _RetType:
        return set()

    def _visit_foreign(self, e: ForeignVal, ctx: None) -> _RetType:
        return set()

    def _visit_decnum(self, e: Decnum, ctx: None) -> _RetType:
        return set()

    def _visit_hexnum(self, e: Hexnum, ctx: None) -> _RetType:
        return set()

    def _visit_integer(self, e: Integer, ctx: None) -> _RetType:
        return set()

    def _visit_rational(self, e: Rational, ctx: None) -> _RetType:
        return set()

    def _visit_constant(self, e: Constant, ctx: None) -> _RetType:
        return set()

    def _visit_digits(self, e: Digits, ctx: None) -> _RetType:
        return set()

    def _visit_unknown(self, e: UnknownCall, ctx: None) -> _RetType:
        fvs = set()
        for arg in e.children:
            fvs.update(self._visit_expr(arg, ctx))
        return fvs

    def _visit_nary_expr(self, e: NaryExpr, ctx: None) -> _RetType:
        fvs = set()
        for arg in e.children:
            fvs.update(self._visit_expr(arg, ctx))
        return fvs

    def _visit_compare(self, e: Compare, ctx: None) -> _RetType:
        fvs = set()
        for arg in e.children:
            fvs.update(self._visit_expr(arg, ctx))
        return fvs

    def _visit_tuple_expr(self, e: TupleExpr, ctx: None) -> _RetType:
        fvs = set()
        for arg in e.children:
            fvs.update(self._visit_expr(arg, ctx))
        return fvs

    def _visit_tuple_ref(self, e: TupleRef, ctx: None) -> _RetType:
        fvs = self._visit_expr(e.value, ctx)
        for arg in e.slices:
            fvs.update(self._visit_expr(arg, ctx))
        return fvs

    def _visit_tuple_set(self, e: TupleSet, ctx: None) -> _RetType:
        fvs = self._visit_expr(e.array, ctx)
        for arg in e.slices:
            fvs.update(self._visit_expr(arg, ctx))
        fvs.update(self._visit_expr(e.value, ctx))
        return fvs

    def _visit_comp_expr(self, e: CompExpr, ctx: None) -> _RetType:
        fvs = set()
        for iterable in e.iterables:
            fvs.update(self._visit_expr(iterable, ctx))
        fvs.update(self._visit_expr(e.elt, ctx))
        return fvs

    def _visit_if_expr(self, e: IfExpr, ctx: None) -> _RetType:
        fvs = self._visit_expr(e.cond, ctx)
        fvs.update(self._visit_expr(e.ift, ctx))
        fvs.update(self._visit_expr(e.iff, ctx))
        return fvs

    def _visit_simple_assign(self, stmt: SimpleAssign, ctx: _RetType) -> _RetType:
        fvs = ctx | self._visit_expr(stmt.expr, None)
        if isinstance(stmt.var, NamedId):
            fvs.discard(stmt.var)
        return fvs

    def _visit_tuple_unpack(self, stmt: TupleUnpack, ctx: _RetType) -> _RetType:
        fvs = ctx | self._visit_expr(stmt.expr, None)
        for var in stmt.binding.names():
            if isinstance(var, NamedId):
                fvs.discard(var)
        return fvs

    def _visit_index_assign(self, stmt: IndexAssign, ctx: _RetType) -> _RetType:
        fvs = ctx | self._visit_expr(stmt.expr, None)
        for slice in stmt.slices:
            fvs.update(self._visit_expr(slice, None))
        fvs.add(stmt.var)
        return fvs

    def _visit_if1(self, stmt: If1Stmt, ctx: _RetType) -> _RetType:
        ift_fvs = self._visit_block(stmt.body, ctx)
        cond_fvs = self._visit_expr(stmt.cond, None)
        return ctx | ift_fvs | cond_fvs

    def _visit_if(self, stmt: IfStmt, ctx: _RetType) -> _RetType:
        ift_fvs = self._visit_block(stmt.ift, ctx)
        iff_fvs = self._visit_block(stmt.iff, ctx)
        cond_fvs = self._visit_expr(stmt.cond, None)
        return ift_fvs | iff_fvs | cond_fvs

    def _visit_while(self, stmt: WhileStmt, ctx: _RetType) -> _RetType:
        ctx = ctx | self._visit_block(stmt.body, ctx)
        return ctx | self._visit_expr(stmt.cond, None)

    def _visit_for(self, stmt: ForStmt, ctx: _RetType) -> _RetType:
        body_fvs = self._visit_block(stmt.body, ctx)
        if isinstance(stmt.var, NamedId):
            body_fvs -= { stmt.var }
        ctx = ctx | body_fvs
        return ctx | self._visit_expr(stmt.iterable, None)

    def _visit_foreign_attr(self, e: ForeignAttribute, ctx: None) -> _RetType:
        return { e.name }

    def _visit_context_expr(self, e: ContextExpr, ctx: None) -> _RetType:
        match e.ctor:
            case Var():
                fvs = self._visit_var(e.ctor, None)
            case ForeignAttribute():
                fvs = self._visit_foreign_attr(e.ctor, None)
            case _:
                raise RuntimeError(f'unreachable {e.ctor}')

        for arg in e.args:
            match arg:
                case ForeignAttribute():
                    fvs |= self._visit_foreign_attr(arg, None)
                case _:
                    fvs |= self._visit_expr(arg, None)

        return fvs

    def _visit_context(self, stmt: ContextStmt, ctx: _RetType) -> _RetType:
        body_fvs = self._visit_block(stmt.body, ctx)
        if isinstance(stmt.name, NamedId):
            body_fvs -= { stmt.name }
        ctx = ctx | body_fvs
        match stmt.ctx:
            case Var():
                ctx |= self._visit_var(stmt.ctx, None)
            case ContextExpr():
                ctx |= self._visit_context_expr(stmt.ctx, None)
            case _:
                raise RuntimeError(f'unreachable {stmt.ctx}')
        return ctx

    def _visit_assert(self, stmt: AssertStmt, ctx: _RetType) -> _RetType:
        return ctx | self._visit_expr(stmt.test, None)

    def _visit_effect(self, stmt: EffectStmt, ctx: _RetType) -> _RetType:
        return ctx | self._visit_expr(stmt.expr, None)

    def _visit_return(self, stmt: ReturnStmt, ctx: _RetType) -> _RetType:
        return self._visit_expr(stmt.expr, None)

    def _visit_block(self, block: StmtBlock, ctx: _RetType) -> _RetType:
        fvs = ctx.copy()
        for stmt in reversed(block.stmts):
            if isinstance(stmt, ReturnStmt):
                fvs = set()
            fvs = self._visit_statement(stmt, fvs)
        return fvs

    def _visit_function(self, func: FuncDef, ctx: None) -> _RetType:
        fvs = self._visit_block(func.body, set())
        for arg in func.args:
            if isinstance(arg, NamedId):
                fvs.remove(arg)
        return fvs

    # overriden for typing hint
    def _visit_expr(self, e: Expr, ctx: None) -> _RetType:
        return super()._visit_expr(e, ctx)

    # overriden for typing hint
    def _visit_statement(self, e: Stmt, ctx: _RetType) -> _RetType:
        return super()._visit_statement(e, ctx)


class LiveVars:
    """Live variable analysis for the FPy AST."""

    @staticmethod
    def analyze(e: FuncDef | StmtBlock | Stmt | Expr):
        """Analyze the live variables in a function."""
        if not isinstance(e, (FuncDef, StmtBlock, Stmt, Expr)):
            raise TypeError(f'Expected FunctionDef, Block, Stmt or Expr, got {type(e)}')
        return _LiveVars(e).analyze()
