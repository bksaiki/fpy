"""Live variable analysis for the FPy AST."""

from ..ast import *

_LiveSet = set[NamedId]

class LiveVarsInstance(Visitor):
    """Single-use live variable analyzer"""

    ast: FuncDef | Expr

    def __init__(self, ast: FuncDef | Expr):
        super().__init__()
        self.ast = ast

    def analyze(self):
        match self.ast:
            case FuncDef():
                return self._visit_function(self.ast, set())
            case Expr():
                return self._visit_expr(self.ast, None)
            case _:
                raise RuntimeError(f'unreachable case: {self.ast}')

    def _visit_var(self, e: Var, ctx: None) -> _LiveSet:
        return { e.name }

    def _visit_bool(self, e: BoolVal, ctx: None) -> _LiveSet:
        return set()

    def _visit_foreign(self, e: ForeignVal, ctx: None):
        return set()

    def _visit_decnum(self, e: Decnum, ctx: None) -> _LiveSet:
        return set()

    def _visit_hexnum(self, e: Hexnum, ctx: None) -> _LiveSet:
        return set()

    def _visit_integer(self, e: Integer, ctx: None) -> _LiveSet:
        return set()

    def _visit_rational(self, e: Rational, ctx: None) -> _LiveSet:
        return set()

    def _visit_digits(self, e: Digits, ctx: None) -> _LiveSet:
        return set()

    def _visit_constant(self, e: Constant, ctx: None) -> _LiveSet:
        return set()

    def _visit_unaryop(self, e: UnaryOp, ctx: None) -> _LiveSet:
        return self._visit_expr(e.arg, ctx)

    def _visit_binaryop(self, e: BinaryOp, ctx: None) -> _LiveSet:
        return self._visit_expr(e.first, ctx) | self._visit_expr(e.second, ctx)

    def _visit_ternaryop(self, e: TernaryOp, ctx: None) -> _LiveSet:
        live0 = self._visit_expr(e.first, ctx)
        live1 = self._visit_expr(e.second, ctx)
        live2 = self._visit_expr(e.third, ctx)
        return live0 | live1 | live2

    def _visit_naryop(self, e: NaryOp, ctx: None) -> _LiveSet:
        live: set[NamedId] = set()
        for arg in e.args:
            live |= self._visit_expr(arg, ctx)
        return live

    def _visit_compare(self, e: Compare, ctx: None) -> _LiveSet:
        live: set[NamedId] = set()
        for arg in e.args:
            live |= self._visit_expr(arg, ctx)
        return live

    def _visit_call(self, e: Call, ctx: None) -> _LiveSet:
        live: set[NamedId] = set()
        for arg in e.args:
            live |= self._visit_expr(arg, ctx)
        return live

    def _visit_tuple_expr(self, e: TupleExpr, ctx: None) -> _LiveSet:
        live: set[NamedId] = set()
        for arg in e.args:
            live |= self._visit_expr(arg, ctx)
        return live

    def _visit_comp_expr(self, e: CompExpr, ctx: None) -> _LiveSet:
        live = self._visit_expr(e.elt, ctx)
        for target in e.targets:
            live |= target.names()
        for iterable in e.iterables:
            live |= self._visit_expr(iterable, ctx)
        return live

    def _visit_tuple_ref(self, e: TupleRef, ctx: None) -> _LiveSet:
        live = self._visit_expr(e.value, ctx)
        for s in e.slices:
            live |= self._visit_expr(s, ctx)
        return live

    def _visit_tuple_set(self, e: TupleSet, ctx: None) -> _LiveSet:
        live = self._visit_expr(e.array, ctx)
        for s in e.slices:
            live |= self._visit_expr(s, ctx)
        live |= self._visit_expr(e.value, ctx)
        return live

    def _visit_if_expr(self, e: IfExpr, ctx: None) -> _LiveSet:
        cond_live = self._visit_expr(e.cond, ctx)
        ift_live = self._visit_expr(e.ift, ctx)
        iff_live = self._visit_expr(e.iff, ctx)
        return cond_live | ift_live | iff_live

    def _visit_context_expr(self, e: ContextExpr, ctx: None) -> _LiveSet:
        live: set[NamedId] = set()
        for arg in e.args:
            if isinstance(arg, NamedId):
                live.add(arg)
        return live

    def _visit_assign(self, stmt: Assign, live: _LiveSet) -> _LiveSet:
        live = set(live)
        live -= stmt.binding.names()
        live |= self._visit_expr(stmt.expr, None)
        return live

    def _visit_indexed_assign(self, stmt: IndexedAssign, live: _LiveSet) -> _LiveSet:
        live = set(live)
        live |= self._visit_expr(stmt.expr, None)
        for s in stmt.slices:
            live |= self._visit_expr(s, None)
        live.add(stmt.var)
        return live

    def _visit_if1(self, stmt: If1Stmt, live: _LiveSet) -> _LiveSet:
        live = set(live)
        live |= self._visit_block(stmt.body, live)
        live |= self._visit_expr(stmt.cond, None)
        return live

    def _visit_if(self, stmt: IfStmt, live: _LiveSet) -> _LiveSet:
        live = set(live)
        ift_live = self._visit_block(stmt.ift, live)
        iff_live = self._visit_block(stmt.iff, live)
        cond_live = self._visit_expr(stmt.cond, None)
        return (ift_live | iff_live) | cond_live

    def _visit_while(self, stmt: WhileStmt, live: _LiveSet) -> _LiveSet:
        live = set(live)
        live |= self._visit_block(stmt.body, live)
        live |= self._visit_expr(stmt.cond, None)
        return live

    def _visit_for(self, stmt: ForStmt, live: _LiveSet) -> _LiveSet:
        live = set(live)
        live |= self._visit_block(stmt.body, live)
        match stmt.target:
            case NamedId():
                live -= { stmt.target }
            case TupleBinding():
                live -= stmt.target.names()
        live |= self._visit_expr(stmt.iterable, None)
        return live

    def _visit_context(self, stmt: ContextStmt, live: _LiveSet) -> _LiveSet:
        live = set(live)
        live = self._visit_block(stmt.body, live)
        if isinstance(stmt.name, NamedId):
            live -= { stmt.name }
        live |= self._visit_expr(stmt.ctx, None)
        return live

    def _visit_assert(self, stmt: AssertStmt, live: _LiveSet) -> _LiveSet:
        return live | self._visit_expr(stmt.test, None)

    def _visit_effect(self, stmt: EffectStmt, live: _LiveSet) -> _LiveSet:
        return live | self._visit_expr(stmt.expr, None)

    def _visit_return(self, stmt: ReturnStmt, live: _LiveSet) -> _LiveSet:
        return self._visit_expr(stmt.expr, None)

    def _visit_block(self, block: StmtBlock, live: _LiveSet) -> _LiveSet:
        live = set(live)
        for stmt in reversed(block.stmts):
            live = self._visit_statement(stmt, live)
        return live

    def _visit_function(self, func: FuncDef, ctx: _LiveSet):
        live =  self._visit_block(func.body, ctx)
        for arg in func.args:
            if isinstance(arg.name, NamedId):
                live -= { arg.name }
        return live

    # override for typing hint
    def _visit_expr(self, e: Expr, ctx: None) -> _LiveSet:
        return super()._visit_expr(e, ctx)

    # override for typing hint
    def _visit_statement(self, stmt: Stmt, ctx: _LiveSet) -> _LiveSet:
        return super()._visit_statement(stmt, ctx)


class LiveVars:
    """Live variable analysis for the FPy AST."""

    @staticmethod
    def analyze(ast: FuncDef | Expr):
        """Analyze the live variables in a function."""
        if not isinstance(ast, FuncDef | Expr):
            raise TypeError(f'expected a \'Function\' or \'Expr\', got {ast}')
        return LiveVarsInstance(ast).analyze()
