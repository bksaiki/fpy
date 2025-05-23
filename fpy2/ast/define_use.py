"""Definition use analysis for FPy ASTs"""

from .fpyast import *
from .visitor import DefaultAstVisitor

class _DefineUseInstance(DefaultAstVisitor):
    """Per-IR instance of definition-use analysis"""
    ast: FuncDef | StmtBlock
    uses: dict[NamedId, set[Var]]

    def __init__(self, ast: FuncDef | StmtBlock):
        self.ast = ast
        self.uses = {}

    def analyze(self):
        match self.ast:
            case FuncDef():
                self._visit_function(self.ast, None)
            case StmtBlock():
                self._visit_block(self.ast, None)
            case _:
                raise RuntimeError(f'unreachable case: {self.ast}')
        return self.uses

    def _visit_var(self, e: Var, ctx: None):
        if e.name not in self.uses:
            raise NotImplementedError(f'undefined variable {e.name}')
        self.uses[e.name].add(e)

    def _visit_comp_expr(self, e: CompExpr, ctx: None):
        for iterable in e.iterables:
            self._visit_expr(iterable, ctx)
        for var in e.vars:
            if isinstance(var, NamedId):
                self.uses[var] = set()
        self._visit_expr(e.elt, ctx)

    def _visit_simple_assign(self, stmt: SimpleAssign, ctx: None):
        self._visit_expr(stmt.expr, ctx)
        if isinstance(stmt.var, NamedId):
            self.uses[stmt.var] = set()

    def _visit_tuple_unpack(self, stmt: TupleUnpack, ctx: None):
        self._visit_expr(stmt.expr, ctx)
        for var in stmt.binding.names():
            self.uses[var] = set()

    def _visit_if1(self, stmt: If1Stmt, ctx: None):
        self._visit_expr(stmt.cond, ctx)
        self._visit_block(stmt.body, ctx)

    def _visit_if(self, stmt: IfStmt, ctx: None):
        self._visit_expr(stmt.cond, ctx)
        self._visit_block(stmt.ift, ctx)
        self._visit_block(stmt.iff, ctx)

    def _visit_while(self, stmt: WhileStmt, ctx: None):
        self._visit_expr(stmt.cond, ctx)
        self._visit_block(stmt.body, ctx)

    def _visit_for(self, stmt: ForStmt, ctx: None):
        self._visit_expr(stmt.iterable, ctx)
        if isinstance(stmt.var, NamedId):
            self.uses[stmt.var] = set()
        self._visit_block(stmt.body, ctx)

    def _visit_function(self, func: FuncDef, ctx):
        for arg in func.args:
            if isinstance(arg.name, NamedId):
                self.uses[arg.name] = set()
        self._visit_block(func.body, ctx)


class DefineUse:
    """
    Definition-use analyzer for the FPy IR.

    Computes the set of definitions and their uses. Associates to
    each statement, the incoming definitions and outgoing definitions.
    """

    @staticmethod
    def analyze(ast: FuncDef | StmtBlock):
        if not isinstance(ast, FuncDef | StmtBlock):
            raise TypeError(f'Expected \'FuncDef\' or \'StmtBlock\', got {type(ast)} for {ast}')
        return _DefineUseInstance(ast).analyze()
