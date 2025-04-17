"""Definition-use analysis on the FPy IR"""

from ..ir import *

class _DefineUseInstance(DefaultVisitor):
    """Per-IR instance of definition-use analysis"""
    func: FunctionDef
    uses: dict[NamedId, set[Var | PhiNode]]
    done: bool

    def __init__(self, func: FunctionDef):
        self.func = func
        self.uses = {}
        self.done = False

    def analyze(self):
        if self.done:
            raise RuntimeError('analysis already performed')
        self._visit_function(self.func, None)
        self.done = True
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

    def _visit_var_assign(self, stmt: VarAssign, ctx: None):
        self._visit_expr(stmt.expr, ctx)
        if isinstance(stmt.var, NamedId):
            self.uses[stmt.var] = set()

    def _visit_tuple_assign(self, stmt: TupleAssign, ctx: None):
        self._visit_expr(stmt.expr, ctx)
        for var in stmt.binding.names():
            self.uses[var] = set()

    def _visit_if1_stmt(self, stmt: If1Stmt, ctx: None):
        self._visit_expr(stmt.cond, ctx)
        self._visit_block(stmt.body, ctx)
        for phi in stmt.phis:
            self.uses[phi.name] = set()
            self.uses[phi.lhs].add(phi)
            self.uses[phi.rhs].add(phi)

    def _visit_if_stmt(self, stmt: IfStmt, ctx: None):
        self._visit_expr(stmt.cond, ctx)
        self._visit_block(stmt.ift, ctx)
        self._visit_block(stmt.iff, ctx)
        for phi in stmt.phis:
            self.uses[phi.name] = set()
            self.uses[phi.lhs].add(phi)
            self.uses[phi.rhs].add(phi)

    def _visit_while_stmt(self, stmt: WhileStmt, ctx: None):
        for phi in stmt.phis:
            self.uses[phi.name] = set()
            self.uses[phi.lhs].add(phi)
        self._visit_expr(stmt.cond, ctx)
        self._visit_block(stmt.body, ctx)
        for phi in stmt.phis:
            self.uses[phi.rhs].add(phi)

    def _visit_for_stmt(self, stmt: ForStmt, ctx: None):
        self._visit_expr(stmt.iterable, ctx)
        if isinstance(stmt.var, NamedId):
            self.uses[stmt.var] = set()
        for phi in stmt.phis:
            self.uses[phi.name] = set()
            self.uses[phi.lhs].add(phi)
        self._visit_block(stmt.body, ctx)
        for phi in stmt.phis:
            self.uses[phi.rhs].add(phi)

    def _visit_function(self, func: FunctionDef, ctx):
        for arg in func.args:
            if isinstance(arg.name, NamedId):
                self.uses[arg.name] = set()
        self._visit_block(func.body, ctx)

    # override to get typing hint
    def _visit_statement(self, stmt: Stmt, ctx: None):
        return super()._visit_statement(stmt, ctx)

class DefineUse:
    """
    Definition-use analyzer for the FPy IR.

    Computes the set of definitions and their uses. Associates to
    each statement, the incoming definitions and outgoing definitions.
    """

    analysis_name = 'define_use'

    @staticmethod
    def analyze(func: FunctionDef):
        return _DefineUseInstance(func).analyze()
