"""
Inverse of Static Single Assignment (SSA) transformation pass.
"""

from ..ir import *

class _SSAUnifyInstance(DefaultVisitor):
    """Compues the canonical variable for each SSA variable."""
    func: FunctionDef
    env: dict[NamedId, NamedId]

    def __init__(self, func: FunctionDef):
        super().__init__()
        self.func = func
        self.env = {}

    def apply(self) -> dict[NamedId, NamedId]:
        self._visit_function(self.func, None)
        return { k: self._find(k) for k in self.env.keys() }

    def _find(self, x: NamedId) -> NamedId:
        """Finds the canonical variable for `x`."""
        # partially mutating lookup
        while x != self.env[x]:
            gp = self.env[self.env[x]]
            self.env[x] = gp
            x = gp

        return x

    def union(self, x: NamedId, y: NamedId):
        """Unifies `x` and `y` with `x` as the leader."""
        # add `x` if not in environment
        if x not in self.env:
            self.env[x] = x

        # case split on if `y` is already added
        if y in self.env:
            # get leader of `y` and set its leader to `x`
            y = self._find(y)
            self.env[y] = x
        else:
            self.env[y] = x


    def _visit_if1_stmt(self, stmt: If1Stmt, ctx: None):
        super()._visit_if1_stmt(stmt, ctx)
        # canonical variable is the incoming variable
        for phi in stmt.phis:
            self.union(phi.lhs, phi.rhs)
            self.union(phi.lhs, phi.name)

    def _visit_if_stmt(self, stmt: IfStmt, ctx: None):
        super()._visit_if_stmt(stmt, ctx)
        # canonical variable is in the ift branch
        # TODO: we should prioritize the incoming variable instead!
        for phi in stmt.phis:
            self.union(phi.lhs, phi.rhs)
            self.union(phi.lhs, phi.name)

    def _visit_while_stmt(self, stmt: WhileStmt, ctx: None):
        super()._visit_while_stmt(stmt, ctx)
        # canonical variable is the incoming variable
        for phi in stmt.phis:
            self.union(phi.lhs, phi.rhs)
            self.union(phi.lhs, phi.name)

    def _visit_for_stmt(self, stmt: ForStmt, ctx: None):
        super()._visit_for_stmt(stmt, ctx)
        # canonical variable is the incoming variable
        for phi in stmt.phis:
            self.union(phi.lhs, phi.rhs)
            self.union(phi.lhs, phi.name)


class _UnSSAInstance(DefaultTransformVisitor):
    """
    Single-use instance of an Un-SSA pass.

    Uses the canonical variable map to replace all phi nodes
    """
    func: FunctionDef
    env: dict[NamedId, NamedId]

    def __init__(self, func: FunctionDef, env: dict[NamedId, NamedId]):
        super().__init__()
        self.func = func
        self.env = env

    def apply(self) -> FunctionDef:
        return self._visit_function(self.func, None)

    def _visit_var(self, e: Var, ctx: None):
        name = self.env.get(e.name, e.name)
        return Var(name)

    def _visit_var_assign(self, stmt: VarAssign, ctx: None):
        e = self._visit_expr(stmt.expr, ctx)
        if isinstance(stmt.var, NamedId):
            var = self.env.get(stmt.var, stmt.var)
            s = VarAssign(var, stmt.ty, e)
        else:
            s = VarAssign(stmt.var, stmt.ty, e)
        return s, None

    def _visit_if1_stmt(self, stmt: If1Stmt, ctx: None):
        cond = self._visit_expr(stmt.cond, ctx)
        body, _ = self._visit_block(stmt.body, ctx)
        s = If1Stmt(cond, body, [])
        return s, None

    def _visit_if_stmt(self, stmt: IfStmt, ctx: None):
        cond = self._visit_expr(stmt.cond, ctx)
        ift, _ = self._visit_block(stmt.ift, ctx)
        iff, _ = self._visit_block(stmt.iff, ctx)
        s = IfStmt(cond, ift, iff, [])
        return s, None


class UnSSA:
    """
    Transformation pass reverts from Static Single Assignment (SSA) form
    into an IR with potentially mutable variables.

    Eliminates all phi nodes and replaces them with them with the
    "canonical" variable; there is no guarantee that the resulting
    variable is the original variable.
    """

    @staticmethod
    def apply(func: FunctionDef) -> FunctionDef:
        canon = _SSAUnifyInstance(func).apply()
        return _UnSSAInstance(func, canon).apply()
