"""
Inverse of Static Single Assignment (SSA) transformation pass.
"""

from ..ir import *

class _SSAUnifyInstance(DefaultVisitor):
    """Compues the canonical variable for each SSA variable."""
    func: FuncDef
    unionfind: dict[NamedId, NamedId]

    def __init__(self, func: FuncDef):
        super().__init__()
        self.func = func
        self.unionfind = {}

    def apply(self) -> dict[NamedId, NamedId]:
        self._visit_function(self.func, None)
        return { k: self._find(k) for k in self.unionfind.keys() }

    def _find(self, x: NamedId) -> NamedId:
        """Finds the canonical variable for `x`."""
        # partially mutating lookup
        while x != self.unionfind[x]:
            gp = self.unionfind[self.unionfind[x]]
            self.unionfind[x] = gp
            x = gp

        return x

    def _union(self, x: NamedId, y: NamedId):
        """Unifies `x` and `y` with `x` as the leader."""
        # add `x` if not in environment
        if x not in self.unionfind:
            self.unionfind[x] = x

        # case split on if `y` is already added
        if y in self.unionfind:
            # get leader of `y` and set its leader to `x`
            y = self._find(y)
            self.unionfind[y] = x
        else:
            self.unionfind[y] = x


    def _visit_if1(self, stmt: If1Stmt, ctx: None):
        super()._visit_if1(stmt, ctx)
        # canonical variable is the incoming variable
        for phi in stmt.phis:
            self._union(phi.lhs, phi.rhs)
            self._union(phi.lhs, phi.name)

    def _visit_if(self, stmt: IfStmt, ctx: None):
        super()._visit_if(stmt, ctx)
        # canonical variable is in the ift branch
        # TODO: we should prioritize the incoming variable instead!
        for phi in stmt.phis:
            self._union(phi.lhs, phi.rhs)
            self._union(phi.lhs, phi.name)

    def _visit_while(self, stmt: WhileStmt, ctx: None):
        super()._visit_while(stmt, ctx)
        # canonical variable is the incoming variable
        for phi in stmt.phis:
            self._union(phi.lhs, phi.rhs)
            self._union(phi.lhs, phi.name)

    def _visit_for(self, stmt: ForStmt, ctx: None):
        super()._visit_for(stmt, ctx)
        # canonical variable is the incoming variable
        for phi in stmt.phis:
            self._union(phi.lhs, phi.rhs)
            self._union(phi.lhs, phi.name)


class _UnSSAInstance(DefaultTransformVisitor):
    """
    Single-use instance of an Un-SSA pass.

    Uses the canonical variable map to replace all phi nodes
    """
    func: FuncDef
    env: dict[NamedId, NamedId]

    def __init__(self, func: FuncDef, env: dict[NamedId, NamedId]):
        super().__init__()
        self.func = func
        self.env = env

    def apply(self) -> FuncDef:
        return self._visit_function(self.func, None)

    def _visit_var(self, e: Var, ctx: None):
        name = self.env.get(e.name, e.name)
        return Var(name)

    def _visit_comp_expr(self, e: CompExpr, ctx: None):
        new_vars: list[Id] = []
        for var in e.vars:
            match var:
                case NamedId():
                    renamed = self.env.get(var, var)
                    new_vars.append(renamed)
                case UnderscoreId():
                    new_vars.append(var)
                case _:
                    raise RuntimeError(f'unexpected {var}')

        iterables = [self._visit_expr(iter, ctx) for iter in e.iterables]
        elt = self._visit_expr(e.elt, ctx)
        return CompExpr(new_vars, iterables, elt)

    def _visit_simple_assign(self, stmt: SimpleAssign, ctx: None):
        if isinstance(stmt.var, NamedId):
            var: Id = self.env.get(stmt.var, stmt.var)
        else:
            var = stmt.var

        e = self._visit_expr(stmt.expr, ctx)
        s = SimpleAssign(var, stmt.ty, e)
        return s, None

    def _visit_tuple_binding(self, vars: TupleBinding):
        new_vars: list[Id | TupleBinding] = []
        for name in vars:
            match name:
                case NamedId():
                    renamed = self.env.get(name, name)
                    new_vars.append(renamed)
                case UnderscoreId():
                    new_vars.append(name)
                case TupleBinding():
                    elts = self._visit_tuple_binding(name)
                    new_vars.append(elts)
                case _:
                    raise RuntimeError(f'unexpected {name}')

        return TupleBinding(new_vars)

    def _visit_tuple_unpack(self, stmt: TupleUnpack, ctx: None):
        bindings = self._visit_tuple_binding(stmt.binding)
        expr = self._visit_expr(stmt.expr, ctx)
        s = TupleUnpack(bindings, stmt.ty, expr)
        return s, None

    def _visit_index_assign(self, stmt: IndexAssign, ctx: None):
        var = self.env.get(stmt.var, stmt.var)
        slices = [self._visit_expr(slice, ctx) for slice in stmt.slices]
        expr = self._visit_expr(stmt.expr, ctx)
        s = IndexAssign(var, slices, expr)
        return s, None

    def _visit_if1(self, stmt: If1Stmt, ctx: None):
        cond = self._visit_expr(stmt.cond, ctx)
        body, _ = self._visit_block(stmt.body, ctx)
        s = If1Stmt(cond, body, [])
        return s, None

    def _visit_if(self, stmt: IfStmt, ctx: None):
        cond = self._visit_expr(stmt.cond, ctx)
        ift, _ = self._visit_block(stmt.ift, ctx)
        iff, _ = self._visit_block(stmt.iff, ctx)
        s = IfStmt(cond, ift, iff, [])
        return s, None

    def _visit_while(self, stmt: WhileStmt, ctx: None):
        cond = self._visit_expr(stmt.cond, ctx)
        body, _ = self._visit_block(stmt.body, ctx)
        s = WhileStmt(cond, body, [])
        return s, None

    def _visit_for(self, stmt: ForStmt, ctx: None):
        if isinstance(stmt.var, NamedId):
            var: Id = self.env.get(stmt.var, stmt.var)
        else:
            var = stmt.var
        iterable = self._visit_expr(stmt.iterable, ctx)
        body, _ = self._visit_block(stmt.body, ctx)
        s = ForStmt(var, stmt.ty, iterable, body, [])
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
    def apply(func: FuncDef) -> FuncDef:
        canon = _SSAUnifyInstance(func).apply()
        return _UnSSAInstance(func, canon).apply()
