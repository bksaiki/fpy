"""Pass to ensure correctness of the FPy IR."""

from ..ir import *

_CtxType = set[NamedId]

class InvalidIRError(Exception):
    pass

# TODO: type check
class _VerifyPassInstance(DefaultVisitor):
    """Single instance of the `VerifyPass`."""
    func: FunctionDef
    types: dict[NamedId, IRType]

    def __init__(self, func: FunctionDef):
        self.func = func
        self.types = {}

    def check(self):
        self.types = {}
        self._visit_function(self.func, set())

    def _visit_var(self, e: Var, ctx: _CtxType):
        if e.name not in ctx:
            raise InvalidIRError(f'undefined variable {e.name}')

    def _visit_comp_expr(self, e: CompExpr, ctx: _CtxType):
        for iterable in e.iterables:
            self._visit_expr(iterable, ctx)
        for var in e.vars:
            match var:
                case NamedId():
                    if var in self.types:
                        raise InvalidIRError(f'reassignment of variable {var}')
                    self.types[var] = AnyType()
                    ctx.add(var)
                case UnderscoreId():
                    pass
                case _:
                    raise InvalidIRError('unreachable', var)
        self._visit_expr(e.elt, ctx)

    def _visit_var_assign(self, stmt: VarAssign, ctx: _CtxType):
        self._visit_expr(stmt.expr, ctx)
        match stmt.var:
            case NamedId():
                if stmt.var in self.types:
                    raise InvalidIRError(f'reassignment of variable {stmt.var}')
                self.types[stmt.var] = AnyType()
                ctx.add(stmt.var)
            case UnderscoreId():
                pass
            case _:
                raise InvalidIRError('unreachable', stmt.var)
        return ctx

    def _visit_tuple_assign(self, stmt: TupleAssign, ctx: _CtxType):
        self._visit_expr(stmt.expr, ctx)
        for var in stmt.binding.names():
            if var in self.types:
                raise InvalidIRError(f'reassignment of variable {var}')
            self.types[var] = AnyType()
            ctx.add(var)
        return ctx

    def _visit_ref_assign(self, stmt: RefAssign, ctx: _CtxType):
        if stmt.var not in ctx:
            raise InvalidIRError(f'undefined variable {stmt.var}')
        for s in stmt.slices:
            self._visit_expr(s, ctx)
        self._visit_expr(stmt.expr, ctx)
        return ctx

    def _visit_if1_stmt(self, stmt: If1Stmt, ctx: _CtxType):
        self._visit_expr(stmt.cond, ctx)
        body_ctx = self._visit_block(stmt.body, ctx.copy())
        # check validty of phi nodes and update context
        for phi in stmt.phis:
            name, orig, new = phi.name, phi.lhs, phi.rhs
            if name in self.types:
                raise InvalidIRError(f'reassignment of variable {name}')
            if orig not in ctx:
                raise InvalidIRError(f'undefined variable in LHS of phi {name} = ({orig}, {new})')
            if new not in body_ctx:
                raise InvalidIRError(f'undefined variable in RHS of phi {name} = ({orig}, {new})')
            if new == name:
                raise InvalidIRError(f'phi variable assigned to itself {name} = ({orig}, {new})')
            self.types[name] = AnyType()
            ctx.add(name)
            ctx -= { orig, new }
        return ctx

    def _visit_if_stmt(self, stmt: IfStmt, ctx: _CtxType):
        self._visit_expr(stmt.cond, ctx)
        ift_ctx = self._visit_block(stmt.ift, ctx.copy())
        iff_ctx = self._visit_block(stmt.iff, ctx.copy())
        # check validty of phi nodes and update context
        for phi in stmt.phis:
            name, ift_name, iff_name = phi.name, phi.lhs, phi.rhs
            if name in self.types:
                raise InvalidIRError(f'reassignment of variable {name}')
            if ift_name not in ift_ctx:
                raise InvalidIRError(f'undefined variable in LHS of phi {name} = ({ift_name}, {iff_name})')
            if iff_name not in iff_ctx:
                raise InvalidIRError(f'undefined variable in RHS of phi {name} = ({ift_name}, {iff_name})')
            if ift_name == iff_name:
                raise InvalidIRError(f'phi variable is unnecessary {name} = ({ift_name}, {iff_name})')
            self.types[name] = AnyType()
            ctx.add(name)
            ctx -= { ift_name, iff_name }
        return ctx

    def _visit_while_stmt(self, stmt: WhileStmt, ctx: _CtxType):
        # check (partial) validity of phi variables and update context
        for phi in stmt.phis:
            name, orig = phi.name, phi.lhs
            if name in self.types:
                raise InvalidIRError(f'reassignment of variable {name}')
            if orig not in ctx:
                raise InvalidIRError(f'undefined variable in LHS of phi {name} = ({orig}, _)')
            self.types[name] = AnyType()
            ctx.add(name)
            ctx -= { orig }
        # check condition and body
        self._visit_expr(stmt.cond, ctx)
        body_ctx = self._visit_block(stmt.body, ctx.copy())
        # check (partial) validity of phi variables
        for phi in stmt.phis:
            name, orig, new = phi.name, phi.lhs, phi.rhs
            if new not in body_ctx:
                raise InvalidIRError(f'undefined variable in RHS of phi {name} = (_, {new})')
            if new == name:
                raise InvalidIRError(f'phi variable assigned to itself {name} = ({orig}, {new})')
            ctx -= { new }
        return ctx

    def _visit_for_stmt(self, stmt: ForStmt, ctx: _CtxType):
        # check iterable expression
        self._visit_expr(stmt.iterable, ctx)
        # bind the loop variable
        match stmt.var:
            case NamedId():
                if stmt.var in self.types:
                    raise InvalidIRError(f'reassignment of variable {stmt.var}')
                self.types[stmt.var] = AnyType()
                ctx.add(stmt.var)
            case UnderscoreId():
                pass
            case _:
                raise InvalidIRError('unreachable', stmt.var)
        # check (partial) validity of phi variables and update context
        for phi in stmt.phis:
            name, orig = phi.name, phi.lhs
            if name in self.types:
                raise InvalidIRError(f'reassignment of variable {name}')
            if orig not in ctx:
                raise InvalidIRError(f'undefined variable in LHS of phi {name} = ({orig}, _)')
            self.types[name] = AnyType()
            ctx.add(name)
            ctx -= { orig }
        # check body
        body_ctx = self._visit_block(stmt.body, ctx.copy())
        # check (partial) validity of phi variables
        for phi in stmt.phis:
            name, orig, new = phi.name, phi.lhs, phi.rhs
            if new not in body_ctx:
                raise InvalidIRError(f'undefined variable in RHS of phi {name} = (_, {new})')
            if new == name:
                raise InvalidIRError(f'phi variable assigned to itself {name} = ({orig}, {new})')
            ctx -= { new }
        return ctx

    def _visit_context(self, stmt: ContextStmt, ctx: _CtxType):
        if stmt.name is not None and isinstance(stmt.name, NamedId):
            if stmt.name in self.types:
                raise InvalidIRError(f'reassignment of variable {stmt.name}')
            self.types[stmt.name] = AnyType()
            ctx.add(stmt.name)
        return self._visit_block(stmt.body, ctx)

    def _visit_assert(self, stmt: AssertStmt, ctx: _CtxType):
        self._visit_expr(stmt.test, ctx)
        return ctx

    def _visit_block(self, block: Block, ctx: _CtxType):
        for stmt in block.stmts:
            if not isinstance(stmt, Stmt):
                raise InvalidIRError(f'expected a statement {stmt}')
            elif isinstance(stmt, Return):
                self._visit_return(stmt, ctx)
                ctx = set()
            else:
                ctx = self._visit_statement(stmt, ctx.copy())
        return ctx

    def _visit_function(self, func: FunctionDef, ctx: _CtxType):
        for arg in func.args:
            if isinstance(arg.name, NamedId):
                self.types[arg.name] = AnyType()
                ctx.add(arg.name)
        self._visit_block(func.body, ctx)

    # override for typing hint
    def _visit_expr(self, e: Expr, ctx: _CtxType) -> None:
        super()._visit_expr(e, ctx)

    # override for typing hint
    def _visit_statement(self, stmt: Stmt, ctx: _CtxType) -> _CtxType:
        return super()._visit_statement(stmt, ctx)


class VerifyIR:
    """
    Checks that an FPy IR instance is syntactically valid,
    well-typed, and in static single assignment (SSA) form.
    """

    @staticmethod
    def check(func: FunctionDef):
        _VerifyPassInstance(func).check()
