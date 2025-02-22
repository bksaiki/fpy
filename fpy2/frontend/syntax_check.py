"""Syntax checking for the FPy AST."""

from typing import Optional, Self

from .fpyast import *
from .visitor import AstVisitor

from ..utils import FPySyntaxError

class _Env:
    """Bound variables in the current scope."""
    env: dict[NamedId, bool]

    def __init__(self, env: Optional[dict[NamedId, bool]] = None):
        if env is None:
            self.env = {}
        else:
            self.env = env.copy()

    def __contains__(self, key):
        return key in self.env

    def __getitem__(self, key):
        return self.env[key]

    def extend(self, var: NamedId):
        copy = _Env(self.env)
        copy.env[var] = True
        return copy

    def merge(self, other: Self):
        copy = _Env()
        for key in self.env.keys() | other.env.keys():
            copy.env[key] = self.env.get(key, False) and other.env.get(key, False)
        return copy

_Ctx = tuple[_Env, bool]
"""
1st element: environment
2nd element: whether the current block is at the top-level.
"""

class SyntaxCheckInstance(AstVisitor):
    """Single-use instance of syntax checking"""
    func: FunctionDef
    rets: set[Stmt]

    def __init__(self, func: FunctionDef):
        self.func = func
        self.rets = set()

    def analyze(self):
        self._visit_function(self.func, (_Env(), False))
        if len(self.rets) == 0:
            raise FPySyntaxError('function has no return statement')
        elif len(self.rets) > 1:
            raise FPySyntaxError('function has multiple return statements')

    def _visit_var(self, e: Var, ctx: _Ctx):
        env, _ = ctx
        if not isinstance(e.name, NamedId):
            raise FPySyntaxError(f'expected a NamedId, got {e.name}')
        if e.name not in env:
            print(repr(e.name), repr(env.env))
            raise FPySyntaxError(f'unbound variable `{e.name}`')
        if not env[e.name]:
            raise FPySyntaxError(f'variable `{e.name}` not defined along all paths')
        return env

    def _visit_bool(self, e: Bool, ctx: _Ctx):
        env, _ = ctx
        return env

    def _visit_decnum(self, e: Decnum, ctx: _Ctx):
        env, _ = ctx
        return env

    def _visit_hexnum(self, e: Hexnum, ctx: _Ctx):
        env, _ = ctx
        return env

    def _visit_integer(self, e: Integer, ctx: _Ctx):
        env, _ = ctx
        return env

    def _visit_rational(self, e: Rational, ctx: _Ctx):
        env, _ = ctx
        return env

    def _visit_digits(self, e: Digits, ctx: _Ctx):
        env, _ = ctx
        return env

    def _visit_constant(self, e: Constant, ctx: _Ctx):
        env, _ = ctx
        return env

    def _visit_unaryop(self, e: UnaryOp, ctx: _Ctx):
        env, _ = ctx
        self._visit_expr(e.arg, ctx)
        return env

    def _visit_binaryop(self, e: BinaryOp, ctx: _Ctx):
        env, _ = ctx
        self._visit_expr(e.left, ctx)
        self._visit_expr(e.right, ctx)
        return env

    def _visit_ternaryop(self, e: TernaryOp, ctx: _Ctx):
        env, _ = ctx
        self._visit_expr(e.arg0, ctx)
        self._visit_expr(e.arg1, ctx)
        self._visit_expr(e.arg2, ctx)
        return env

    def _visit_naryop(self, e: NaryOp, ctx: _Ctx):
        env, _ = ctx
        for c in e.args:
            self._visit_expr(c, ctx)
        return env

    def _visit_compare(self, e: Compare, ctx: _Ctx):
        env, _ = ctx
        for c in e.args:
            self._visit_expr(c, ctx)
        return env

    def _visit_call(self, e: Call, ctx: _Ctx):
        env, _ = ctx
        for c in e.args:
            self._visit_expr(c, ctx)
        return env

    def _visit_tuple_expr(self, e: TupleExpr, ctx: _Ctx):
        env, _ = ctx
        for c in e.args:
            self._visit_expr(c, ctx)
        return env

    def _visit_comp_expr(self, e: CompExpr, ctx: _Ctx):
        env, _ = ctx
        for iterable in e.iterables:
            self._visit_expr(iterable, ctx)
        for var in e.vars:
            if isinstance(var, NamedId):
                env = env.extend(var)
        self._visit_expr(e.elt, (env, False))
        return env

    def _visit_ref_expr(self, e: RefExpr, ctx: _Ctx):
        env, _ = ctx
        self._visit_expr(e.value, ctx)
        for s in e.slices:
            self._visit_expr(s, ctx)
        return env

    def _visit_if_expr(self, e: IfExpr, ctx: _Ctx):
        env, _ = ctx
        self._visit_expr(e.cond, ctx)
        self._visit_expr(e.ift, ctx)
        self._visit_expr(e.iff, ctx)
        return env

    def _visit_var_assign(self, stmt: VarAssign, ctx: _Ctx):
        env, _ = ctx
        self._visit_expr(stmt.expr, ctx)
        if isinstance(stmt.var, NamedId):
            env = env.extend(stmt.var)
        return env

    def _visit_tuple_binding(self, binding: TupleBinding, ctx: _Ctx):
        env, _ = ctx
        for elt in binding.elts:
            match elt:
                case NamedId():
                    env = env.extend(elt)
                case TupleBinding():
                    env = self._visit_tuple_binding(elt, ctx)
                case UnderscoreId():
                    pass
                case _:
                    raise NotImplementedError('unreachable', elt)
        return env

    def _visit_tuple_assign(self, stmt: TupleAssign, ctx: _Ctx):
        self._visit_expr(stmt.expr, ctx)
        return self._visit_tuple_binding(stmt.binding, ctx)

    def _visit_ref_assign(self, stmt: RefAssign, ctx: _Ctx):
        env, _ = ctx
        for s in stmt.slices:
            self._visit_expr(s, ctx)
        self._visit_expr(stmt.expr, ctx)
        return env

    def _visit_if_stmt(self, stmt: IfStmt, ctx: _Ctx):
        self._visit_expr(stmt.cond, ctx)
        ift_env = self._visit_block(stmt.ift, ctx)
        if stmt.iff is None:
            # 1-armed if statement
            env, _ = ctx
            return ift_env.merge(env)
        else:
            iff_env = self._visit_block(stmt.iff, ctx)
            return ift_env.merge(iff_env)

    def _visit_while_stmt(self, stmt: WhileStmt, ctx: _Ctx):
        env, _ = ctx
        body_env = self._visit_block(stmt.body, ctx)
        env = env.merge(body_env)
        self._visit_expr(stmt.cond, (env, False))
        return env

    def _visit_for_stmt(self, stmt: ForStmt, ctx: _Ctx):
        env, _ = ctx
        self._visit_expr(stmt.iterable, ctx)
        if isinstance(stmt.var, NamedId):
            env = env.extend(stmt.var)
        body_env = self._visit_block(stmt.body, (env, False))
        return env.merge(body_env)

    def _visit_context(self, stmt: ContextStmt, ctx: _Ctx):
        env, is_top = ctx
        if stmt.name is not None and isinstance(stmt.name, NamedId):
            env = env.extend(stmt.name)
        return self._visit_block(stmt.body, (env, is_top))

    def _visit_assert(self, stmt: AssertStmt, ctx: _Ctx):
        env, _ = ctx
        self._visit_expr(stmt.test, ctx)
        return env

    def _visit_return(self, stmt: Return, ctx: _Ctx):
        return self._visit_expr(stmt.expr, ctx)

    def _visit_block(self, block: Block, ctx: _Ctx):
        env, is_top = ctx
        for i, stmt in enumerate(block.stmts):
            match stmt:
                case Return():
                    if not is_top:
                        raise FPySyntaxError('return statement must be at the top-level')
                    if i != len(block.stmts) - 1:
                        raise FPySyntaxError('return statement must be at the end of the function definition')
                    env = self._visit_statement(stmt, (env, False))
                    self.rets.add(stmt)
                case ContextStmt():
                    env = self._visit_statement(stmt, (env, is_top))
                case _:
                    env = self._visit_statement(stmt, (env, False))

        return env

    def _visit_function(self, func: FunctionDef, ctx: _Ctx):
        env, _ = ctx
        for arg in func.args:
            if isinstance(arg.name, NamedId):
                env = env.extend(arg.name)
        return self._visit_block(func.body, (env, True))

    # override to get typing hint
    def _visit_statement(self, stmt: Stmt, ctx: _Ctx) -> _Env:
        return super()._visit_statement(stmt, ctx)

    # override to get typing hint
    def _visit_expr(self, e: Expr, ctx: _Ctx) -> _Env:
        return super()._visit_expr(e, ctx)


class SyntaxCheck:
    """
    Syntax checker for the FPy AST.

    Basic syntax check to eliminate malformed FPy programs that
    the parser can't detect.

    Rules enforced:

    - any variables must be defined before it is used;

    - `return` statement:
      - all functions must have exactly one return statement,
      - return statement must be at the end of the function definiton;

    - `if` statement:
      - any variable must be defined along both branches when
        used after the `if` statement
    """

    @staticmethod
    def analyze(func: FunctionDef):
        if not isinstance(func, FunctionDef):
            raise TypeError(f'expected a Function, got {func}')
        SyntaxCheckInstance(func).analyze()
