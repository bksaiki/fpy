"""Syntax checking for the FPy AST."""

from typing import Optional, Self

from ..utils import FPySyntaxError

from ..ast.fpyast import *
from ..ast.visitor import Visitor
from .live_vars import LiveVars

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

class SyntaxCheckInstance(Visitor):
    """Single-use instance of syntax checking"""
    func: FuncDef
    free_vars: set[NamedId]
    ignore_unknown: bool
    ignore_noreturn: bool
    allow_wildcard: bool

    free_var_args: set[NamedId]
    rets: set[Stmt]

    def __init__(
        self,
        func: FuncDef,
        free_vars: set[NamedId],
        ignore_unknown: bool,
        ignore_noreturn: bool,
        allow_wildcard: bool
    ):
        self.func = func
        self.free_vars = free_vars
        self.ignore_unknown = ignore_unknown
        self.ignore_noreturn = ignore_noreturn
        self.allow_wildcard = allow_wildcard

        self.free_var_args = set()
        self.rets = set()

    def analyze(self):
        self._visit_function(self.func, (_Env(), False))
        if not self.ignore_noreturn:
            if len(self.rets) == 0:
                raise FPySyntaxError('function has no return statement')
            elif len(self.rets) > 1:
                raise FPySyntaxError('function has multiple return statements')
        return self.free_var_args

    def _mark_use(
        self,
        name: NamedId,
        env: _Env,
        *,
        ignore_missing: bool = False
    ):
        if not ignore_missing:
            if name not in env:
                raise FPySyntaxError(f'unbound variable `{name}`')
            if not env[name]:
                raise FPySyntaxError(f'variable `{name}` not defined along all paths')
        if name in self.free_vars:
            self.free_var_args.add(name)

    def _visit_var(self, e: Var, ctx: _Ctx):
        env, _ = ctx
        match e.name:
            case NamedId():
                self._mark_use(e.name, env)
            case UnderscoreId():
                if not self.allow_wildcard:
                    raise FPySyntaxError('wildcard `_` not allowed in this context')
            case _:
                raise FPySyntaxError(f'expected a NamedId, got {e.name}')
        return env

    def _visit_bool(self, e: BoolVal, ctx: _Ctx):
        env, _ = ctx
        return env

    def _visit_foreign(self, e: ForeignVal, ctx: _Ctx):
        env, _ = ctx
        return env

    def _visit_context_val(self, e, ctx):
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
        self._visit_expr(e.first, ctx)
        self._visit_expr(e.second, ctx)
        return env

    def _visit_ternaryop(self, e: TernaryOp, ctx: _Ctx):
        env, _ = ctx
        self._visit_expr(e.first, ctx)
        self._visit_expr(e.second, ctx)
        self._visit_expr(e.third, ctx)
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
        self._mark_use(NamedId(e.name), env, ignore_missing=self.ignore_unknown)
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
        for target in e.targets:
            env = self._visit_binding(target, env)
        self._visit_expr(e.elt, (env, False))
        return env

    def _visit_tuple_ref(self, e: TupleRef, ctx: _Ctx):
        env, _ = ctx
        self._visit_expr(e.value, ctx)
        for s in e.slices:
            self._visit_expr(s, ctx)
        return env

    def _visit_tuple_set(self, e: TupleSet, ctx: _Ctx):
        env, _ = ctx
        self._visit_expr(e.array, ctx)
        for s in e.slices:
            self._visit_expr(s, ctx)
        self._visit_expr(e.value, ctx)
        return env

    def _visit_if_expr(self, e: IfExpr, ctx: _Ctx):
        env, _ = ctx
        self._visit_expr(e.cond, ctx)
        self._visit_expr(e.ift, ctx)
        self._visit_expr(e.iff, ctx)
        return env

    def _visit_foreign_attr(self, e: ForeignAttribute, ctx: _Ctx):
        env, _ = ctx
        self._mark_use(e.name, env)

    def _visit_context_expr(self, e: ContextExpr, ctx: _Ctx):
        # check the constructor
        match e.ctor:
            case ForeignAttribute():
                self._visit_foreign_attr(e.ctor, ctx)
            case Var():
                self._visit_var(e.ctor, ctx)
            case _:
                raise RuntimeError('unreachable', e.ctor)
        # check that context is not data-dependent
        for arg in e.args:
            match arg:
                case ForeignAttribute():
                    self._visit_foreign_attr(arg, ctx)
                case _:
                    for free in LiveVars.analyze(arg):
                        if free not in self.free_vars:
                            raise FPySyntaxError('context is data-dependent')

    def _visit_binding(self, binding: Id | TupleBinding, env: _Env):
        match binding:
            case NamedId():
                env = env.extend(binding)
            case UnderscoreId():
                pass
            case TupleBinding():
                env = self._visit_tuple_binding(binding, env)
            case _:
                raise RuntimeError('unreachable', binding)
        return env

    def _visit_tuple_binding(self, binding: TupleBinding, env: _Env):
        for elt in binding.elts:
            env = self._visit_binding(elt, env)
        return env

    def _visit_assign(self, stmt: Assign, ctx: _Ctx):
        env, _ = ctx
        self._visit_expr(stmt.expr, ctx)
        return self._visit_binding(stmt.binding, env)

    def _visit_indexed_assign(self, stmt: IndexedAssign, ctx: _Ctx):
        env, _ = ctx
        self._mark_use(stmt.var, env)
        for s in stmt.slices:
            self._visit_expr(s, ctx)
        self._visit_expr(stmt.expr, ctx)
        return env

    def _visit_if1(self, stmt: If1Stmt, ctx: _Ctx):
        env, _ = ctx
        self._visit_expr(stmt.cond, ctx)
        ift_env = self._visit_block(stmt.body, ctx)
        return env.merge(ift_env)

    def _visit_if(self, stmt: IfStmt, ctx: _Ctx):
        self._visit_expr(stmt.cond, ctx)
        ift_env = self._visit_block(stmt.ift, ctx)
        iff_env = self._visit_block(stmt.iff, ctx)
        return ift_env.merge(iff_env)

    def _visit_while(self, stmt: WhileStmt, ctx: _Ctx):
        env, _ = ctx
        body_env = self._visit_block(stmt.body, ctx)
        env = env.merge(body_env)
        self._visit_expr(stmt.cond, (env, False))
        return env

    def _visit_for(self, stmt: ForStmt, ctx: _Ctx):
        env, _ = ctx
        self._visit_expr(stmt.iterable, ctx)
        env = self._visit_binding(stmt.target, env)
        body_env = self._visit_block(stmt.body, (env, False))
        return env.merge(body_env)

    def _visit_context(self, stmt: ContextStmt, ctx: _Ctx):
        env, is_top = ctx
        self._visit_expr(stmt.ctx, ctx)
        if isinstance(stmt.name, NamedId):
            env = env.extend(stmt.name)
        return self._visit_block(stmt.body, (env, is_top))

    def _visit_assert(self, stmt: AssertStmt, ctx: _Ctx):
        env, _ = ctx
        self._visit_expr(stmt.test, ctx)
        return env

    def _visit_effect(self, stmt: EffectStmt, ctx: _Ctx):
        env, _ = ctx
        self._visit_expr(stmt.expr, ctx)
        return env

    def _visit_return(self, stmt: ReturnStmt, ctx: _Ctx):
        return self._visit_expr(stmt.expr, ctx)

    def _visit_block(self, block: StmtBlock, ctx: _Ctx):
        env, is_top = ctx
        for i, stmt in enumerate(block.stmts):
            match stmt:
                case ReturnStmt():
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

    def _visit_function(self, func: FuncDef, ctx: _Ctx):
        env, _ = ctx
        for var in self.free_vars:
            env = env.extend(var)
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

    Basic syntax check to eliminate malformed FPy programs
    that the parser can't detect.

    Rules enforced:

    Variables:

    - any variables must be defined before it is used;

    Return statements:

    - all functions must have exactly one return statement,
    - must be at the end of the function definiton;

    If statements

    - any variable must be defined along both branches when
      used after the `if` statement
    """

    @staticmethod
    def check(
        func: FuncDef,
        *,
        free_vars: Optional[set[NamedId]] = None,
        ignore_unknown: bool = False,
        ignore_noreturn: bool = False,
        allow_wildcard: bool = False
    ):
        """
        Analyzes the function for syntax errors.

        Returns the subset of `free_vars` that are relevant to FPy.
        """

        if not isinstance(func, FuncDef):
            raise TypeError(f'expected a Function, got {func}')

        if free_vars is None:
            free_vars = set(func.free_vars)

        inst = SyntaxCheckInstance(func, free_vars, ignore_unknown, ignore_noreturn, allow_wildcard)
        return inst.analyze()
