"""Syntax checking for the FPy AST."""

from dataclasses import dataclass
from typing import Self

from ..ast.fpyast import *
from ..ast.visitor import Visitor
from .live_vars import LiveVars

class FPySyntaxError(Exception):
    """Syntax error for FPy programs."""
    pass


class _Env:
    """Bound variables in the current scope."""
    env: dict[NamedId, bool]

    def __init__(self, env: dict[NamedId, bool] | None = None):
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


@dataclass
class _Ctx:
    """Visitor context while syntax checking"""

    env: _Env
    """mapping from variable to whether the variable is bound on all paths"""

    within_call: bool
    """are we checking the function part of a call?"""

    @staticmethod
    def default():
        return _Ctx(_Env(), False)


class SyntaxCheckInstance(Visitor):
    """Single-use instance of syntax checking"""
    func: FuncDef
    free_vars: set[NamedId]
    ignore_unknown: bool
    allow_wildcard: bool
    free_var_args: set[NamedId]

    def __init__(
        self,
        func: FuncDef,
        free_vars: set[NamedId],
        ignore_unknown: bool,
        allow_wildcard: bool
    ):
        self.func = func
        self.free_vars = free_vars
        self.ignore_unknown = ignore_unknown
        self.allow_wildcard = allow_wildcard
        self.free_var_args = set()

    def analyze(self):
        self._visit_function(self.func, _Ctx.default())
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
        env = ctx.env
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
        env = ctx.env
        return env

    def _visit_foreign(self, e: ForeignVal, ctx: _Ctx):
        env = ctx.env
        return env

    def _visit_context_val(self, e, ctx):
        env = ctx.env
        return env

    def _visit_decnum(self, e: Decnum, ctx: _Ctx):
        env = ctx.env
        return env

    def _visit_hexnum(self, e: Hexnum, ctx: _Ctx):
        env = ctx.env
        return env

    def _visit_integer(self, e: Integer, ctx: _Ctx):
        env = ctx.env
        return env

    def _visit_rational(self, e: Rational, ctx: _Ctx):
        env = ctx.env
        return env

    def _visit_digits(self, e: Digits, ctx: _Ctx):
        env = ctx.env
        return env

    def _visit_nullaryop(self, e: NullaryOp, ctx: _Ctx):
        env = ctx.env
        return env

    def _visit_unaryop(self, e: UnaryOp, ctx: _Ctx):
        env = ctx.env
        self._visit_expr(e.arg, ctx)
        return env

    def _visit_binaryop(self, e: BinaryOp, ctx: _Ctx):
        env = ctx.env
        self._visit_expr(e.first, ctx)
        self._visit_expr(e.second, ctx)
        return env

    def _visit_ternaryop(self, e: TernaryOp, ctx: _Ctx):
        env = ctx.env
        self._visit_expr(e.first, ctx)
        self._visit_expr(e.second, ctx)
        self._visit_expr(e.third, ctx)
        return env

    def _visit_naryop(self, e: NaryOp, ctx: _Ctx):
        env = ctx.env
        for c in e.args:
            self._visit_expr(c, ctx)
        return env

    def _visit_compare(self, e: Compare, ctx: _Ctx):
        env = ctx.env
        for c in e.args:
            self._visit_expr(c, ctx)
        return env

    def _visit_call(self, e: Call, ctx: _Ctx):
        env = ctx.env
        match e.func:
            case Var():
                self._mark_use(e.func.name, env, ignore_missing=self.ignore_unknown)
            case Attribute():
                self._visit_attribute(e.func, _Ctx(env, True))
            case _:
                raise RuntimeError('unreachable', e.func)
        for arg in e.args:
            self._visit_expr(arg, ctx)
        for _, arg in e.kwargs:
            self._visit_expr(arg, ctx)
        return env

    def _visit_tuple_expr(self, e: TupleExpr, ctx: _Ctx):
        env = ctx.env
        for c in e.elts:
            self._visit_expr(c, ctx)
        return env

    def _visit_list_expr(self, e: ListExpr, ctx: _Ctx):
        env = ctx.env
        for c in e.elts:
            self._visit_expr(c, ctx)
        return env

    def _visit_list_comp(self, e: ListComp, ctx: _Ctx):
        env = ctx.env
        for iterable in e.iterables:
            self._visit_expr(iterable, ctx)
        for target in e.targets:
            env = self._visit_binding(target, env)
        self._visit_expr(e.elt, _Ctx(env, False))
        return env

    def _visit_list_ref(self, e: ListRef, ctx: _Ctx):
        env = ctx.env
        self._visit_expr(e.value, ctx)
        self._visit_expr(e.index, ctx)
        return env

    def _visit_list_slice(self, e: ListSlice, ctx: _Ctx):
        env = ctx.env
        self._visit_expr(e.value, ctx)
        if e.start is not None:
            self._visit_expr(e.start, ctx)
        if e.stop is not None:
            self._visit_expr(e.stop, ctx)
        return env

    def _visit_list_set(self, e: ListSet, ctx: _Ctx):
        env = ctx.env
        self._visit_expr(e.value, ctx)
        for s in e.indices:
            self._visit_expr(s, ctx)
        self._visit_expr(e.expr, ctx)
        return env

    def _visit_if_expr(self, e: IfExpr, ctx: _Ctx):
        env = ctx.env
        self._visit_expr(e.cond, ctx)
        self._visit_expr(e.ift, ctx)
        self._visit_expr(e.iff, ctx)
        return env

    def _visit_attribute(self, e: Attribute, ctx: _Ctx):
        if ctx.within_call and not isinstance(e.value, Var | Attribute):
            raise FPySyntaxError('attribute base in function position must be either a variable or another attribute')
        self._visit_expr(e.value, ctx)

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
        env = ctx.env
        self._visit_expr(stmt.expr, ctx)
        return self._visit_binding(stmt.target, env)

    def _visit_indexed_assign(self, stmt: IndexedAssign, ctx: _Ctx):
        env = ctx.env
        self._mark_use(stmt.var, env)
        for s in stmt.indices:
            self._visit_expr(s, ctx)
        self._visit_expr(stmt.expr, ctx)
        return env

    def _visit_if1(self, stmt: If1Stmt, ctx: _Ctx):
        env = ctx.env
        self._visit_expr(stmt.cond, ctx)
        ift_env = self._visit_block(stmt.body, ctx)
        return env.merge(ift_env)

    def _visit_if(self, stmt: IfStmt, ctx: _Ctx):
        self._visit_expr(stmt.cond, ctx)
        ift_env = self._visit_block(stmt.ift, ctx)
        iff_env = self._visit_block(stmt.iff, ctx)
        return ift_env.merge(iff_env)

    def _visit_while(self, stmt: WhileStmt, ctx: _Ctx):
        env = ctx.env
        body_env = self._visit_block(stmt.body, ctx)
        env = env.merge(body_env)
        self._visit_expr(stmt.cond, _Ctx(env, False))
        return env

    def _visit_for(self, stmt: ForStmt, ctx: _Ctx):
        env = ctx.env
        self._visit_expr(stmt.iterable, ctx)
        env = self._visit_binding(stmt.target, env)
        body_env = self._visit_block(stmt.body, _Ctx(env, False))
        return env.merge(body_env)

    def _visit_context(self, stmt: ContextStmt, ctx: _Ctx):
        env = ctx.env
        self._visit_expr(stmt.ctx, ctx)
        if isinstance(stmt.target, NamedId):
            env = env.extend(stmt.target)
        body_env = self._visit_block(stmt.body, _Ctx(env, False))
        return body_env # no merge

    def _visit_assert(self, stmt: AssertStmt, ctx: _Ctx):
        env = ctx.env
        self._visit_expr(stmt.test, ctx)
        if stmt.msg is not None:
            self._visit_expr(stmt.msg, ctx)
        return env

    def _visit_effect(self, stmt: EffectStmt, ctx: _Ctx):
        env = ctx.env
        self._visit_expr(stmt.expr, ctx)
        return env

    def _visit_return(self, stmt: ReturnStmt, ctx: _Ctx):
        return self._visit_expr(stmt.expr, ctx)

    def _visit_pass(self, stmt: PassStmt, ctx: _Ctx):
        return ctx.env

    def _visit_block(self, block: StmtBlock, ctx: _Ctx):
        env = ctx.env
        for stmt in block.stmts:
            env = self._visit_statement(stmt, _Ctx(env, False))
        return env

    def _visit_function(self, func: FuncDef, ctx: _Ctx):
        env = ctx.env
        for var in self.free_vars:
            env = env.extend(var)
        for arg in func.args:
            if isinstance(arg.name, NamedId):
                env = env.extend(arg.name)
        return self._visit_block(func.body, _Ctx(env, False))

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

    If statements:

    - any variable must be defined along both branches when
      used after the `if` statement

    Function calls:

    - the function part of a call must be either a variable
      or an attribute access (e.g., `obj.method`)
    """

    @staticmethod
    def check(
        func: FuncDef,
        *,
        free_vars: set[NamedId] | None = None,
        ignore_unknown: bool = True,
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

        inst = SyntaxCheckInstance(func, free_vars, ignore_unknown, allow_wildcard)
        return inst.analyze()
