"""
Decorators for the FPy language.
"""

import builtins
import functools
import inspect
from collections.abc import Callable
from typing import Any, overload

from .analysis import Reachability, SyntaxCheck
from .ast import EffectStmt, FuncMeta, NamedId
from .env import ForeignEnv
from .frontend import Parser
from .function import Function
from .number import Context
from .primitive import Primitive
from .rewrite import ExprPattern, StmtPattern
from .utils import getfunclines

###########################################################
# @fpy decorator

@overload
def fpy[**P, R](func: Callable[P, R]) -> Function[P, R]:
    ...

@overload
def fpy[**P, R](
    *,
    ctx: Context | None = None,
    pre: Any = None,
    spec: Any = None,
    meta: dict[str, Any] | None = None,
) -> Callable[[Callable[P, R]], Function[P, R]]:
    ...

def fpy[**P, R](
    func: Callable[P, R] | None = None,
    *,
    ctx: Context | None = None,
    pre: Any = None,
    spec: Any = None,
    meta: dict[str, Any] | None = None,
) -> Function[P, R] | Callable[[Callable[P, R]], Function[P, R]]:
    """
    Decorator to parse a Python function into FPy.

    Constructs an FPy `Function` from a Python function.
    FPy is a stricter subset of Python, so this decorator will reject
    any function that is not valid in FPy.
    
    Args:
        func: The function to decorate (when used without parentheses)
        pre: Optional precondition for the function
        spec: Optional specification for the function
        meta: Optional metadata dictionary for the function
    """

    if func is None:
        # create a new decorator to be applied directly
        return lambda func: _apply_fpy_decorator(func, ctx=ctx, pre=pre, spec=spec, meta=meta)
    else:
        return _apply_fpy_decorator(func, ctx=ctx, pre=pre, spec=spec, meta=meta)


###########################################################
# @pattern decorator

def pattern[**P, R](func: Callable[P, R]) -> ExprPattern | StmtPattern:
    """
    Decorator to parse a Python function into an FPy pattern.
    Constructs an FPy `Pattern` from a Python function.
    FPy is a stricter subset of Python, so this decorator will reject
    any function that is not valid in FPy.
    """
    fn = _apply_fpy_decorator(func, is_pattern=True)

    # check which pattern it is
    # TODO: should there be separate decorators?
    stmts = fn.ast.body.stmts
    if len(stmts) == 1 and isinstance(stmts[0], EffectStmt):
        return ExprPattern(fn.ast)
    else:
        return StmtPattern(fn.ast)

############################################################
# @fpy_primitive decorator

@overload
def fpy_primitive[**P, R](func: Callable[P, R]) -> Primitive[P, R]:
    ...

@overload
def fpy_primitive[**P, R](
    *,
    ctx: str | None = None,
    arg_ctxs: list[str | tuple] | None = None,
    ret_ctx: Context | str | tuple | None = None,
    spec: Any = None,
    meta: dict[str, Any] | None = None,
    pure: bool = True,
) -> Callable[[Callable[P, R]], Primitive[P, R]]:
    ...

def fpy_primitive[**P, R](
    func: Callable[P, R] | None = None,
    *,
    ctx: str | None = None,
    arg_ctxs: list[str | tuple] | None = None,
    ret_ctx: Context | str | tuple | None = None,
    spec: Any = None,
    meta: dict[str, Any] | None = None,
    pure: bool = True,
) -> Primitive[P, R] | Callable[[Callable[P, R]], Primitive[P, R]]:
    """
    Decorator to parse a Python function into an FPy primitive.
    Constructs an FPy `Primitive` from a Python function.

    Primitives are Python functions that can be called from the FPy runtime.

    Args:
        func: The function to decorate (when used without parentheses)
        spec: Optional specification for the primitive
        meta: Optional metadata dictionary for the primitive
    """
    decorate = functools.partial(
        _apply_fpy_prim_decorator,
        ctx=ctx, arg_ctxs=arg_ctxs, ret_ctx=ret_ctx, spec=spec, meta=meta, pure=pure,
    )
    return decorate if func is None else decorate(func)

###########################################################
# Utilities

def _trim_source(lines: list[str], col_offset: int) -> None:
    if col_offset > 0:
        for i in range(len(lines)):
            lines[i] = lines[i][col_offset:]

def _function_env(func: Callable) -> ForeignEnv:
    globs = func.__globals__
    built_ins = {
        name: getattr(builtins, name)
        for name in dir(builtins)
        if not name.startswith("__")
    }

    if func.__closure__ is None:
        nonlocals = {}
    else:
        nonlocals = {
            v: c for v, c in
            zip(func.__code__.co_freevars, func.__closure__)
        }

    return ForeignEnv(globs, nonlocals, built_ins)


def _closure_names(func: Callable, env: ForeignEnv) -> set[str]:
    """Every name *func*'s body may read from its defining scope.

    ``co_names`` also holds attribute names, so ``fp.logb`` contributes
    ``logb``; membership in *env* is the filter, as it is in
    ``getclosurevars``.
    """
    return set(inspect.getclosurevars(func).nonlocals) | {
        n for n in func.__code__.co_names if n in env
    }


def _apply_fpy_decorator[**P, R](
    func: Callable[P, R],
    *,
    ctx: Context | None = None,
    pre: Any = None,
    spec: Any = None,
    meta: dict[str, Any] | None = None,
    is_pattern: bool = False,
) -> Function[P, R]:
    # fetch the source for the function
    lines, src_name, start_line, col_offset = getfunclines(func)
    _trim_source(lines, col_offset)

    # get defining environment
    env = _function_env(func)

    # set of free variables as `NamedId`
    free_vars = { NamedId(name) for name in _closure_names(func, env) }

    # parse the source as an FPy function
    parser = Parser(src_name, lines, env, start_line=start_line, col_offset=col_offset)
    ast, lambdas = parser.parse_function()

    # inline `lambda`s for `pre` and `spec` are parsed as FPy expressions
    pre = lambdas.get('pre', pre)
    spec = lambdas.get('spec', spec)
    if inspect.isfunction(pre) or inspect.isfunction(spec):
        raise TypeError(f'`pre` and `spec` of `{func.__name__}` must be inline `lambda`s or FPy functions')

    if is_pattern:
        free_vars = SyntaxCheck.check(ast, free_vars=free_vars, ignore_unknown=True, allow_wildcard=True)
    else:
        free_vars = SyntaxCheck.check(ast, free_vars=free_vars)
    ast._meta = FuncMeta(free_vars, ctx, pre, spec, meta or {}, env)

    if not is_pattern:
        # Multiple return statements are allowed: they're well-defined
        # for the interpreter and the cpp backend, and the fpc backend
        # rejects them at compile time with a backend-specific error.
        # We still require every statement to be reachable and every
        # path to end in a return.
        Reachability.analyze(
            ast,
            check_all_reachable=True,
            check_no_fallthrough=True,
        )

    # wrap the AST in a Function
    return Function(ast)

def _free_context_vars(ctx: object) -> set[str] | None:
    """The context variables in *ctx*, or `None` where it is not a context."""
    match ctx:
        case str():
            return {ctx}
        case Context():
            return set()
        case tuple():
            vars: set[str] = set()
            for c in ctx:
                c_vars = _free_context_vars(c)
                if c_vars is None:
                    return None
                vars |= c_vars
            return vars
        case _:
            return None

def _check_prim_contexts(
    num_args: int,
    ctx: str | None,
    arg_ctxs: list[str | tuple] | None,
    ret_ctx: Context | str | tuple | None,
) -> None:
    input_vars: set[str] = set()

    # check contexts are well-formed and that ret_ctx has
    # no free context variables
    if ctx is not None:
        ctx_vars = _free_context_vars(ctx)
        if ctx_vars is None:
            raise ValueError(f"invalid context: {ctx}")
        input_vars |= ctx_vars
    if arg_ctxs is not None:
        if not isinstance(arg_ctxs, list):
            raise TypeError(f"Expected \'list\': arg_ctxs={arg_ctxs}")
        if len(arg_ctxs) != num_args:
            raise ValueError(f"arg_ctxs length mismatch: expected {num_args}, got {len(arg_ctxs)}")
        for c in arg_ctxs:
            c_vars = _free_context_vars(c)
            if c_vars is None:
                raise ValueError(f"invalid context in arg_ctxs: {c}")
            input_vars |= c_vars
    if ret_ctx is not None:
        ret_vars = _free_context_vars(ret_ctx)
        if ret_vars is None:
            raise ValueError(f"invalid context in ret_ctx: {ret_ctx}")
        for v in ret_vars:
            if v not in input_vars:
                raise ValueError(f"unbound context variable in ret_ctx: {v}")


def _apply_fpy_prim_decorator[**P, R](
    func: Callable[P, R],
    *,
    ctx: str | None = None,
    arg_ctxs: list[str | tuple] | None = None,
    ret_ctx: Context | str | tuple | None = None,
    spec: Any = None,
    meta: dict[str, Any] | None = None,
    pure: bool = True,
) -> Primitive[P, R]:
    """
    Applies the `@fpy_prim` decorator to a function.
    """
    # reparse for the typing annotations
    lines, src_name, start_line, col_offset = getfunclines(func)
    _trim_source(lines, col_offset)

    # parse for the type signature
    env = _function_env(func)
    parser = Parser(src_name, lines, env, start_line=start_line, col_offset=col_offset)
    arg_types, return_type = parser.parse_signature()

    # check primitive context signature
    _check_prim_contexts(len(arg_types), ctx, arg_ctxs, ret_ctx)

    # create primitive
    return Primitive(
        func,
        arg_types,
        return_type,
        ctx=ctx,
        arg_ctxs=arg_ctxs,
        ret_ctx=ret_ctx,
        spec=spec,
        meta=meta,
        pure=pure
    )
