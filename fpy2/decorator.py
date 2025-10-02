"""
Decorators for the FPy language.
"""

import builtins
import inspect
import textwrap

from typing import Any, Callable, ParamSpec, TypeVar, overload

from .analysis import Reachability, SyntaxCheck
from .ast import EffectStmt, NamedId
from .env import ForeignEnv
from .frontend import Parser
from .function import Function
from .number import Context
from .primitive import Primitive
from .rewrite import ExprPattern, StmtPattern


P = ParamSpec('P')
R = TypeVar('R')

###########################################################
# @fpy decorator

@overload
def fpy(func: Callable[P, R]) -> Function[P, R]:
    ...

@overload
def fpy(
    *,
    ctx: Context | None = None,
    spec: Any = None,
    meta: dict[str, Any] | None = None,
) -> Callable[[Callable[P, R]], Function[P, R]]:
    ...

def fpy(
    func: Callable[P, R] | None = None,
    *,
    ctx: Context | None = None,
    spec: Any = None,
    meta: dict[str, Any] | None = None,
):
    """
    Decorator to parse a Python function into FPy.

    Constructs an FPy `Function` from a Python function.
    FPy is a stricter subset of Python, so this decorator will reject
    any function that is not valid in FPy.
    
    Args:
        func: The function to decorate (when used without parentheses)
        spec: Optional specification for the function
        meta: Optional metadata dictionary for the function
    """

    if func is None:
        # create a new decorator to be applied directly
        return lambda func: _apply_fpy_decorator(func, ctx=ctx, spec=spec, meta=meta)
    else:
        return _apply_fpy_decorator(func, ctx=ctx, spec=spec, meta=meta)


###########################################################
# @pattern decorator

def pattern(func: Callable[P, R]):
    """
    Decorator to parse a Python function into an FPy pattern.
    Constructs an FPy `Pattern` from a Python function.
    FPy is a stricter subset of Python, so this decorator will reject
    any function that is not valid in FPy.
    """
    fn = _apply_fpy_decorator(func, decorator=pattern)

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
def fpy_primitive(func: Callable[P, R]) -> Primitive[P, R]:
    ...

@overload
def fpy_primitive(
    *,
    ctx: str | None = None,
    arg_ctxs: list[str | tuple] | None = None,
    ret_ctx: Context | str | tuple | None = None,
    spec: Any = None,
    meta: dict[str, Any] | None = None,
) -> Callable[[Callable[P, R]], Primitive[P, R]]:
    ...

def fpy_primitive(
    func: Callable[P, R] | None = None,
    *,
    ctx: str | None = None,
    arg_ctxs: list[str | tuple] | None = None,
    ret_ctx: Context | str | tuple | None = None,
    spec: Any = None,
    meta: dict[str, Any] | None = None,
):
    """
    Decorator to parse a Python function into an FPy primitive.
    Constructs an FPy `Primitive` from a Python function.

    Primitives are Python functions that can be called from the FPy runtime.

    Args:
        func: The function to decorate (when used without parentheses)
        spec: Optional specification for the primitive
        meta: Optional metadata dictionary for the primitive
    """
    if func is None:
        # create a new decorator to be applied directly
        return lambda func: _apply_fpy_prim_decorator(
            func, 
            ctx=ctx,
            arg_ctxs=arg_ctxs,
            ret_ctx=ret_ctx,
            spec=spec,
            meta=meta
        )
    else:
        # parse the function as an FPy primitive
        return _apply_fpy_prim_decorator(
            func, 
            ctx=ctx,
            arg_ctxs=arg_ctxs,
            ret_ctx=ret_ctx,
            spec=spec,
            meta=meta
        )

###########################################################
# Utilities

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

def _apply_fpy_decorator(
    func: Callable[P, R],
    *,
    ctx: Context | None = None,
    spec: Any = None,
    meta: dict[str, Any] | None = None,
    decorator: Callable = fpy,
):
    # read the original source the function
    src_name = inspect.getabsfile(func)
    _, start_line = inspect.getsourcelines(func)
    src = textwrap.dedent(inspect.getsource(func))

    # get defining environment
    cvars = inspect.getclosurevars(func)
    cfree_vars = cvars.nonlocals.keys() | cvars.globals.keys() | cvars.builtins.keys()
    env = _function_env(func)

    # set of free variables as `NamedId`
    free_vars = { NamedId(name) for name in cfree_vars }

    # parse the source as an FPy function
    parser = Parser(src_name, src, env, start_line=start_line)
    ast, _ = parser.parse_function(env)

    # function may have a global context
    ast.ctx = ctx

    # spec and metadata
    ast.spec = spec
    ast.meta = meta

    if decorator == pattern:
        # syntax checking
        ast.free_vars = SyntaxCheck.check(
            ast,
            free_vars=free_vars,
            ignore_unknown=True,
            allow_wildcard=True
        )
    else:
        # syntax checking
        ast.free_vars = SyntaxCheck.check(ast, free_vars=free_vars)
        Reachability.analyze(ast, check=True)

    # wrap the IR in a Function
    return Function(ast, None)

def _is_valid_context(ctx: Context | str | tuple):
    match ctx:
        case tuple():
            return all(_is_valid_context(c) for c in ctx)
        case _:
            return isinstance(ctx, (Context, str))

def _free_context_vars(ctx: Context | str | tuple) -> set[str]:
    match ctx:
        case str():
            return {ctx}
        case Context():
            return set()
        case tuple():
            vars = set()
            for c in ctx:
                vars.update(_free_context_vars(c))
            return vars
        case _:
            raise ValueError(f"invalid context: {ctx}")

def _check_prim_contexts(
    num_args: int,
    ctx: str | None,
    arg_ctxs: list[str | tuple] | None,
    ret_ctx: Context | str | tuple | None,
):
    input_vars: set[str] = set()

    # check contexts are well-formed and that ret_ctx has
    # no free context variables
    if ctx is not None:
        if not _is_valid_context(ctx):
            raise ValueError(f"invalid context: {ctx}")
        input_vars.update(_free_context_vars(ctx))
    if arg_ctxs is not None:
        if not isinstance(arg_ctxs, list):
            raise TypeError(f"Expected \'list\': arg_ctxs={arg_ctxs}")
        if len(arg_ctxs) != num_args:
            raise ValueError(f"arg_ctxs length mismatch: expected {num_args}, got {len(arg_ctxs)}")
        for c in arg_ctxs:
            if not _is_valid_context(c):
                raise ValueError(f"invalid context in arg_ctxs: {c}")
            input_vars.update(_free_context_vars(c))
    if ret_ctx is not None:
        if not _is_valid_context(ret_ctx):
            raise ValueError(f"invalid context in ret_ctx: {ret_ctx}")
        for v in _free_context_vars(ret_ctx):
            if v not in input_vars:
                raise ValueError(f"unbound context variable in ret_ctx: {v}")


def _apply_fpy_prim_decorator(
    func: Callable[P, R],
    *,
    ctx: str | None = None,
    arg_ctxs: list[str | tuple] | None = None,
    ret_ctx: Context | str | tuple | None = None,
    spec: Any = None,
    meta: dict[str, Any] | None = None,
):
    """
    Applies the `@fpy_prim` decorator to a function.
    """

    # reparse for the typing annotations
    src_name = inspect.getabsfile(func)
    _, start_line = inspect.getsourcelines(func)
    src = textwrap.dedent(inspect.getsource(func))

    # parse for the type signature
    env = _function_env(func)
    parser = Parser(src_name, src, env, start_line=start_line)
    arg_types, return_type = parser.parse_signature(ignore_ctx=True)

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
        meta=meta
    )
