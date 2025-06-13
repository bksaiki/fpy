"""
Decorators for the FPy language.
"""

import builtins
import inspect
import textwrap

from typing import (
    Any,
    Callable,
    Optional,
    overload,
    ParamSpec,
    TypeVar
)

from .analysis import SyntaxCheck
from .ast import EffectStmt, NamedId
from .env import ForeignEnv
from .frontend import Parser
from .function import Function
from .primitive import Primitive
from .rewrite import ExprPattern, StmtPattern


P = ParamSpec('P')
R = TypeVar('R')

###########################################################
# @fpy decorator

@overload
def fpy(func: Callable[P, R]) -> Function:
    ...

@overload
def fpy(**kwargs) -> Callable[[Callable[P, R]], Function]:
    ...

def fpy(
    func: Optional[Callable[P, R]] = None,
    **kwargs
):
    """
    Decorator to parse a Python function into FPy.

    Constructs an FPy `Function` from a Python function.
    FPy is a stricter subset of Python, so this decorator will reject
    any function that is not valid in FPy.
    """
    if func is None:
        # create a new decorator to be applied directly
        return lambda func: _apply_fpy_decorator(func, kwargs)
    else:
        return _apply_fpy_decorator(func, kwargs)


###########################################################
# @pattern decorator

def pattern(func: Callable[P, R]):
    """
    Decorator to parse a Python function into an FPy pattern.
    Constructs an FPy `Pattern` from a Python function.
    FPy is a stricter subset of Python, so this decorator will reject
    any function that is not valid in FPy.
    """
    fn = _apply_fpy_decorator(func, {}, decorator=pattern, is_pattern=True)

    # check which pattern it is
    # TODO: should there be separate decorators?
    stmts = fn.ast.body.stmts
    if len(stmts) == 1 and isinstance(stmts[0], EffectStmt):
        return ExprPattern(fn.ast)
    else:
        return StmtPattern(fn.ast)

############################################################
# @fpy_prim decorator

@overload
def fpy_prim(func: Callable[P, R]) -> Primitive:
    ...

@overload
def fpy_prim(**kwargs) -> Callable[[Callable[P, R]], Primitive]:
    ...

def fpy_prim(
    func: Optional[Callable[P, R]] = None,
    **kwargs
):
    """
    Decorator to parse a Python function into an FPy primitive.
    Constructs an FPy `Primitive` from a Python function.

    Primitives are Python functions that can be called from the FPy runtime.
    """
    if func is None:
        # create a new decorator to be applied directly
        return lambda func: _apply_fpy_prim_decorator(func, kwargs)
    else:
        # parse the function as an FPy primitive
        return _apply_fpy_prim_decorator(func, kwargs)

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
    kwargs: dict[str, Any],
    *,
    decorator: Callable = fpy,
    is_pattern: bool = False
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
    parser = Parser(src_name, src, start_line)
    ast, decorator_list = parser.parse_function()

    # try to reparse the @fpy decorator
    dec_ast = parser.find_decorator(
        decorator_list,
        decorator,
        globals=func.__globals__,
        locals=cvars.nonlocals
    )

    # parse any relevant properties from the decorator
    props = parser.parse_decorator(dec_ast)

    # strictness
    strict = kwargs.get('strict', True)


    # function may have a global context
    if 'context' in kwargs:
        ast.ctx = kwargs['context']

    # add context information
    ast.metadata = { **kwargs, **props }

    # syntax checkng (and compute relevant free vars)
    if is_pattern:
        ast.free_vars = SyntaxCheck.check(
            ast,
            free_vars=free_vars,
            ignore_unknown=True,
            ignore_noreturn=True,
            allow_wildcard=True
        )
    else:
        ast.free_vars = SyntaxCheck.check(ast, free_vars=free_vars, ignore_unknown=not strict)

    # wrap the IR in a Function
    return Function(ast, env)


def _apply_fpy_prim_decorator(
    func: Callable[P, R],
    kwargs: dict[str, Any]
):
    """
    Applies the `@fpy_prim` decorator to a function.
    """
    return Primitive(func, kwargs)
