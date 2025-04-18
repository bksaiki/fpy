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

from .codegen import IRCodegen
from .parser import Parser
from .syntax_check import SyntaxCheck

from ..analysis import VerifyIR
from ..runtime import Function, ForeignEnv
from ..transform import SSA

P = ParamSpec('P')
R = TypeVar('R')

@overload
def fpy(func: Callable[P, R]) -> Callable[P, R]:
    ...

@overload
def fpy(**kwargs) -> Callable[[Callable[P, R]], Callable[P, R]]:
    ...

def fpy(
    func: Optional[Callable[P, R]] = None,
    **kwargs
) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Decorator to parse a Python function into FPy.

    Constructs an FPy `Function` from a Python function.
    FPy is a stricter subset of Python, so this decorator will reject
    any function that is not valid in FPy.
    """
    if func is None:
        # create a new decorator to be applied directly
        return lambda func: _apply_decorator(func, kwargs)
    else:
        return _apply_decorator(func, kwargs)

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

def _apply_decorator(func: Callable[P, R], kwargs: dict[str, Any]):
    # read the original source the function
    src_name = inspect.getabsfile(func)
    _, start_line = inspect.getsourcelines(func)
    src = textwrap.dedent(inspect.getsource(func))

    # get defining environment
    cvars = inspect.getclosurevars(func)
    free_vars = cvars.nonlocals.keys() | cvars.globals.keys() | cvars.builtins.keys()
    env = _function_env(func)

    # parse the source as an FPy function
    parser = Parser(src_name, src, start_line)
    ast, decorator_list = parser.parse_function()

    # try to reparse the @fpy decorator
    dec_ast = parser.find_decorator(
        decorator_list,
        fpy,
        globals=func.__globals__,
        locals=cvars.nonlocals
    )

    # parse any relevant properties from the decorator
    props = parser.parse_decorator(dec_ast)

    # add context information
    ast.ctx = { **kwargs, **props }

    # syntax checking (and compute relevant free vars)
    ast.free_vars = SyntaxCheck.analyze(ast, free_vars=free_vars)

    # analyze and lower to the IR
    ir = IRCodegen.lower(ast)
    ir = SSA.apply(ir)
    VerifyIR.check(ir)

    # wrap the IR in a Function
    return Function(ir, env)
