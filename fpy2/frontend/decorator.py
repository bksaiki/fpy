"""
Decorators for the FPy language.
"""

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
from .definition import DefinitionAnalysis
from .fpyast import FunctionDef
from .live_vars import LiveVarAnalysis
from .parser import Parser
from .syntax_check import SyntaxCheck

from ..passes import SSA, VerifyIR
from ..runtime import Function, ForeignEnv

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
    if func.__closure__ is None:
        nonlocals = {}
    else:
        nonlocals = {
            v: c for v, c in
            zip(func.__code__.co_freevars, func.__closure__)
        }

    return ForeignEnv(globs, nonlocals)

def _apply_decorator(func: Callable[P, R], kwargs: dict[str, Any]):
    # read the original source the function
    src_name = inspect.getabsfile(func)
    _, start_line = inspect.getsourcelines(func)
    src = textwrap.dedent(inspect.getsource(func))

    # get defining environment
    cvars = inspect.getclosurevars(func)
    fvs = cvars.nonlocals.keys() | cvars.globals.keys()
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
    ast.fvs = fvs

    # analyze and lower to the IR
    SyntaxCheck.analyze(ast)
    DefinitionAnalysis.analyze(ast)
    LiveVarAnalysis.analyze(ast)
    ir = IRCodegen.lower(ast)
    ir = SSA.apply(ir)
    VerifyIR.check(ir)

    # wrap the IR in a Function
    return Function(ir, env)
