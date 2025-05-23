"""
Defines the abstract base class for FPy interpreters.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

from ..function import Function, set_default_function_call
from ..ir import Expr
from ..number import Context
from ..runtime.trace import ExprTraceEntry


class Interpreter(ABC):
    """Abstract base class for FPy interpreters."""

    @abstractmethod
    def eval(self, func: Function, args, ctx: Optional[Context] = None):
        ...

    @abstractmethod
    def eval_with_trace(self, func: Function, args, ctx: Optional[Context] = None) -> tuple[Any, list[ExprTraceEntry]]:
        ...

    @abstractmethod
    def eval_expr(self, expr: Expr, env: dict, ctx: Context):
        ...

class FunctionReturnException(Exception):
    """Raised when a function returns a value."""

    def __init__(self, value):
        self.value = value

###########################################################
# Default interpreter

_default_interpreter: Optional[Interpreter] = None

def get_default_interpreter() -> Interpreter:
    """Get the default FPy interpreter."""
    global _default_interpreter
    if _default_interpreter is None:
        raise RuntimeError('no default interpreter available')
    return _default_interpreter

def set_default_interpreter(rt: Interpreter):
    """Sets the default FPy interpreter"""
    global _default_interpreter
    if not isinstance(rt, Interpreter):
        raise TypeError(f'expected BaseInterpreter, got {rt}')
    _default_interpreter = rt

###########################################################
# Default function call

def _default_function_call(fn: Function, *args, ctx: Optional[Context] = None):
    """Default function call."""
    if fn.runtime is None:
        rt = get_default_interpreter()
    else:
        rt = fn.runtime
    return rt.eval(fn, args, ctx)


set_default_function_call(_default_function_call)
