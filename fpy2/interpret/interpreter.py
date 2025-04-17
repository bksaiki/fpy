"""
Defines the abstract base class for FPy interpreters.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

from titanfp.arithmetic.evalctx import EvalCtx

from ..ir import Expr
from ..runtime import Function
from ..runtime.trace import ExprTraceEntry


class Interpreter(ABC):
    """Abstract base class for FPy interpreters."""

    @abstractmethod
    def eval(self, func: Function, args, ctx: Optional[EvalCtx] = None):
        raise NotImplementedError('virtual method')

    @abstractmethod
    def eval_with_trace(self, func: Function, args, ctx: Optional[EvalCtx] = None) -> tuple[Any, list[ExprTraceEntry]]:
        raise NotImplementedError('virtual method')

    @abstractmethod
    def eval_expr(self, expr: Expr, env: dict, ctx: EvalCtx):
        raise NotADirectoryError('virtual method')

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
