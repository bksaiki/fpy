"""
Defines the abstract base class for FPy interpreters.
"""

from abc import ABC, abstractmethod
from typing import Optional

from ..fpc_context import FPCoreContext
from ..function import Function, set_default_function_call
from ..number import Context, IEEEContext, RM


_PY_CTX = IEEEContext(11, 64, RM.RNE)
"""the native Python floating-point context"""


class Interpreter(ABC):
    """Abstract base class for FPy interpreters."""

    @abstractmethod
    def eval(self, func: Function, args, ctx: Optional[Context] = None):
        ...

    def _func_ctx(self, func: Function, ctx: Optional[Context] = None) -> Context:
        """
        Computes the context to use during evaluation.

        If `func` specifies a context, it will be used.
        Otherwise, the provided context `ctx` will be used.
        """
        if func.ast.ctx is None:
            if ctx is None:
                ctx = _PY_CTX
        else:
            match func.ast.ctx:
                case Context():
                    ctx = func.ast.ctx
                case FPCoreContext():
                    ctx = func.ast.ctx.to_context()
                case _:
                    raise RuntimeError('unreachable', func.ast.ctx)

        return ctx


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
