"""FPy functions are the result of `@fpy` decorators."""

from abc import abstractmethod
from typing import Optional
from types import FunctionType
from titanfp.fpbench.fpcast import FPCore
from titanfp.arithmetic.evalctx import EvalCtx

from .env import PythonEnv
from ..ir import FunctionDef
from ..frontend.fpc import fpcore_to_fpy

class Function:
    """
    FPy function.

    This object is created by the `@fpy` decorator and represents
    a function in the FPy runtime.
    """
    ir: FunctionDef
    env: PythonEnv
    runtime: Optional['Interpreter']

    _func: Optional[FunctionType]
    """original native function"""

    def __init__(
        self,
        ir: FunctionDef,
        env: PythonEnv,
        runtime: Optional['Interpreter'] = None,
        func: Optional[FunctionType] = None
    ):
        self.ir = ir
        self.env = env
        self.runtime = runtime
        self._func = func

    def __repr__(self):
        return f'{self.__class__.__name__}(ir={self.ir}, ...)'

    def __call__(self, *args, ctx: Optional[EvalCtx] = None):
        rt = get_default_interpreter() if self.runtime is None else self.runtime
        return rt.eval(self, args, ctx=ctx)

    def format(self):
        return self.ir.format()

    @property
    def args(self):
        return self.ir.args

    @property
    def name(self):
        return self.ir.name

    @staticmethod
    def from_fpcore(core: FPCore, default_name: str = 'f'):
        if not isinstance(core, FPCore):
            raise TypeError(f'expected FPCore, got {core}')
        ir = fpcore_to_fpy(core, default_name=default_name)
        return Function(ir, PythonEnv.empty())

    def with_rt(self, rt: 'Interpreter'):
        if not isinstance(rt, Interpreter):
            raise TypeError(f'expected BaseInterpreter, got {rt}')
        if not isinstance(self._func, FunctionType):
            raise TypeError(f'expected FunctionType, got {self._func}')
        return Function(self.ir, self.env, runtime=rt, func=self._func)

class Interpreter:
    """Abstract base class for FPy interpreters."""

    @abstractmethod
    def eval(self, func: Function, args, ctx: Optional[EvalCtx] = None):
        raise NotImplementedError('virtual method')


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


class FunctionReturnException(Exception):
    """Raised when a function returns a value."""

    def __init__(self, value):
        self.value = value
