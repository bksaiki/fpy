"""FPy functions are the result of `@fpy` decorators."""

from typing import Callable, Optional, TYPE_CHECKING
from titanfp.fpbench.fpcast import FPCore
from . import ast as fpyast

from .env import ForeignEnv
from .frontend import fpcore_to_fpy
from .number import Context

# avoids circular dependency issues (useful for type checking)
if TYPE_CHECKING:
    from .interpret import Interpreter


class Function:
    """
    FPy function.

    This object is created by the `@fpy` decorator and represents
    a function in the FPy runtime.
    """
    ast: fpyast.FuncDef
    env: ForeignEnv
    runtime: Optional['Interpreter']

    _func: Optional[Callable]
    """original native function"""

    def __init__(
        self,
        ast: fpyast.FuncDef,
        env: ForeignEnv,
        runtime: Optional['Interpreter'] = None,
        func: Optional[Callable] = None
    ):
        self.ast = ast
        self.env = env
        self.runtime = runtime
        self._ir = None
        self._func = func

    def __repr__(self):
        return f'{self.__class__.__name__}(ast={self.ast}, ...)'

    def __str__(self):
        return self.ast.format()

    def __call__(self, *args, ctx: Optional[Context] = None):
        fn = get_default_function_call()
        return fn(self, *args, ctx=ctx)

    def format(self):
        return self.ast.format()

    @property
    def args(self):
        return self.ast.args

    @property
    def name(self):
        return self.ast.name

    @staticmethod
    def from_fpcore(
        core: FPCore,
        *,
        default_name: str = 'f',
        ignore_unknown: bool = False
    ):
        """
        Converts an `FPCore` (from `titanfp`) to an `FPy` function.

        Optionally, specify `default_name` to set the name of the function.
        If `ignore_unknown` is set to `True`, then the syntax checker will not
        raise an exception when encountering unknown functions.
        """
        if not isinstance(core, FPCore):
            raise TypeError(f'expected FPCore, got {core}')
        ir = fpcore_to_fpy(core, default_name=default_name, ignore_unknown=ignore_unknown)
        return Function(ir, ForeignEnv.empty())

    def with_rt(self, rt: 'Interpreter'):
        if not isinstance(rt, Interpreter):
            raise TypeError(f'expected \'BaseInterpreter\', got {rt}')
        return Function(self.ast, self.env, runtime=rt, func=self._func)

    def with_ast(self, ast: fpyast.FuncDef):
        if not isinstance(ast, fpyast.FuncDef):
            raise TypeError(f'expected \'FuncDef\', got {ast}')
        return Function(ast, self.env, runtime=self.runtime, func=self._func)

###########################################################
# Default function call

_default_function_call: Optional[Callable] = None

def get_default_function_call() -> Callable:
    """Get the default function call."""
    global _default_function_call
    if _default_function_call is None:
        raise RuntimeError('no default function call available')
    return _default_function_call

def set_default_function_call(func: Callable):
    """Sets the default function call"""
    global _default_function_call
    if not callable(func):
        raise TypeError(f'expected callable, got {func}')
    _default_function_call = func
