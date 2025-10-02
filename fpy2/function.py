"""FPy functions are the result of `@fpy` decorators."""

from typing import Callable, Generic, Optional, ParamSpec, TypeVar, TYPE_CHECKING
from titanfp.fpbench.fpcast import FPCore

from . import ast as fpyast

from .env import ForeignEnv
from .number import Context

# avoids circular dependency issues (useful for type checking)
if TYPE_CHECKING:
    from .analysis.types import FunctionType
    from .interpret import Interpreter

P = ParamSpec('P')
R = TypeVar('R')


class Function(Generic[P, R]):
    """
    FPy function.

    This object is created by the `@fpy` decorator and represents
    a function in the FPy runtime.
    """

    ast: fpyast.FuncDef
    sig: Optional['FunctionType']
    runtime: Optional['Interpreter']

    def __init__(
        self,
        ast: fpyast.FuncDef,
        sig: Optional['FunctionType'],
        *,
        runtime: Optional['Interpreter'] = None,
    ):
        self.ast = ast
        self.sig = sig
        self.runtime = runtime

    def __repr__(self):
        return f'{self.__class__.__name__}(ast={self.ast}, ...)'

    def __str__(self):
        return self.ast.format()

    def __call__(self, *args, ctx: Context | None = None) -> R:
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

    @property
    def env(self):
        return self.ast.env

    @staticmethod
    def from_fpcore(
        core: FPCore,
        *,
        env: ForeignEnv | None = None,
        default_name: str = 'f',
        ignore_unknown: bool = False
    ):
        """
        Converts an `FPCore` (from `titanfp`) to an `FPy` function.

        Optionally, specify `default_name` to set the name of the function.
        If `ignore_unknown` is set to `True`, then the syntax checker will not
        raise an exception when encountering unknown functions.
        """
        # get around circular dependency
        from .frontend import fpcore_to_fpy

        if not isinstance(core, FPCore):
            raise TypeError(f'expected FPCore, got {core}')
        ir = fpcore_to_fpy(core, env=env, default_name=default_name, ignore_unknown=ignore_unknown)
        return Function(ir, None)

    def with_rt(self, rt: 'Interpreter'):
        if not isinstance(rt, Interpreter):
            raise TypeError(f'expected \'BaseInterpreter\', got {rt}')
        return Function(self.ast, self.sig, runtime=rt)

    def with_ast(self, ast: fpyast.FuncDef):
        if not isinstance(ast, fpyast.FuncDef):
            raise TypeError(f'expected \'FuncDef\', got {ast}')
        return Function(ast, self.sig, runtime=self.runtime)

###########################################################
# Default function call

_default_function_call: Callable | None = None

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
