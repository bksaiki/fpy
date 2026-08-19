"""FPy functions are the result of `@fpy` decorators."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Generic, Optional, ParamSpec, TypeVar

from titanfp.fpbench.fpcast import FPCore

from . import ast as fpyast
from .env import ForeignEnv
from .number import Context

if TYPE_CHECKING:
    # `interpret` imports `function`, so only import for type checking here;
    # runtime uses do a local import (see `with_rt`).
    from .interpret import Interpreter

    # `transform` imports `function` too (see `forward`)
    from .transform.utils.cursor import Block, Cursor, EditLog

P = ParamSpec('P')
R = TypeVar('R')


class Function(Generic[P, R]):
    """
    FPy function.

    This object is created by the `@fpy` decorator and represents
    a function in the FPy runtime.

    Example::

      @fp.fpy
      def my_function(x: fp.Real) -> fp.Real:
          return x * 2

    """

    ast: fpyast.FuncDef
    runtime: Optional['Interpreter']
    parent: Optional['Function']
    """the program this one was derived from, if any"""
    edits: Optional['EditLog']
    """what the pass that derived it did, if the pass says"""

    def __init__(
        self,
        ast: fpyast.FuncDef,
        *,
        runtime: Optional['Interpreter'] = None,
        parent: Optional['Function'] = None,
        edits: Optional['EditLog'] = None,
    ):
        self.ast = ast
        self.runtime = runtime
        self.parent = parent
        self.edits = edits

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
        return Function(ir)

    def with_rt(self, rt: 'Interpreter'):
        from .interpret import Interpreter
        if not isinstance(rt, Interpreter):
            raise TypeError(f'expected \'BaseInterpreter\', got {rt}')
        # the same program, so it keeps the same place in the chain
        return Function(self.ast, runtime=rt, parent=self.parent, edits=self.edits)

    def with_ast(self, ast: fpyast.FuncDef):
        """The result of a pass that does not say what it rewrote.

        Cursors do not forward across such a step; :meth:`with_edits` is the
        one that carries them.
        """
        if not isinstance(ast, fpyast.FuncDef):
            raise TypeError(f'expected \'FuncDef\', got {ast}')
        return Function(ast, runtime=self.runtime, parent=self)

    def with_edits(self, log: 'EditLog'):
        """The result of a pass that reported its rewrites, so a cursor of
        this program forwards across the step."""
        if log.source is not self.ast:
            raise ValueError('edit log was not produced from this program')
        return Function(log.result, runtime=self.runtime, parent=self, edits=log)

    def forward(self, cursor: 'Cursor | Block') -> 'Cursor | Block':
        """*cursor*, in this program.

        Walks back to the program the cursor names, then replays what each
        pass reported.  A pass that reported nothing stops the walk: a cursor
        cannot cross it, and guessing that it survived is exactly the silent
        mis-aim cursors exist to prevent.
        """
        from .transform.utils.error import TransformReferenceError

        logs: list[EditLog | None] = []
        f: Function | None = self
        while f is not None and f.ast is not cursor.func:
            logs.append(f.edits)
            f = f.parent
        if f is None:
            raise TransformReferenceError(
                f'`{cursor}` names a statement of an unrelated program'
            )

        out = cursor
        for log in reversed(logs):
            if log is None:
                raise TransformReferenceError(
                    f'`{cursor}` does not reach this program: a pass in between '
                    'does not report what it rewrote'
                )
            out = log.forward(out)
        return out

###########################################################
# Default function call

_default_function_call: Callable | None = None

def get_default_function_call() -> Callable:
    """Get the default function call."""
    if _default_function_call is None:
        raise RuntimeError('no default function call available')
    return _default_function_call

def set_default_function_call(func: Callable):
    """Sets the default function call"""
    global _default_function_call
    if not callable(func):
        raise TypeError(f'expected callable, got {func}')
    _default_function_call = func
