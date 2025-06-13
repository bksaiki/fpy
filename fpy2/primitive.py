"""FPy primitives are the result of `@fpy_prim` decorators."""

from typing import Any, Callable, Optional, TYPE_CHECKING

from .utils import has_keyword
from .number import Context

class Primitive:
    """
    FPy primitive.

    This object is created by the `@fpy_prim` decorator and
    represents arbitrary Python code that may be called from
    the FPy runtime.
    """

    func: Callable
    """the function implementing the primitive"""

    metadata: dict[str, Any]

    def __init__(self, func: Callable, metadata: dict[str, Any]):
        self.func = func
        self.metadata = metadata

    def __repr__(self):
        return f'{self.__class__.__name__}(func={self.func}, ...)'

    def __call__(self, *args, ctx: Optional[Context] = None):
        if has_keyword(self.func, 'ctx'):
            return self.func(*args, ctx=ctx)
        else:
            return self.func(*args)
