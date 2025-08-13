"""FPy primitives are the result of `@fpy_prim` decorators."""

from typing import Any, Callable, Generic, ParamSpec, Sequence, TypeVar

from .ast import TypeAnn
from .utils import has_keyword
from .number import Context, FP64

P = ParamSpec('P')
R = TypeVar('R')

class Primitive(Generic[P, R]):
    """
    FPy primitive.

    This object is created by the `@fpy_prim` decorator and
    represents arbitrary Python code that may be called from
    the FPy runtime.
    """

    func: Callable[..., R]

    arg_types: tuple[TypeAnn, ...]

    return_type: TypeAnn

    metadata: dict[str, Any]

    def __init__(
        self,
        func: Callable[P, R],
        arg_types: Sequence[TypeAnn],
        return_type: TypeAnn,
        metadata: dict[str, Any]
    ):
        self.func = func
        self.arg_types = tuple(arg_types)
        self.return_type = return_type
        self.metadata = metadata

    def __repr__(self):
        return f'{self.__class__.__name__}(func={self.func}, ...)'

    def __call__(self, *args, ctx: Context = FP64):
        if has_keyword(self.func, 'ctx'):
            return self.func(*args, ctx=ctx)
        else:
            return self.func(*args)
