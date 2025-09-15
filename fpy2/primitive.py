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
    # type info
    arg_types: tuple[TypeAnn, ...]
    ret_type: TypeAnn
    # context info
    ctx: Context | str | None
    arg_ctxs: list | None
    ret_ctx: Context | str | tuple | None
    # metadata
    spec: Any | None
    meta: dict[str, Any]

    def __init__(
        self,
        func: Callable[P, R],
        arg_types: Sequence[TypeAnn],
        return_type: TypeAnn,
        ctx: Context | str | None = None,
        arg_ctxs: list | None = None,
        ret_ctx: Context | str | tuple | None = None,
        spec: Any | None = None,
        meta: dict[str, Any] | None = None
    ):
        if meta is None:
            meta = {}

        self.func = func
        self.arg_types = tuple(arg_types)
        self.ret_type = return_type
        self.ctx = ctx
        self.arg_ctxs = arg_ctxs
        self.ret_ctx = ret_ctx
        self.spec = spec
        self.meta = meta

    def __repr__(self):
        return f'{self.__class__.__name__}(func={self.func}, ...)'

    def __call__(self, *args, ctx: Context = FP64):
        if has_keyword(self.func, 'ctx'):
            return self.func(*args, ctx=ctx)
        else:
            return self.func(*args)
