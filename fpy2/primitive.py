"""FPy primitives are the result of `@fpy_prim` decorators."""

from typing import Any, Callable, Generic, Iterable, ParamSpec, TypeVar

from .ast import TypeAnn
from .utils import has_keyword
from .number import Context, Float, FP64, INTEGER

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
    ctx: str | None
    arg_ctxs: list[str | tuple] | None
    ret_ctx: Context | str | tuple | None
    # metadata
    spec: Any | None
    meta: dict[str, Any] | None

    def __init__(
        self,
        func: Callable[P, R],
        arg_types: Iterable[TypeAnn],
        return_type: TypeAnn,
        ctx: str | None = None,
        arg_ctxs: list[str | tuple] | None = None,
        ret_ctx: Context | str | tuple | None = None,
        spec: Any | None = None,
        meta: dict[str, Any] | None = None
    ):
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
        args = tuple(self._arg_to_value(arg) for arg in args)
        if has_keyword(self.func, 'ctx'):
            return self.func(*args, ctx=ctx)
        else:
            return self.func(*args)

    @property
    def name(self) -> str:
        """The name of the primitive function."""
        return self.func.__name__

    def _arg_to_value(self, arg: Any):
        """
        Converts a Python argument to an FPy value.

        Copied from `fpy2/interpret/default.py`.
        """
        match arg:
            case Float():
                return arg
            case int():
                return Float.from_int(arg, ctx=INTEGER, checked=False)
            case float():
                return Float.from_float(arg, ctx=FP64, checked=False)
            case tuple():
                return tuple(self._arg_to_value(x) for x in arg)
            case list():
                return [self._arg_to_value(x) for x in arg]
            case _:
                return arg
