"""
Records every value each expression of an FPy program takes.

A site is `(call path, expression)`: a helper's expressions are one site per
chain of call sites reaching them, as if the program were fully inlined, which
matches format inference's one analysis per call site (`by_call`).
"""

import ast as pyast
from collections import defaultdict
from collections.abc import Callable
from typing import Any

import fpy2 as fp
from fpy2.ast.fpyast import Call, Expr, FuncDef
from fpy2.env import ForeignEnv
from fpy2.interpret import get_default_interpreter
from fpy2.interpret.byte import CTX_NAME, BytecodeCompiler
from fpy2.interpret.value import from_value, to_value

Site = tuple[tuple[Call, ...], Expr]
"""An expression, and the calls it was reached through, outermost first."""

Record = Callable[[Site, Any], None]
"""Called with each site and each value it takes."""


class TracingCompiler(BytecodeCompiler):
    """The bytecode compiler, with every expression reporting its value to
    *record*, and every call to an FPy function compiled the same way."""

    _call: tuple[Call, ...]
    _record: Record

    def __init__(
        self, func: FuncDef, env: ForeignEnv, record: Record,
        *, call: tuple[Call, ...] = (),
    ) -> None:
        super().__init__(func, env, hook=self._observe)
        self._call = call
        self._record = record

    def _observe(self, i: int, v: Any) -> Any:
        self._record((self._call, self.hook_sites[i]), v)
        return v

    def _visit_call(self, e: Call, ctx: None) -> pyast.expr:
        fn = e.fn
        if not isinstance(fn, fp.Function):
            return super()._visit_call(e, ctx)
        callee = TracingCompiler(fn.ast, fn.env, self._record, call=(*self._call, e)).compile()
        interp = get_default_interpreter()
        name = str(self.gensym.fresh('__fpy_traced'))
        self.foreign_vals[name] = lambda c, *args: callee(*args, __ctx__=interp._func_ctx(fn.ast, c))
        attrs = self._location_to_attributes(e.loc)
        args = [self._visit_expr(arg, ctx) for arg in e.args]
        return pyast.Call(pyast.Name(name, pyast.Load(), **attrs),
                          [pyast.Name(CTX_NAME, pyast.Load(), **attrs), *args], [], **attrs)


def tracer(fn: fp.Function, record: Record) -> Callable[..., Any]:
    """*fn*, run under the real context, reporting each value to *record*."""
    run = TracingCompiler(fn.ast, fn.env, record).compile()
    ctx = get_default_interpreter()._func_ctx(fn.ast, fp.REAL)
    return lambda *args: from_value(run(*map(to_value, args), __ctx__=ctx))


def _snapshot(v: Any) -> Any:
    """*v*, with its lists copied: a later write through a shared list would
    otherwise change what was recorded."""
    if isinstance(v, list):
        return [_snapshot(x) for x in v]
    if isinstance(v, tuple):
        return tuple(_snapshot(x) for x in v)
    return v


def trace(fn: fp.Function, *args: Any) -> tuple[Any, dict[Site, list[Any]]]:
    """*fn*'s result on *args*, and each value each site took, in order."""
    seen: dict[Site, list[Any]] = defaultdict(list)
    result = tracer(fn, lambda s, v: seen[s].append(_snapshot(v)))(*args)
    return result, dict(seen)
