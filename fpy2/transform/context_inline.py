"""
Context inlining for FPy ASTs

TODO: this looks something like partial evaluation
- can the evaluator be adapted here?
- is this just part of a larger constant folding pass?
"""

import inspect

from typing import Any

from ..fpc_context import FPCoreContext
from ..number import Context, Float, RealFloat, REAL
from ..env import ForeignEnv

from ..ast.fpyast import *
from ..ast.visitor import DefaultTransformVisitor

class _ContextInlineInstance(DefaultTransformVisitor):
    """Single-use instance of context inlining"""
    ast: FuncDef
    env: ForeignEnv

    def __init__(self, ast: FuncDef, env: ForeignEnv):
        self.ast = ast
        self.env = env

    def apply(self):
        return self._visit_function(self.ast, None)

    def _lookup(self, name: NamedId):
        return self.env.get(name.base)

    def _eval_var(self, e: Var):
        return self._lookup(e.name)

    def _eval_attribute(self, e: Attribute):
        match e.value:
            case Var():
                value = self._eval_var(e.value)
            case Attribute():
                value = self._eval_attribute(e.value)
            case _:
                return None

        if value is None:
            return None
        elif isinstance(value, dict):
            if e.attr not in value:
                raise RuntimeError(f'unknown attribute {e.attr} for {value}')
            return value[e.attr]
        elif hasattr(value, e.attr):
            return getattr(value, e.attr)
        else:
            raise RuntimeError(f'unknown attribute {e.attr} for {value}')

    def _eval_call_arg(self, e: Expr):
        match e:
            case Var():
                return self._eval_var(e)
            case Attribute():
                return self._eval_attribute(e)
            case ForeignVal():
                return e.val
            case RationalVal():
                # must evaluate under real context
                return REAL.round(e.as_rational())
            case _:
                return None

    # TODO: copied from `fpy2/interpret/default.py`
    def _cvt_context_arg(self, cls: type[Context], name: str, arg: Any, ty: type):
        if ty is int:
            # convert to int
            if not isinstance(arg, Float) or not arg.is_integer():
                raise TypeError(f'expected an integer argument for `{name}={arg}`')
            return int(arg)
        elif ty is float:
            # convert to float
            raise NotImplementedError(arg, ty)
        elif ty is RealFloat:
            # convert to RealFloat
            raise NotImplementedError(arg, ty)
        else:
            # don't apply a conversion
            return arg

    # TODO: copied from `fpy2/interpret/default.py`
    def _construct_context(self, cls: type[Context], args: list[Any], kwargs: dict[str, Any]):
        sig = inspect.signature(cls.__init__)

        ctor_args = []
        for arg, name in zip(args, list(sig.parameters)[1:]):
            param = sig.parameters[name]
            ctor_arg = self._cvt_context_arg(cls, name, arg, param.annotation)
            ctor_args.append(ctor_arg)

        ctor_kwargs = {}
        for name, val in kwargs.items():
            if name not in sig.parameters:
                raise TypeError(f'unknown parameter {name} for constructor {cls}')
            param = sig.parameters[name]
            ctor_kwargs[name] = self._cvt_context_arg(cls, name, val, param.annotation)

        return cls(*ctor_args, **ctor_kwargs)

    def _eval_call(self, e: Call):
        match e.func:
            case NamedId():
                func = self._lookup(e.func)
            case Attribute():
                func = self._eval_attribute(e.func)
            case _:
                raise RuntimeError(f'unreachable: `{e.func}`')

        if isinstance(func, type) and issubclass(func, Context):
            # calling context constructor
            args = [self._eval_call_arg(arg) for arg in e.args]
            kwargs = { kw: self._eval_call_arg(value) for kw, value in e.kwargs }
            if any(arg is None for arg in args) or any(v is None for v in kwargs.values()):
                return None
            return self._construct_context(func, args, kwargs)
        else:
            return None

    def _visit_context(self, stmt: ContextStmt, ctx: None):
        # context expressions are implicitly evaluated under
        # a real context so we don't need to round
        match stmt.ctx:
            case Var():
                # if variables can be resolved to be a context,
                # replace the variable with the context value
                v = self._eval_var(stmt.ctx)
            case Attribute():
                # if attributes can be resolved to be a context,
                # replace the attribute with the context value
                v = self._eval_attribute(stmt.ctx)
            case Call():
                # if a context constructor may be evaluated, do so
                v = self._eval_call(stmt.ctx)
            case ForeignVal():
                # foreign values are already evaluated
                v = stmt.ctx.val
            case _:
                v = None

        body, _ = self._visit_block(stmt.body, None)
        if v is None:
            s = ContextStmt(stmt.target, stmt.ctx, body, stmt.loc)
        else:
            if not isinstance(v, Context | FPCoreContext):
                raise RuntimeError(f'expected a Context | FPCoreContext, got `{v}` for `{stmt.ctx.format()}`')
            ctx_val = ForeignVal(v, stmt.ctx.loc)
            s = ContextStmt(stmt.target, ctx_val, body, stmt.loc)

        return s, None


class ContextInline:
    """
    Context inliner.

    Contexts in FPy programs may be metaprogrammed.
    This pass resolves the context at each site.
    """

    @staticmethod
    def apply(ast: FuncDef, env: ForeignEnv) -> FuncDef:
        if not isinstance(ast, FuncDef):
            raise TypeError(f'Expected `FuncDef`, got {type(ast)} for {ast}')
        return _ContextInlineInstance(ast, env).apply()
