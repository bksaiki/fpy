"""
Constant folding.
"""

import inspect

from fractions import Fraction
from typing import Any, TypeAlias

from ..analysis import DefineUse, DefineUseAnalysis, Definition
from ..ast.fpyast import *
from ..ast.visitor import DefaultTransformVisitor
from ..env import ForeignEnv
from ..fpc_context import FPCoreContext
from ..number import Float, RealFloat, REAL
from ..utils import is_dyadic


ScalarValue: TypeAlias = bool | Float | Fraction | Context
TupleValue: TypeAlias = tuple['Value', ...]
Value: TypeAlias = ScalarValue | TupleValue

class _ConstFoldInstance(DefaultTransformVisitor):
    """
    Constant folding instance for a function.
    """

    func: FuncDef
    env: ForeignEnv
    def_use: DefineUseAnalysis
    vals: dict[Definition, Value]

    def __init__(self, func: FuncDef, env: ForeignEnv, def_use: DefineUseAnalysis):
        self.func = func
        self.env = env
        self.def_use = def_use
        self.vals = {}

    def _is_value(self, e: Expr) -> bool:
        match e:
            case BoolVal() | RationalVal() | ForeignVal():
                return True
            case Attribute():
                return self._is_value(e.value)
            case TupleExpr():
                return all(self._is_value(elt) for elt in e.elts)
            case _:
                return False

    def _value(self, e: Expr) -> Value:
        match e:
            case BoolVal() | ForeignVal():
                return e.val
            case RationalVal():
                return e.as_rational()
            case Attribute():
                val = self._value(e.value)
                if hasattr(val, e.attr):
                    return getattr(val, e.attr)
                elif isinstance(val, dict) and e.attr in val:
                    return val[e.attr]
                else:
                    raise RuntimeError(f'unknown attribute `{e.attr}` for `{val}`')
            case TupleExpr():
                return tuple(self._value(elt) for elt in e.elts)
            case _:
                raise NotImplementedError(e)

    def _cvt_float(self, x: Value) -> Float:
        match x:
            case Float():
                return x
            case Fraction():
                if not is_dyadic(x):
                    raise TypeError(f'expected a dyadic rational, got `{x}`')
                return Float.from_rational(x, ctx=REAL)
            case _:
                raise TypeError(f'expected a real number, got `{x}`')

    def _cvt_context_arg(self, cls: type[Context], name: str, arg: Any, ty: type):
        if ty is int:
            # convert to int
            val = self._cvt_float(arg)
            if not val.is_integer():
                raise TypeError(f'expected an integer argument for `{name}={arg}`')
            return int(val)
        elif ty is float:
            # convert to float
            raise NotImplementedError(arg, ty)
        elif ty is RealFloat:
            # convert to RealFloat
            raise NotImplementedError(arg, ty)
        else:
            # don't apply a conversion
            return arg

    def _construct_context(self, cls: type[Context], args: list[Value], kwargs: dict[str, Value]):
        sig = inspect.signature(cls.__init__)
        params = list(sig.parameters)

        ctor_args = []
        for arg, name in zip(args, params[1:]):
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

    def _visit_call(self, e: Call, ctx: Context | None):
        args = [ self._visit_expr(arg, ctx) for arg in e.args ]
        kwargs = [ (k, self._visit_expr(v, ctx)) for k, v in e.kwargs ]
        if (
            isinstance(e.fn, type)
            and issubclass(e.fn, Context)
            and all(self._is_value(arg) for arg in args)
            and all(self._is_value(v) for _, v in kwargs)
        ):
            # context constructor
            vals = [ self._value(arg) for arg in args ]
            kwvals = { k: self._value(v) for k, v in kwargs }
            val = self._construct_context(e.fn, vals, kwvals)
            return ForeignVal(val, e.loc)
        else:
            # otherwise
            return Call(e.func, e.fn, args, kwargs, e.loc)


    def _visit_function(self, func: FuncDef, ctx: None):
        # extract overriding context
        match func.ctx:
            case None:
                fctx: Context | None = None
            case FPCoreContext():
                fctx = func.ctx.to_context()
            case Context():
                fctx = func.ctx
            case _:
                raise RuntimeError(f'unreachable: {func.ctx}')

        # bind foreign values
        for name in func.free_vars:
            d = self.def_use.find_def_from_site(name, func)
            self.vals[d] = self.env[name.base]

        return super()._visit_function(func, fctx)

    def apply(self) -> FuncDef:
        return self._visit_function(self.func, None)


class ConstFold:
    """
    Constant folding.
    """

    @staticmethod
    def apply(func: FuncDef, env: ForeignEnv, *, def_use: DefineUseAnalysis | None = None) -> FuncDef:
        if not isinstance(func, FuncDef):
            raise TypeError(f'Expected `FuncDef`, got {type(func)} for {func}')
        if def_use is None:
            def_use = DefineUse.analyze(func)
        return _ConstFoldInstance(func, env, def_use).apply()
