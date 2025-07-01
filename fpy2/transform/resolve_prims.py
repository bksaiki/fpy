"""
Replaces library calls with primitive operations.
"""

from typing import Callable

from ..ast import *
from ..env import ForeignEnv
from ..ops import *

_unary_table: dict[Callable, type[UnaryOp]] = {
    fabs: Fabs,
    sqrt: Sqrt,
    cbrt: Cbrt,
    ceil: Ceil,
    floor: Floor,
    nearbyint: NearbyInt,
    round: Round,
    trunc: Trunc,
    acos: Acos,
    asin: Asin,
    atan: Atan,
    cos: Cos,
    sin: Sin,
    tan: Tan,
    acosh: Acosh,
    asinh: Asinh,
    atanh: Atanh,
    cosh: Cosh,
    sinh: Sinh,
    tanh: Tanh,
    exp: Exp,
    exp2: Exp2,
    expm1: Expm1,
    log: Log,
    log10: Log10,
    log1p: Log1p,
    log2: Log2,
    erf: Erf,
    erfc: Erfc,
    lgamma: Lgamma,
    tgamma: Tgamma,
    isfinite: IsFinite,
    isinf: IsInf,
    isnan: IsNan,
    isnormal: IsNormal,
    signbit: Signbit,
    cast: Cast,
    range: Range,
    dim: Dim,
    enumerate: Enumerate
}

_binary_table: dict[Callable, type[BinaryOp]] = {
    add: Add,
    sub: Sub,
    mul: Mul,
    div: Div,
    copysign: Copysign,
    fdim: Fdim,
    fmax: Fmax,
    min: Fmin,
    max: Fmax,
    fmin: Fmin,
    fmod: Fmod,
    remainder: Remainder,
    hypot: Hypot,
    atan2: Atan2,
    pow: Pow,
    size: Size
}

_ternary_table: dict[Callable, type[TernaryOp]] = {
    fma: Fma
}

_nary_table: dict[Callable, type[NaryOp]] = {
    zip: Zip,
}


class _ResolvePrimsInstance(DefaultTransformVisitor):
    """Single-use instance to resolve primitives"""
    func: FuncDef
    env: ForeignEnv

    def __init__(self, func: FuncDef, env: ForeignEnv):
        self.func = func
        self.env = env

    def apply(self) -> FuncDef:
        return self._visit_function(self.func, None)

    def _lookup(self, func: NamedId | ForeignAttribute):
        match func:
            case NamedId():
                return self.env[func.base]
            case ForeignAttribute():
                value = self._lookup(func.name)
                for attr in func.attrs:
                    if not hasattr(value, attr.base):
                        return None
                    value = getattr(value, attr.base)
                return value
            case _:
                raise RuntimeError('unreachable', func)

    def _visit_call(self, e: Call, ctx: None):
        func = self._lookup(e.func)
        if func is None:
            return super()._visit_call(e, ctx)
        elif func == rational:
            # special case: rational
            if len(e.args) != 2:
                raise ValueError(f'`rational`: expectes 2 arguments, got {e.format()}')
            if not isinstance(e.args[0], Integer):
                raise ValueError(f'`rational` expects an integer as first argument, got {e.args[0].format()}')
            if not isinstance(e.args[1], Integer):
                raise ValueError(f'`rational` expects an integer as second argument, got {e.args[1].format()}')
            return Rational(e.args[0].val, e.args[1].val, e.loc)
        elif func == hexfloat:
            # special case: hexfloat
            if len(e.args) != 1:
                raise ValueError(f'`hexfloat`: expects 1 argument, got {e.format()}')
            if not isinstance(e.args[0], ForeignVal) or not isinstance(e.args[0].val, str):
                raise ValueError(f'`hexfloat` expects a string as argument, got {e.args[0].format()}')
            return Hexnum(e.args[0].val, e.loc)
        elif func == digits:
            # special case: digits
            if len(e.args) != 3:
                raise ValueError(f'`digits`: expects 3 arguments, got {e.format()}')
            if not isinstance(e.args[0], Integer):
                raise ValueError(f'`digits` expects an integer as first argument, got {e.args[0].format()}')
            if not isinstance(e.args[1], Integer):
                raise ValueError(f'`digits` expects an integer as second argument, got {e.args[1].format()}')
            if not isinstance(e.args[2], Integer):
                raise ValueError(f'`digits` expects an integer as third argument, got {e.args[2].format()}')
            return Digits(e.args[0].val, e.args[1].val, e.args[2].val, e.loc)
        else:
            match len(e.args):
                case 1 if func in _unary_table:
                    cls1 = _unary_table[func]
                    arg = self._visit_expr(e.args[0], None)
                    return cls1(arg, e.loc)
                case 2 if func in _binary_table:
                    cls2 = _binary_table[func]
                    arg1 = self._visit_expr(e.args[0], None)
                    arg2 = self._visit_expr(e.args[1], None)
                    return cls2(arg1, arg2, e.loc)
                case 3 if func in _ternary_table:
                    cls3 = _ternary_table[func]
                    arg1 = self._visit_expr(e.args[0], None)
                    arg2 = self._visit_expr(e.args[1], None)
                    arg3 = self._visit_expr(e.args[2], None)
                    return cls3(arg1, arg2, arg3, e.loc)
                case _ if func in _nary_table:
                    cls = _nary_table[func]
                    args = [self._visit_expr(arg, None) for arg in e.args]
                    return cls(args, e.loc)
                case _:
                    return super()._visit_call(e, ctx)

class ResolvePrims:
    """
    Primitive resolver.

    Replaces calls to FPy's reserved primitives with their corresponding AST nodes.
    FPy treats certain operations specially suc as the usual arithmetic,
    and transcendental operations found in math libraries.
    """

    @staticmethod
    def apply(func: FuncDef, env: ForeignEnv) -> FuncDef:
        if not isinstance(func, FuncDef):
            raise TypeError(f'Expected `FuncDef`, got {type(func)} for {func}')
        return _ResolvePrimsInstance(func, env).apply()
