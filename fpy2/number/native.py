"""
Conversion between FPy types and native Python types
"""

from ..utils import bits_to_float
from .gmp import float_to_mpfr
from .globals import set_current_float_converter, set_current_str_converter
from .ieee754 import IEEEContext
from .number import RealFloat, Float
from .round import RoundingMode

_FP64 = IEEEContext(11, 64, RoundingMode.RNE)

def default_float_convert(x: RealFloat | Float):
    if isinstance(x, Float) and x.ctx == _FP64:
        r = x
    else:
        r = _FP64.round(x)
        if r.inexact:
            raise ValueError(f'Expected representable value in \'float\': x={x}')

    return bits_to_float(_FP64.encode(r))

def default_str_convert(x: RealFloat | Float) -> str:
    cls_name = x.__class__.__name__
    if isinstance(x, Float):
        if x.isnan:
            s = '-' if x.s else '+'
            return f'{cls_name}(\'{s}nan\')'
        elif x.isinf:
            s = '-' if x.s else '+'
            return f'{cls_name}(\'{s}inf\')'

    if x.is_zero():
        s = '-' if x.s else '+'
        return f'{cls_name}(\'{s}0.0\')'
    else:
        return f'{cls_name}(\'{str(float_to_mpfr(x))}\')'

set_current_float_converter(default_float_convert)
set_current_str_converter(default_str_convert)
