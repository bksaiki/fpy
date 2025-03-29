"""
Conversion between FPy types and native Python types
"""

from ..utils import bits_to_float
from .globals import set_current_float_converter
from .ieee754 import IEEEContext
from .number import Float
from .real import RealFloat
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


set_current_float_converter(default_float_convert)
