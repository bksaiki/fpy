import math

from typing import TypeAlias

# Numbers
from .float import Float, set_default_float_converter, get_default_float_converter
from .real import RealFloat

# Contexts
from .context import Context, OrdinalContext, SizedContext, EncodableContext
from .ieee754 import IEEEContext
from .mp import MPContext
from .mpb import MPBContext
from .mps import MPSContext

# Rounding
from .round import RoundingMode, RoundingDirection


RM: TypeAlias = RoundingMode
"""alias for `RoundingMode`"""


_FP64 = IEEEContext(11, 64, RM.RNE)

def _float_convert(x: Float):
    if x.ctx == _FP64:
        r = x
    else:
        r = _FP64.round(x)
        if r.inexact:
            raise TypeError(f'Expected representable value in \'float\': x={x}')

    if r.isnan:
        # NaN
        if r.s:
            return -math.nan
        else:
            return math.nan
    elif r.isinf:
        # Inf
        if r.s:
            return -math.inf
        else:
            return math.inf
    elif r.is_zero():
        # +/- 0
        if r.s:
            return -0.0
        else:
            return 0.0
    else:
        # finite, non-zero
        s_str = '-' if r.s else '+'
        c_str = hex(r.c)
        raise NotImplementedError(s_str, c_str)


set_default_float_converter(_float_convert)
