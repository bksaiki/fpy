"""
Error metrics when comparing floating-point values.
"""

from titanfp.titanic.digital import Digital
from titanfp.arithmetic.evalctx import EvalCtx
from titanfp.arithmetic.ieee754 import IEEECtx, digital_to_bits

def digital_to_ordinal(x: Digital, ctx: EvalCtx):
    """Converts a Digital value to its ordinal representation."""
    match ctx:
        case IEEECtx():
            s = x.negative
            mag = Digital(x=x, negative=False)
            return (-1 if s else 1) * digital_to_bits(mag, ctx)
        case _:
            raise NotImplementedError('unsupported context')

def ordinal_error(x: Digital, y: Digital, ctx: EvalCtx) -> Digital:
    """
    Compute the ordinal error between two floating-point numbers `x` and `y`.
    Ordinal error measures approximately how many floating-point values
    are between `x` and `y`.
    """
    if x.isnan:
        if y.isnan:
            return 0
        else:
            return 1 << ctx.nbits
    elif y.isnan:
        return 1 << ctx.nbits
    else:
        x_ord = digital_to_ordinal(x, ctx)
        y_ord = digital_to_ordinal(y, ctx)
        return abs(x_ord - y_ord)
