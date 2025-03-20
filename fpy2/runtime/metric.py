"""
Error metrics when comparing floating-point values.
"""

from titanfp.arithmetic.ieee754 import Float, digital_to_bits

def digital_to_ordinal(x: Float):
    """Converts a Digital value to its ordinal representation."""
    s = x.negative
    mag = x.fabs()
    return (-1 if s else 1) * digital_to_bits(mag, ctx=x.ctx)

def ordinal_error(x: Float, y: Float) -> Float:
    """
    Compute the ordinal error between two floating-point numbers `x` and `y`.
    Ordinal error measures approximately how many floating-point values
    are between `x` and `y`.
    """
    assert x.ctx is y.ctx, 'must be under the same context'

    ctx = x.ctx
    if x.isnan:
        if y.isnan:
            return 0
        else:
            return 1 << ctx.nbits
    elif y.isnan:
        return 1 << ctx.nbits
    else:
        x_ord = digital_to_ordinal(x)
        y_ord = digital_to_ordinal(y)
        return abs(x_ord - y_ord)
