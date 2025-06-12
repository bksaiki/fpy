"""
Core numerical functions.
"""

from fpy2 import *
from fpy2.typing import *

@fpy
def _logb_spec(x: Real):
    """
    Returns the normalized exponent of `x`.

    Special cases:
    - If `x == 0`, the result is `-INFINITY`.
    - If `x` is NaN, the result is NaN.
    - If `x` is infinite, the result is `INFINITY`.

    Under the `RealContext`, this function is the specification of logb.
    """
    return floor(log2(abs(x)))

@fpy_prim(spec=_logb_spec)
def logb(x: Float, ctx: Context):
    """
    Returns the normalized exponent of `x`.

    Special cases:
    - If `x == 0`, the result is `-INFINITY`.
    - If `x` is NaN, the result is NaN.
    - If `x` is infinite, the result is `INFINITY`.
    """
    if x.isnan:
        return Float(isnan=True, ctx=ctx)
    elif x.isinf:
        return Float(isinf=True, ctx=ctx)
    elif x.is_zero():
        return Float(s=True, isinf=True, ctx=ctx)
    else:
        return Float.from_int(x.e, ctx=ctx)

@fpy
def max_e(xs: tuple[Real, ...]) -> tuple[Real, bool]:
    """
    Computes the largest (normalized) exponent of the
    subset of finite, non-zero elements of `xs`.

    Returns the largest exponent and whether any such element exists.
    If all elements are zero, infinite, or NaN, the exponent is `-INFINITY`.
    """
    largest_e = -INFINITY
    any_non_zero = False
    for x in xs:
        if isfinite(x) and x != 0:
            any_non_zero = True
            largest_e = max(largest_e, logb(x))

    return (largest_e, any_non_zero)
