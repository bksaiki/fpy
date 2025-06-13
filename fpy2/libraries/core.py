"""
Core numerical functions.
"""

from fpy2 import *
from fpy2.typing import *

@fpy_prim
def split(x: Float, n: Float):
    """
    Splits `x` into two parts:
    - all digits of `x` that are above the `n`th digit
    - all digits of `x` that are at or below the `n`th digit

    The operation is performed exactly.

    Special cases:
    - if `x` is NaN, the result is `(NaN, NaN)`
    - if `x` is infinite, the result is `(x, x)`
    - if `n` is not an integer, a `ValueError` is raised.
    """

    if not n.is_integer():
        raise ValueError("n must be an integer")

    if x.isnan:
        return NDArray((Float(isnan=True, ctx=x.ctx), Float(isnan=True, ctx=x.ctx)))
    elif x.isinf:
        return NDArray((Float(s=x.s, isinf=True, ctx=x.ctx), Float(s=x.s, isinf=True, ctx=x.ctx)))
    else:
        hi, lo = x.as_real().split(int(n))
        return NDArray((Float.from_real(hi, ctx=x.ctx), Float.from_real(lo, ctx=x.ctx)))

@fpy(ctx=RealContext())
def modf(x: Real) -> tuple[Real, Real]:
    """
    Decomposes `x` into its integral and fractional parts.
    The operation is performed exactly.

    Mirroring the behavior of C/C++ `modf`:
    - if `x` is `+/-0`, the result is `(+/-0, +/-0)`
    - if `x` is `+/-Inf`, the result is `(+/-0, +/-Inf)`
    - if `x` is NaN, the result is `(NaN, NaN)`
    """
    if isnan(x):
        ret = (NAN, NAN)
    elif isinf(x):
        ret = (copysign(0, x), x)
    elif x == 0:
        ret = (copysign(0, x), copysign(0, x))
    else:
        ret = split(x, -1)

    return ret

@fpy
def isinteger(x: Real) -> bool:
    """Checks if `x` is an integer."""
    _, fpart = modf(x)
    return isfinite(fpart) and fpart == 0

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
def _ldexp_spec(x: Real, n: Real) -> Real:
    """
    Computes `x * 2**n` with correct rounding.

    Special cases:
    - If `x` is NaN, the result is NaN.
    - If `x` is infinite, the result is infinite.

    If `n` is not an integer, a `ValueError` is raised.
    Under the `RealContext`, this function is the specification of ldexp.
    """
    assert isinteger(n)

    if isnan(x):
        ret = NAN
    elif isinf(x):
        ret = copysign(INFINITY, x)
    else:
        ret = x * pow(2, n)

    return ret

@fpy_prim(spec=_ldexp_spec)
def ldexp(x: Float, n: Float, ctx: Context) -> Float:
    """
    Computes `x * 2**n` with correct rounding.

    Special cases:
    - If `x` is NaN, the result is NaN.
    - If `x` is infinite, the result is infinite.

    If `n` is not an integer, a `ValueError` is raised.
    """
    if not n.is_integer():
        raise ValueError("n must be an integer")

    if x.isnan:
        return Float(isnan=True, ctx=ctx)
    elif x.isinf:
        return Float(s=x.s, isinf=True, ctx=ctx)
    else:
        xr = x.as_real()
        scale = RealFloat.power_of_2(int(n))
        return Float.from_real(xr * scale, ctx=ctx)

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

