"""
Core numerical functions.
"""

from . import base as fp

__all__ = [
    "split",
    "modf",
    "isinteger",
    "logb",
    "ldexp",
    "frexp",
    "max_e",
]

@fp.fpy_primitive
def split(x: fp.Float, n: fp.Float):
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
        return [fp.Float(isnan=True, ctx=x.ctx), fp.Float(isnan=True, ctx=x.ctx)]
    elif x.isinf:
        return [fp.Float(s=x.s, isinf=True, ctx=x.ctx), fp.Float(s=x.s, isinf=True, ctx=x.ctx)]
    else:
        hi, lo = x.as_real().split(int(n))
        return [fp.Float.from_real(hi, ctx=x.ctx), fp.Float.from_real(lo, ctx=x.ctx)]

@fp.fpy
def _modf_spec(x: fp.Real) -> tuple[fp.Real, fp.Real]:
    """
    Decomposes `x` into its integral and fractional parts.
    The operation is performed exactly.

    Mirroring the behavior of C/C++ `modf`:
    - if `x` is `+/-0`, the result is `(+/-0, +/-0)`
    - if `x` is `+/-Inf`, the result is `(+/-0, +/-Inf)`
    - if `x` is NaN, the result is `(NaN, NaN)`
    """
    if fp.isnan(x):
        ret: tuple[fp.Real, fp.Real] = (fp.nan(), fp.nan())
    elif fp.isinf(x):
        ret = (fp.copysign(0, x), x)
    elif x == 0:
        ret = (fp.copysign(0, x), fp.copysign(0, x))
    else:
        ret = split(x, -1)

    return ret

@fp.fpy_primitive(spec=_modf_spec)
def modf(x: fp.Float):
    """
    Decomposes `x` into its integral and fractional parts.
    The operation is performed exactly.

    Mirroring the behavior of C/C++ `modf`:
    - if `x` is `+/-0`, the result is `(+/-0, +/-0)`
    - if `x` is `+/-Inf`, the result is `(+/-0, +/-Inf)`
    - if `x` is NaN, the result is `(NaN, NaN)`
    """
    if x.isnan:
        return [fp.Float(x=x, ctx=x.ctx), fp.Float(x=x, ctx=x.ctx)]
    elif x.isinf:
        return [fp.Float(s=x.s, ctx=x.ctx), fp.Float(s=x.s, isinf=True, ctx=x.ctx)]
    elif x.is_zero():
        return [fp.Float(s=x.s, ctx=x.ctx), fp.Float(s=x.s, ctx=x.ctx)]
    else:
        hi, lo = x.as_real().split(-1)
        return [fp.Float.from_real(hi, ctx=x.ctx), fp.Float.from_real(lo, ctx=x.ctx)]

@fp.fpy
def isinteger(x: fp.Real) -> bool:
    """Checks if `x` is an integer."""
    _, fpart = modf(x)
    return fp.isfinite(fpart) and fpart == 0

@fp.fpy
def _logb_spec(x: fp.Real):
    """
    Returns the normalized exponent of `x`.

    Special cases:
    - If `x == 0`, the result is `-INFINITY`.
    - If `x` is NaN, the result is NaN.
    - If `x` is infinite, the result is `INFINITY`.

    Under the `RealContext`, this function is the specification of logb.
    """
    return fp.floor(fp.log2(abs(x)))

@fp.fpy_primitive(spec=_logb_spec)
def logb(x: fp.Float, ctx: fp.Context):
    """
    Returns the normalized exponent of `x`.

    Special cases:
    - If `x == 0`, the result is `-INFINITY`.
    - If `x` is NaN, the result is NaN.
    - If `x` is infinite, the result is `INFINITY`.
    """
    if x.isnan:
        return fp.Float(isnan=True, ctx=ctx)
    elif x.isinf:
        return fp.Float(isinf=True, ctx=ctx)
    elif x.is_zero():
        return fp.Float(s=True, isinf=True, ctx=ctx)
    else:
        return ctx.round(x.e)

@fp.fpy
def _ldexp_spec(x: fp.Real, n: fp.Real) -> fp.Real:
    """
    Computes `x * 2**n` with correct rounding.

    Special cases:
    - If `x` is NaN, the result is NaN.
    - If `x` is infinite, the result is infinite.

    If `n` is not an integer, a `ValueError` is raised.
    Under the `RealContext`, this function is the specification of ldexp.
    """
    assert isinteger(n)

    if fp.isnan(x):
        ret: fp.Real = fp.nan()
    elif fp.isinf(x):
        ret = fp.copysign(fp.inf(), x)
    else:
        ret = x * pow(2, n)

    return ret

@fp.fpy_primitive(spec=_ldexp_spec)
def ldexp(x: fp.Float, n: fp.Float, ctx: fp.Context) -> fp.Float:
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
        return fp.Float(isnan=True, ctx=ctx)
    elif x.isinf:
        return fp.Float(s=x.s, isinf=True, ctx=ctx)
    else:
        xr = x.as_real()
        scale = fp.RealFloat.power_of_2(int(n))
        return ctx.round(xr * scale)

@fp.fpy_primitive
def frexp(x: fp.Float) -> tuple[fp.Float, fp.Float]:
    """
    Decomposes `x` into its mantissa and exponent.

    Mirroring the behavior of C/C++ `frexp`:
    - if `x` is NaN, the result is `(NaN, NaN)`.
    - if `x` is infinity, the result is `(x, NaN)`.
    - if `x` is zero, the result is `(x, 0)`.
    """
    if x.isnan:
        return (fp.Float(isnan=True, ctx=x.ctx), fp.Float(isnan=True, ctx=x.ctx))
    elif x.isinf:
        return (fp.Float(s=x.s, isinf=True, ctx=x.ctx), fp.Float(isnan=True, ctx=x.ctx))
    elif x.is_zero():
        return (fp.Float(x=x, ctx=x.ctx), fp.Float(ctx=x.ctx))
    else:
        x = x.normalize()
        mant = fp.Float(s=x.s, e=0, c=x.c, ctx=x.ctx)
        e = fp.Float.from_int(x.e, ctx=x.ctx)
        return (mant, e)

@fp.fpy(ctx=fp.INTEGER)
def max_e(xs: tuple[fp.Real, ...]) -> tuple[fp.Real, bool]:
    """
    Computes the largest (normalized) exponent of the
    subset of finite, non-zero elements of `xs`.

    Returns the largest exponent and whether any such element exists.
    If all elements are zero, infinite, or NaN, the exponent is `0`.
    """
    largest_e = 0
    any_non_zero = False
    for x in xs:
        if fp.isfinite(x) and x != 0:
            if any_non_zero:
                largest_e = max(largest_e, logb(x))
            else:
                # First non-zero finite element found
                largest_e = logb(x)
                any_non_zero = True

    return (largest_e, any_non_zero)

