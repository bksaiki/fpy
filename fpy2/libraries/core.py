"""
Core numerical functions.
"""

from . import base as fp

###########################################################
# Splitting functions

@fp.fpy_primitive(ctx='R', ret_ctx=('R', 'R'))
def split(x: fp.Float, n: fp.Float, ctx: fp.Context) -> tuple[fp.Float, fp.Float]:
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
        hi = ctx.round(fp.Float.nan(), exact=True)
        lo = ctx.round(fp.Float.nan(), exact=True)
        return hi, lo
    elif x.isinf:
        hi = ctx.round(fp.Float(s=x.s, isinf=True), exact=True)
        lo = ctx.round(fp.Float(s=x.s, isinf=True), exact=True)
        return hi, lo
    else:
        above, below = x.as_real().split(int(n))
        hi = ctx.round(above, exact=True)
        lo = ctx.round(below, exact=True)
        return hi, lo

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

@fp.fpy_primitive(ctx='R', ret_ctx=('R', 'R'), spec=_modf_spec)
def modf(x: fp.Float, ctx: fp.Context) -> tuple[fp.Float, fp.Float]:
    """
    Decomposes `x` into its integral and fractional parts.
    The operation is performed exactly.

    Mirroring the behavior of C/C++ `modf`:
    - if `x` is `+/-0`, the result is `(+/-0, +/-0)`
    - if `x` is `+/-Inf`, the result is `(+/-0, +/-Inf)`
    - if `x` is NaN, the result is `(NaN, NaN)`
    """
    if x.isnan:
        i = ctx.round(x, exact=True)
        f = ctx.round(x, exact=True)
        return i, f
    elif x.isinf:
        i = ctx.round(fp.Float(s=x.s), exact=True)
        f = ctx.round(fp.Float(s=x.s, isinf=True), exact=True)
        return i, f
    elif x.is_zero():
        i = ctx.round(fp.Float(s=x.s), exact=True)
        f = ctx.round(fp.Float(s=x.s), exact=True)
        return i, f
    else:
        hi, lo = x.as_real().split(-1)
        i = ctx.round(hi, exact=True)
        f = ctx.round(lo, exact=True)
        return i, f

@fp.fpy_primitive(ctx='R', ret_ctx=('R', 'R'))
def frexp(x: fp.Float, ctx: fp.Context) -> tuple[fp.Float, fp.Float]:
    """
    Decomposes `x` into its mantissa and exponent.
    The computation is performed exactly.

    Mirroring the behavior of C/C++ `frexp`:
    - if `x` is NaN, the result is `(NaN, NaN)`.
    - if `x` is infinity, the result is `(x, NaN)`.
    - if `x` is zero, the result is `(x, 0)`.
    """
    if x.isnan:
        m = ctx.round(fp.Float.nan(), exact=True)
        e = ctx.round(fp.Float.nan(), exact=True)
        return m, e
    elif x.isinf:
        m = ctx.round(fp.Float(s=x.s, isinf=True), exact=True)
        e = ctx.round(fp.Float.nan(), exact=True)
        return m, e
    elif x.is_zero():
        m = ctx.round(fp.Float(s=x.s), exact=True)
        e = ctx.round(fp.Float.zero(), exact=True)
        return m, e
    else:
        x = x.normalize()
        m = ctx.round(fp.RealFloat(s=x.s, e=0, c=x.c), exact=True)
        e = ctx.round(x.e)
        return m, e

############################################################
# Predicates

@fp.fpy
def isinteger(x: fp.Real) -> bool:
    """Checks if `x` is an integer."""
    _, fpart = modf(x)
    return fp.isfinite(fpart) and fpart == 0

@fp.fpy
def isnar(x: fp.Real) -> bool:
    """Checks if `x` is either NaN or infinity."""
    return fp.isnan(x) or fp.isinf(x)

###########################################################
# Exponent extraction and scaling

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

@fp.fpy_primitive(ctx='R', ret_ctx='R', spec=_logb_spec)
def logb(x: fp.Float, ctx: fp.Context) -> fp.Float:
    """
    Returns the normalized exponent of `x`.

    Special cases:
    - If `x == 0`, the result is `-INFINITY`.
    - If `x` is NaN, the result is NaN.
    - If `x` is infinite, the result is `INFINITY`.
    """
    if x.isnan:
        return ctx.round(fp.Float.nan())
    elif x.isinf:
        return ctx.round(fp.Float.inf())
    elif x.is_zero():
        return ctx.round(fp.Float.inf(True))
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
        ret = x * fp.pow(2, n)

    return ret

@fp.fpy_primitive(ctx='R', ret_ctx='R', spec=_ldexp_spec)
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

    if x.isnan or x.isinf:
        return ctx.round(x)
    else:
        xr = x.as_real()
        scale = fp.RealFloat.power_of_2(int(n))
        return ctx.round(xr * scale)

@fp.fpy(ctx=fp.INTEGER)
def max_e(xs: list[fp.Real]) -> tuple[fp.Real, bool]:
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

############################################################
# Context operations

@fp.fpy_primitive(ctx='R', ret_ctx='R')
def max_p(ctx: fp.Context) -> fp.Float:
    """
    Returns the maximum precision of the context.
    This is a no-op for the `RealContext`.
    """
    p, _ = ctx.round_params()
    if p is None:
        raise ValueError(f"ctx={ctx} does not have a maximum precision")
    return ctx.round(p)

@fp.fpy_primitive(ctx='R', ret_ctx='R')
def min_n(ctx: fp.Context) -> fp.Float:
    """
    Returns the least absolute digit of the context.
    This is the position of the most significant digit that
    can never be represented.
    """
    _, n = ctx.round_params()
    if n is None:
        raise ValueError(f"ctx={ctx} does not have a least absolute digit")
    return ctx.round(n)
