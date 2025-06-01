"""
Mathematical functions under rounding contexts.
"""

from typing import Any, Callable, TypeAlias

from .number import Context, Float
from .number.gmp import *
from .number.real import (
    RealContext,
    real_add, real_sub, real_mul, real_neg, real_abs,
    real_ceil, real_floor, real_trunc, real_round
)
from .number.round import RoundingMode

_real_ops: dict[Any, Callable[..., Float]] = {
    mpfr_fabs: real_abs,
    mpfr_neg: real_neg,
    mpfr_add: real_add,
    mpfr_sub: real_sub,
    mpfr_mul: real_mul,
}

def _apply_mpfr(fn: Callable[..., Float], *args: Float, ctx: Context) -> Float:
    """
    Applies a MPFR function with the given arguments and context.
    The function is expected to take a variable number of `Float` arguments
    followed by an integer for precision.
    """
    p, n = ctx.round_params()
    match p, n:
        case int(), _:
            # floating-point style rounding
            r = fn(*args, prec=p)  # compute with round-to-odd (safe at p digits)
            return ctx.round(r)  # re-round under desired rounding mode
        case _, int():
            # fixed-point style rounding
            r = fn(*args, n=n)
            return ctx.round(r)  # re-round under desired rounding mode
        case _:
            # real computation; no rounding
            real_fn = _real_ops.get(fn)
            if real_fn is None:
                raise NotImplementedError(f'p={p}, n={n}, func={fn}')
            return real_fn(*args)

################################################################################
# General operations

def acos(x: Float, ctx: Context):
    """Computes the inverse cosine of `x` rounded under `ctx`."""
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_acos, x, ctx=ctx)

def acosh(x: Float, ctx: Context):
    """Computes the inverse hyperbolic cosine of `x` rounded under `ctx`."""
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_acosh, x, ctx=ctx)

def add(x: Float, y: Float, ctx: Context):
    """Adds `x` and `y` rounded under `ctx`."""
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(y, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(y)}\' for x={y}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_add, x, y, ctx=ctx)

def asin(x: Float, ctx: Context):
    """Computes the inverse sine of `x` rounded under `ctx`."""
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_asin, x, ctx=ctx)

def asinh(x: Float, ctx: Context):
    """Computes the inverse hyperbolic sine of `x` rounded under `ctx`."""
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_asinh, x, ctx=ctx)

def atan(x: Float, ctx: Context):
    """Computes the inverse tangent of `x` rounded under `ctx`."""
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_atan, x, ctx=ctx)

def atan2(y: Float, x: Float, ctx: Context):
    """
    Computes `atan(y / x)` taking into account the correct quadrant
    that the point `(x, y)` resides in. The result is rounded under `ctx`.
    """
    if not isinstance(y, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(y)}\' for x={y}')
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_atan2, y, x, ctx=ctx)

def atanh(x: Float, ctx: Context):
    """Computes the inverse hyperbolic tangent of `x` under `ctx`."""
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_atanh, x, ctx=ctx)

def cbrt(x: Float, ctx: Context):
    """Computes the cube root of `x` rounded under `ctx`."""
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_cbrt, x, ctx=ctx)

def copysign(x: Float, y: Float, ctx: Context):
    """Returns `|x| * sign(y)` rounded under `ctx`."""
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(y, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(y)}\' for x={y}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_copysign, x, y, ctx=ctx)

def cos(x: Float, ctx: Context):
    """Computes the cosine of `x` rounded under `ctx`."""
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_cos, x, ctx=ctx)

def cosh(x: Float, ctx: Context):
    """Computes the hyperbolic cosine `x` rounded under `ctx`."""
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_cosh, x, ctx=ctx)

def div(x: Float, y: Float, ctx: Context):
    """Computes `x / y` rounded under `ctx`."""
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(y, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(y)}\' for x={y}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_div, x, y, ctx=ctx)

def erf(x: Float, ctx: Context):
    """Computes the error function of `x` rounded under `ctx`."""
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_erf, x, ctx=ctx)

def erfc(x: Float, ctx: Context):
    """Computes `1 - erf(x)` rounded under `ctx`."""
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_erfc, x, ctx=ctx)

def exp(x: Float, ctx: Context):
    """Computes `e ** x` rounded under `ctx`."""
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_exp, x, ctx=ctx)

def exp2(x: Float, ctx: Context):
    """Computes `2 ** x` rounded under `ctx`."""
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_exp2, x, ctx=ctx)

def exp10(x: Float, ctx: Context):
    """Computes `10 *** x` rounded under `ctx`."""
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_exp10, x, ctx=ctx)

def expm1(x: Float, ctx: Context):
    """Computes `exp(x) - 1` rounded under `ctx`."""
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_expm1, x, ctx=ctx)

def fabs(x: Float, ctx: Context):
    """Computes `|x|` rounded under `ctx`."""
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_fabs, x, ctx=ctx)

def fdim(x: Float, y: Float, ctx: Context):
    """Computes `max(x - y, 0)` rounded under `ctx`."""
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(y, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(y)}\' for x={y}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_fdim, x, y, ctx=ctx)

def fma(x: Float, y: Float, z: Float, ctx: Context):
    """Computes `x * y + z` rounded under `ctx`."""
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(y, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(y)}\' for x={y}')
    if not isinstance(z, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(z)}\' for x={z}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_fma, x, y, z, ctx=ctx)

def fmax(x: Float, y: Float, ctx: Context):
    """Computes `max(x, y)` rounded under `ctx`."""
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(y, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(y)}\' for x={y}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_fmax, x, y, ctx=ctx)

def fmin(x: Float, y: Float, ctx: Context):
    """Computes `min(x, y)` rounded under `ctx`."""
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(y, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(y)}\' for x={y}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_fmin, x, y, ctx=ctx)

def fmod(x: Float, y: Float, ctx: Context):
    """
    Computes the remainder of `x / y` rounded under this context.

    The remainder has the same sign as `x`; it is exactly `x - iquot * y`,
    where `iquot` is the `x / y` with its fractional part truncated.
    """
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(y, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(y)}\' for x={y}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_fmod, x, y, ctx=ctx)

def hypot(x: Float, y: Float, ctx: Context):
    """Computes `sqrt(x * x + y * y)` rounded under `ctx`."""
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(y, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(y)}\' for x={y}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_hypot, x, y, ctx=ctx)

def lgamma(x: Float, ctx: Context):
    """Computes the log-gamma of `x` rounded under `ctx`."""
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_lgamma, x, ctx=ctx)

def log(x: Float, ctx: Context):
    """Computes `log(x)` rounded under `ctx`."""
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_log, x, ctx=ctx)

def log10(x: Float, ctx: Context):
    """Computes `log10(x)` rounded under `ctx`."""
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_log10, x, ctx=ctx)

def log1p(x: Float, ctx: Context):
    """Computes `log1p(x)` rounded under `ctx`."""
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_log1p, x, ctx=ctx)

def log2(x: Float, ctx: Context):
    """Computes `log2(x)` rounded under `ctx`."""
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_log2, x, ctx=ctx)

def mul(x: Float, y: Float, ctx: Context):
    """Multiplies `x` and `y` rounded under `ctx`."""
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(y, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(y)}\' for x={y}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_mul, x, y, ctx=ctx)

def neg(x: Float, ctx: Context):
    """Computes `-x` rounded under `ctx`."""
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for ctx={ctx}')
    return _apply_mpfr(mpfr_neg, x, ctx=ctx)

def pow(x: Float, y: Float, ctx: Context):
    """Computes `x**y` rounded under `ctx`."""
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(y, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(y)}\' for x={y}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_pow, x, y, ctx=ctx)

def remainder(x: Float, y: Float, ctx: Context):
    """
    Computes the remainder of `x / y` rounded under `ctx`.

    The remainder is exactly `x - quo * y`, where `quo` is the
    integral value nearest the exact value of `x / y`.
    """
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(y, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(y)}\' for x={y}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_remainder, x, y, ctx=ctx)

def sin(x: Float, ctx: Context):
    """Computes the sine of `x` rounded under `ctx`."""
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_sin, x, ctx=ctx)

def sinh(x: Float, ctx: Context):
    """Computes the hyperbolic sine of `x` rounded under `ctx`."""
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_sinh, x, ctx=ctx)

def sqrt(x: Float, ctx: Context):
    """Computes square-root of `x` rounded under `ctx`."""
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_sqrt, x, ctx=ctx)

def sub(x: Float, y: Float, ctx: Context):
    """Subtracts `y` from `x` rounded under `ctx`."""
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(y, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(y)}\' for x={y}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_sub, x, y, ctx=ctx)

def tan(x: Float, ctx: Context):
    """Computes the tangent of `x` rounded under `ctx`."""
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_tan, x, ctx=ctx)

def tanh(x: Float, ctx: Context):
    """Computes the hyperbolic tangent of `x` rounded under `ctx`."""
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_tanh, x, ctx=ctx)

def tgamma(x: Float, ctx: Context):
    """Computes gamma of `x` rounded under `ctx`."""
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_tgamma, x, ctx=ctx)

#############################################################################
# Round-to-integer operations

def ceil(x: Float, ctx: Context):
    """
    Computes the smallest integer greater than or equal to `x`
    that is representable under `ctx`.

    If the context supports overflow, the result may be infinite.
    """
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
    match ctx:
        case RealContext():
            # use rounding primitives
            return real_ceil(x)
        case _:
            return ctx.with_rm(RoundingMode.RTP).round_integer(x)

def floor(x: Float, ctx: Context):
    """
    Computes the largest integer less than or equal to `x`
    that is representable under `ctx`.

    If the context supports overflow, the result may be infinite.
    """
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
    match ctx:
        case RealContext():
            # use rounding primitives
            return real_floor(x)
        case _:
            return ctx.with_rm(RoundingMode.RTN).round_integer(x)

def trunc(x: Float, ctx: Context):
    """
    Computes the integer with the largest magnitude whose
    magnitude is less than or equal to the magnitude of `x`
    that is representable under `ctx`.

    If the context supports overflow, the result may be infinite.
    """
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
    match ctx:
        case RealContext():
            # use rounding primitives
            return real_trunc(x)
        case _:
            return ctx.with_rm(RoundingMode.RTZ).round_integer(x)

def nearbyint(x: Float, ctx: Context):
    """
    Rounds `x` to a representable integer according to
    the rounding mode of this context.

    If the context supports overflow, the result may be infinite.
    """
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
    match ctx:
        case RealContext():
            raise RuntimeError('nearbyint() not supported in RealContext')
        case _:
            return ctx.round_integer(x)

def round(x: Float, ctx: Context):
    """
    Rounds `x` to the nearest representable integer,
    rounding ties away from zero in halfway cases.

    If the context supports overflow, the result may be infinite.
    """
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
    match ctx:
        case RealContext():
            # use rounding primitives
            return real_round(x)
        case _:
            return ctx.with_rm(RoundingMode.RNA).round_integer(x)
