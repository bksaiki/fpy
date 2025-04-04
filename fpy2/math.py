"""
Mathematical functions under rounding contexts.
"""

from typing import Callable, TypeAlias

from .number import Context, Float
from .number.gmp import *

MPFR_1ary: TypeAlias = Callable[[Float, int], Float]
MPFR_2ary: TypeAlias = Callable[[Float, Float, int], Float]
MPFR_3ary: TypeAlias = Callable[[Float, Float, Float, int], Float]


def _apply_1ary(func: MPFR_1ary, x: Float, ctx: Context):
    p, n = ctx.round_params()
    if p is None:
        raise NotImplementedError(f'p={p}, n={n}')
    else:
        r = func(x, p)       # compute with round-to-odd (safe at p digits)
        return ctx.round(r)  # re-round under desired rounding mode

def _apply_2ary(func: MPFR_2ary, x: Float, y: Float, ctx: Context):
    p, n = ctx.round_params()
    if p is None:
        raise NotImplementedError(f'p={p}, n={n}')
    else:
        r = func(x, y, p)    # compute with round-to-odd (safe at p digits)
        return ctx.round(r)  # re-round under desired rounding mode

def _apply_3ary(func: MPFR_3ary, x: Float, y: Float, z: Float, ctx: Context):
    p, n = ctx.round_params()
    if p is None:
        raise NotImplementedError(f'p={p}, n={n}')
    else:
        r = func(x, y, z, p) # compute with round-to-odd (safe at p digits)
        return ctx.round(r)  # re-round under desired rounding mode

################################################################################
# General operations

def acos(x: Float, ctx: Context):
    """Computes `acos(x)` rounded under `ctx`."""
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_1ary(mpfr_acos, x, ctx)

def acosh(x: Float, ctx: Context):
    """Computes `acosh(x)` rounded under `ctx`."""
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_1ary(mpfr_acosh, x, ctx)

def add(x: Float, y: Float, ctx: Context):
    """Adds `x` and `y` rounded under `ctx`."""
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(y, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(y)}\' for x={y}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_2ary(mpfr_add, x, y, ctx)

def asin(x: Float, ctx: Context):
    """Computes `asin(x)` rounded under `ctx`."""
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_1ary(mpfr_asin, x, ctx)

def asinh(x: Float, ctx: Context):
    """Computes `asinh(x)` rounded under `ctx`."""
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_1ary(mpfr_asinh, x, ctx)

def atan(x: Float, ctx: Context):
    """Computes `atan(x)` rounded under `ctx`."""
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_1ary(mpfr_atan, x, ctx)

def atan2(y: Float, x: Float, ctx: Context):
    """Computes `atan2(y, x)` rounded under `ctx`."""
    if not isinstance(y, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(y)}\' for x={y}')
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_2ary(mpfr_atan2, y, x, ctx)

def atanh(x: Float, ctx: Context):
    """Computes `atanh(x)` rounded under `ctx`."""
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_1ary(mpfr_atanh, x, ctx)

def cbrt(x: Float, ctx: Context):
    """Computes `cbrt(x)` rounded under `ctx`."""
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_1ary(mpfr_cbrt, x, ctx)

def copysign(x: Float, y: Float, ctx: Context):
    """Computes `copysign(x, y)` rounded under `ctx`."""
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(y, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(y)}\' for x={y}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_2ary(mpfr_copysign, x, y, ctx)

def cos(x: Float, ctx: Context):
    """Computes `cos(x)` rounded under `ctx`."""
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_1ary(mpfr_cos, x, ctx)

def cosh(x: Float, ctx: Context):
    """Computes `cosh(x)` rounded under `ctx`."""
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_1ary(mpfr_cosh, x, ctx)

def div(x: Float, y: Float, ctx: Context):
    """Computes `x / y` rounded under `ctx`."""
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(y, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(y)}\' for x={y}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_2ary(mpfr_div, x, y, ctx)

def erf(x: Float, ctx: Context):
    """Computes `erf(x)` rounded under `ctx`."""
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_1ary(mpfr_erf, x, ctx)

def erfc(x: Float, ctx: Context):
    """Computes `erfc(x)` rounded under `ctx`."""
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_1ary(mpfr_erfc, x, ctx)

def exp(x: Float, ctx: Context):
    """Computes `exp(x)` rounded under `ctx`."""
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_1ary(mpfr_exp, x, ctx)

def exp2(x: Float, ctx: Context):
    """Computes `exp2(x)` rounded under `ctx`."""
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_1ary(mpfr_exp2, x, ctx)

def exp10(x: Float, ctx: Context):
    """Computes `exp10(x)` rounded under `ctx`."""
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_1ary(mpfr_exp10, x, ctx)

def expm1(x: Float, ctx: Context):
    """Computes `expm1(x)` rounded under `ctx`."""
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_1ary(mpfr_expm1, x, ctx)

def fabs(x: Float, ctx: Context):
    """Computes `fabs(x)` rounded under `ctx`."""
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_1ary(mpfr_fabs, x, ctx)

def fdim(x: Float, y: Float, ctx: Context):
    """Computes `fdim(x, y)` rounded under `ctx`."""
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(y, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(y)}\' for x={y}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_2ary(mpfr_fdim, x, y, ctx)

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
    return _apply_3ary(mpfr_fma, x, y, z, ctx)

def fmod(x: Float, y: Float, ctx: Context):
    """Computes `fmod(x, y)` rounded under `ctx`."""
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(y, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(y)}\' for x={y}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_2ary(mpfr_fmod, x, y, ctx)

def fmax(x: Float, y: Float, ctx: Context):
    """Computes `fmax(x, y)` rounded under `ctx`."""
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(y, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(y)}\' for x={y}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_2ary(mpfr_fmax, x, y, ctx)

def fmin(x: Float, y: Float, ctx: Context):
    """Computes `fmin(x, y)` rounded under `ctx`."""
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(y, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(y)}\' for x={y}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_2ary(mpfr_fmin, x, y, ctx)

def hypot(x: Float, y: Float, ctx: Context):
    """Computes `sqrt(x * x + y * y)` rounded under `ctx`."""
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(y, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(y)}\' for x={y}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_2ary(mpfr_hypot, x, y, ctx)

def lgamma(x: Float, ctx: Context):
    """Computes `lgamma(x)` rounded under `ctx`."""
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_1ary(mpfr_lgamma, x, ctx)

def log(x: Float, ctx: Context):
    """Computes `log(x)` rounded under `ctx`."""
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_1ary(mpfr_log, x, ctx)

def log10(x: Float, ctx: Context):
    """Computes `log10(x)` rounded under `ctx`."""
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_1ary(mpfr_log10, x, ctx)

def log1p(x: Float, ctx: Context):
    """Computes `log1p(x)` rounded under `ctx`."""
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_1ary(mpfr_log1p, x, ctx)

def log2(x: Float, ctx: Context):
    """Computes `log2(x)` rounded under `ctx`."""
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_1ary(mpfr_log2, x, ctx)

def mul(x: Float, y: Float, ctx: Context):
    """Multiplies `x` and `y` rounded under `ctx`."""
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(y, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(y)}\' for x={y}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_2ary(mpfr_mul, x, y, ctx)

def neg(x: Float, ctx: Context):
    """Computes `-x` rounded under `ctx`."""
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for ctx={ctx}')
    return _apply_1ary(mpfr_neg, x, ctx)

def pow(x: Float, y: Float, ctx: Context):
    """Computes `x**y` rounded under `ctx`."""
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(y, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(y)}\' for x={y}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_2ary(mpfr_pow, x, y, ctx)

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
    return _apply_2ary(mpfr_remainder, x, y, ctx)

def sin(x: Float, ctx: Context):
    """Computes `sin(x)` rounded under `ctx`."""
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_1ary(mpfr_sin, x, ctx)

def sinh(x: Float, ctx: Context):
    """Computes `sinh(x)` rounded under `ctx`."""
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_1ary(mpfr_sinh, x, ctx)

def sqrt(x: Float, ctx: Context):
    """Computes `sqrt(x)` rounded under `ctx`."""
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_1ary(mpfr_sqrt, x, ctx)

def sub(x: Float, y: Float, ctx: Context):
    """Subtracts `y` from `x` rounded under `ctx`."""
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(y, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(y)}\' for x={y}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_2ary(mpfr_sub, x, y, ctx)

def tan(x: Float, ctx: Context):
    """Computes `tan(x)` rounded under `ctx`."""
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_1ary(mpfr_tan, x, ctx)

def tanh(x: Float, ctx: Context):
    """Computes `tanh(x)` rounded under `ctx`."""
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_1ary(mpfr_tanh, x, ctx)

def tgamma(x: Float, ctx: Context):
    """Computes `tgamma(x)` rounded under `ctx`."""
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_1ary(mpfr_tgamma, x, ctx)

#############################################################################
# Round-to-integer operations

# def ceil(x: Float, ctx: Context):
#     """
#     Computes the nearest integer greater than or equal to `x`
#     representable under `ctx`.
#     """
#     if not isinstance(x, Float):
#         raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
#     if not isinstance(ctx, Context):
#         raise TypeError(f'Expected \'Context\', got \'{type(ctx)}\' for x={ctx}')
#     return ctx.round_integer(x)

# def floor(x: Float, ctx: Context):



# # ceil
# # floor
# # trunc
# # round
# # nearbyint
