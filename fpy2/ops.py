"""
Mathematical functions under rounding contexts.
"""

from fractions import Fraction
from typing import Any, Callable

from .number import Context, Float, Real, RoundingMode
from .number.gmp import *
from .number.real import (
    RealContext,
    real_add, real_sub, real_mul, real_neg, real_abs,
    real_ceil, real_floor, real_trunc, real_round,
    real_fma
)

from .utils import digits_to_fraction, hexnum_to_fraction, is_dyadic

__all__ = [
    # General operations
    'acos',
    'acosh',
    'add',
    'asin',
    'asinh',
    'atan',
    'atan2',
    'atanh',
    'cbrt',
    'copysign',
    'cos',
    'cosh',
    'div',
    'erf',
    'erfc',
    'exp',
    'exp2',
    'exp10',
    'expm1',
    'fabs',
    'fdim',
    'fma',
    'fmax',
    'fmin',
    'fmod',
    'hypot',
    'lgamma',
    'log',
    'log10',
    'log1p',
    'log2',
    'mul',
    'neg',
    'pow',
    'remainder',
    'sin',
    'sinh',
    'sqrt',
    'sub',
    'tan',
    'tanh',
    'tgamma',
    # Rounding operations
    'cast',
    # Round-to-integer operations
    'ceil',
    'floor',
    'trunc',
    'nearbyint',
    'round',
    # Classification
    'isnan',
    'isinf',
    'isfinite',
    'isnormal',
    'signbit',
    # Tensor
    'dim',
    'size',
    # Constants
    'digits',
    'hexfloat',
    'rational',
    'nan',
    'inf',
    'const_pi',
    'const_e',
    'const_log2e',
    'const_log10e',
    'const_ln2',
    'const_pi_2',
    'const_pi_4',
    'const_1_pi',
    'const_2_pi',
    'const_2_sqrt_pi',
    'const_sqrt2',
    'const_sqrt1_2',
]


_real_ops: dict[Any, Callable[..., Float]] = {
    mpfr_fabs: real_abs,
    mpfr_neg: real_neg,
    mpfr_add: real_add,
    mpfr_sub: real_sub,
    mpfr_mul: real_mul,
    mpfr_fma: real_fma
}

def _apply_real(fn: Callable[..., Float], args: tuple[Float, ...]) -> Float:
    # real computation; no rounding
    real_fn = _real_ops.get(fn)
    if real_fn is None:
        raise NotImplementedError(f'cannot evaluate exactly: fn={fn}, args={args}')
    return real_fn(*args)

def _apply_mpfr(fn: Callable[..., Float], *args: Float, ctx: Optional[Context] = None) -> Float:
    """
    Applies a MPFR function with the given arguments and context.
    The function is expected to take a variable number of `Float` arguments
    followed by an integer for precision.
    """
    if ctx is None:
        # real computation; no rounding
        return _apply_real(fn, args)
    else:
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
                return _apply_real(fn, args)

def _real_constant(name: str, ctx: Optional[Context] = None) -> Float:
    if name == 'NAN':
        return Float(isnan=True, ctx=ctx)
    elif name == 'INFINITY':
        return Float(isinf=True, ctx=ctx)
    else:
        raise NotImplementedError(f'cannot evaluate exactly: name={name}')

def _apply_mpfr_constant(name: str, ctx: Optional[Context] = None) -> Float:
    """
    Computes an MPFR constant function with the given context.
    """
    if ctx is None:
        # real computation; no rounding
        return _real_constant(name)
    else:
        p, n = ctx.round_params()
        match p, n:
            case int(), _:
                # floating-point style rounding
                r = mpfr_constant(name, prec=p)  # compute with round-to-odd (safe at p digits)
                return ctx.round(r)  # re-round under desired rounding mode
            case _, int():
                # fixed-point style rounding
                r = mpfr_constant(name, n=n)
                return ctx.round(r)  # re-round under desired rounding mode
            case _:
                # real computation; no rounding
                return _real_constant(name)

################################################################################
# Types

def _real_to_float(x: Real) -> Float:
    match x:
        case Float():
            return x
        case int():
            return Float.from_int(x)
        case float():
            return Float.from_float(x)
        case _:
            raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')

################################################################################
# General operations

def acos(x: Real, ctx: Optional[Context] = None):
    """Computes the inverse cosine of `x` rounded under `ctx`."""
    x = _real_to_float(x)
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_acos, x, ctx=ctx)

def acosh(x: Real, ctx: Optional[Context] = None):
    """Computes the inverse hyperbolic cosine of `x` rounded under `ctx`."""
    x = _real_to_float(x)
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_acosh, x, ctx=ctx)

def add(x: Real, y: Real, ctx: Optional[Context] = None):
    """Adds `x` and `y` rounded under `ctx`."""
    x = _real_to_float(x)
    y = _real_to_float(y)
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_add, x, y, ctx=ctx)

def asin(x: Real, ctx: Optional[Context] = None):
    """Computes the inverse sine of `x` rounded under `ctx`."""
    x = _real_to_float(x)
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_asin, x, ctx=ctx)

def asinh(x: Real, ctx: Optional[Context] = None):
    """Computes the inverse hyperbolic sine of `x` rounded under `ctx`."""
    x = _real_to_float(x)
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_asinh, x, ctx=ctx)

def atan(x: Real, ctx: Optional[Context] = None):
    """Computes the inverse tangent of `x` rounded under `ctx`."""
    x = _real_to_float(x)
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_atan, x, ctx=ctx)

def atan2(y: Real, x: Real, ctx: Optional[Context] = None):
    """
    Computes `atan(y / x)` taking into account the correct quadrant
    that the point `(x, y)` resides in. The result is rounded under `ctx`.
    """
    y = _real_to_float(y)
    x = _real_to_float(x)
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_atan2, y, x, ctx=ctx)

def atanh(x: Real, ctx: Optional[Context] = None):
    """Computes the inverse hyperbolic tangent of `x` under `ctx`."""
    x = _real_to_float(x)
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_atanh, x, ctx=ctx)

def cbrt(x: Real, ctx: Optional[Context] = None):
    """Computes the cube root of `x` rounded under `ctx`."""
    x = _real_to_float(x)
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_cbrt, x, ctx=ctx)

def copysign(x: Real, y: Real, ctx: Optional[Context] = None):
    """Returns `|x| * sign(y)` rounded under `ctx`."""
    x = _real_to_float(x)
    y = _real_to_float(y)
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_copysign, x, y, ctx=ctx)

def cos(x: Real, ctx: Optional[Context] = None):
    """Computes the cosine of `x` rounded under `ctx`."""
    x = _real_to_float(x)
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_cos, x, ctx=ctx)

def cosh(x: Real, ctx: Optional[Context] = None):
    """Computes the hyperbolic cosine `x` rounded under `ctx`."""
    x = _real_to_float(x)
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_cosh, x, ctx=ctx)

def div(x: Real, y: Real, ctx: Optional[Context] = None):
    """Computes `x / y` rounded under `ctx`."""
    x = _real_to_float(x)
    y = _real_to_float(y)
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_div, x, y, ctx=ctx)

def erf(x: Real, ctx: Optional[Context] = None):
    """Computes the error function of `x` rounded under `ctx`."""
    x = _real_to_float(x)
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_erf, x, ctx=ctx)

def erfc(x: Real, ctx: Optional[Context] = None):
    """Computes `1 - erf(x)` rounded under `ctx`."""
    x = _real_to_float(x)
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_erfc, x, ctx=ctx)

def exp(x: Real, ctx: Optional[Context] = None):
    """Computes `e ** x` rounded under `ctx`."""
    x = _real_to_float(x)
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_exp, x, ctx=ctx)

def exp2(x: Real, ctx: Optional[Context] = None):
    """Computes `2 ** x` rounded under `ctx`."""
    x = _real_to_float(x)
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_exp2, x, ctx=ctx)

def exp10(x: Real, ctx: Optional[Context] = None):
    """Computes `10 *** x` rounded under `ctx`."""
    x = _real_to_float(x)
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_exp10, x, ctx=ctx)

def expm1(x: Real, ctx: Optional[Context] = None):
    """Computes `exp(x) - 1` rounded under `ctx`."""
    x = _real_to_float(x)
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_expm1, x, ctx=ctx)

def fabs(x: Real, ctx: Optional[Context] = None):
    """Computes `|x|` rounded under `ctx`."""
    x = _real_to_float(x)
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_fabs, x, ctx=ctx)

def fdim(x: Real, y: Real, ctx: Optional[Context] = None):
    """Computes `max(x - y, 0)` rounded under `ctx`."""
    x = _real_to_float(x)
    y = _real_to_float(y)
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_fdim, x, y, ctx=ctx)

def fma(x: Real, y: Real, z: Real, ctx: Optional[Context] = None):
    """Computes `x * y + z` rounded under `ctx`."""
    x = _real_to_float(x)
    y = _real_to_float(y)
    z = _real_to_float(z)
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_fma, x, y, z, ctx=ctx)

def fmax(x: Real, y: Real, ctx: Optional[Context] = None):
    """Computes `max(x, y)` rounded under `ctx`."""
    x = _real_to_float(x)
    y = _real_to_float(y)
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_fmax, x, y, ctx=ctx)

def fmin(x: Real, y: Real, ctx: Optional[Context] = None):
    """Computes `min(x, y)` rounded under `ctx`."""
    x = _real_to_float(x)
    y = _real_to_float(y)
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_fmin, x, y, ctx=ctx)

def fmod(x: Real, y: Real, ctx: Optional[Context] = None):
    """
    Computes the remainder of `x / y` rounded under this context.

    The remainder has the same sign as `x`; it is exactly `x - iquot * y`,
    where `iquot` is the `x / y` with its fractional part truncated.
    """
    x = _real_to_float(x)
    y = _real_to_float(y)
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_fmod, x, y, ctx=ctx)

def hypot(x: Real, y: Real, ctx: Optional[Context] = None):
    """Computes `sqrt(x * x + y * y)` rounded under `ctx`."""
    x = _real_to_float(x)
    y = _real_to_float(y)
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_hypot, x, y, ctx=ctx)

def lgamma(x: Real, ctx: Optional[Context] = None):
    """Computes the log-gamma of `x` rounded under `ctx`."""
    x = _real_to_float(x)
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_lgamma, x, ctx=ctx)

def log(x: Real, ctx: Optional[Context] = None):
    """Computes `log(x)` rounded under `ctx`."""
    x = _real_to_float(x)
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_log, x, ctx=ctx)

def log10(x: Real, ctx: Optional[Context] = None):
    """Computes `log10(x)` rounded under `ctx`."""
    x = _real_to_float(x)
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_log10, x, ctx=ctx)

def log1p(x: Real, ctx: Optional[Context] = None):
    """Computes `log1p(x)` rounded under `ctx`."""
    x = _real_to_float(x)
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_log1p, x, ctx=ctx)

def log2(x: Real, ctx: Optional[Context] = None):
    """Computes `log2(x)` rounded under `ctx`."""
    x = _real_to_float(x)
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_log2, x, ctx=ctx)

def mul(x: Real, y: Real, ctx: Optional[Context] = None):
    """Multiplies `x` and `y` rounded under `ctx`."""
    x = _real_to_float(x)
    y = _real_to_float(y)
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_mul, x, y, ctx=ctx)

def neg(x: Real, ctx: Optional[Context] = None):
    """Computes `-x` rounded under `ctx`."""
    x = _real_to_float(x)
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for ctx={ctx}')
    return _apply_mpfr(mpfr_neg, x, ctx=ctx)

def pow(x: Real, y: Real, ctx: Optional[Context] = None):
    """Computes `x**y` rounded under `ctx`."""
    x = _real_to_float(x)
    y = _real_to_float(y)
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_pow, x, y, ctx=ctx)

def remainder(x: Real, y: Real, ctx: Optional[Context] = None):
    """
    Computes the remainder of `x / y` rounded under `ctx`.

    The remainder is exactly `x - quo * y`, where `quo` is the
    integral value nearest the exact value of `x / y`.
    """
    x = _real_to_float(x)
    y = _real_to_float(y)
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_remainder, x, y, ctx=ctx)

def sin(x: Real, ctx: Optional[Context] = None):
    """Computes the sine of `x` rounded under `ctx`."""
    x = _real_to_float(x)
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_sin, x, ctx=ctx)

def sinh(x: Real, ctx: Optional[Context] = None):
    """Computes the hyperbolic sine of `x` rounded under `ctx`."""
    x = _real_to_float(x)
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_sinh, x, ctx=ctx)

def sqrt(x: Real, ctx: Optional[Context] = None):
    """Computes square-root of `x` rounded under `ctx`."""
    x = _real_to_float(x)
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_sqrt, x, ctx=ctx)

def sub(x: Real, y: Real, ctx: Optional[Context] = None):
    """Subtracts `y` from `x` rounded under `ctx`."""
    x = _real_to_float(x)
    y = _real_to_float(y)
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_sub, x, y, ctx=ctx)

def tan(x: Real, ctx: Optional[Context] = None):
    """Computes the tangent of `x` rounded under `ctx`."""
    x = _real_to_float(x)
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_tan, x, ctx=ctx)

def tanh(x: Real, ctx: Optional[Context] = None):
    """Computes the hyperbolic tangent of `x` rounded under `ctx`."""
    x = _real_to_float(x)
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_tanh, x, ctx=ctx)

def tgamma(x: Real, ctx: Optional[Context] = None):
    """Computes gamma of `x` rounded under `ctx`."""
    x = _real_to_float(x)
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_tgamma, x, ctx=ctx)

#############################################################################
# Rounding operations

def cast(x: Real, ctx: Optional[Context] = None) -> Float:
    """
    Rounds `x` under the given context `ctx`.

    If `ctx` is `None`, this operation is the identity operation.
    """
    x = _real_to_float(x)
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    match ctx:
        case None | RealContext():
            # real computation; no rounding
            return x
        case _:
            return ctx.round(x)

#############################################################################
# Round-to-integer operations

def ceil(x: Real, ctx: Optional[Context] = None):
    """
    Computes the smallest integer greater than or equal to `x`
    that is representable under `ctx`.

    If the context supports overflow, the result may be infinite.
    """
    x = _real_to_float(x)
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    match ctx:
        case None | RealContext():
            # use rounding primitives
            return real_ceil(x)
        case _:
            return ctx.with_rm(RoundingMode.RTP).round_integer(x)

def floor(x: Real, ctx: Optional[Context] = None):
    """
    Computes the largest integer less than or equal to `x`
    that is representable under `ctx`.

    If the context supports overflow, the result may be infinite.
    """
    x = _real_to_float(x)
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    match ctx:
        case None | RealContext():
            # use rounding primitives
            return real_floor(x)
        case _:
            return ctx.with_rm(RoundingMode.RTN).round_integer(x)

def trunc(x: Real, ctx: Optional[Context] = None):
    """
    Computes the integer with the largest magnitude whose
    magnitude is less than or equal to the magnitude of `x`
    that is representable under `ctx`.

    If the context supports overflow, the result may be infinite.
    """
    x = _real_to_float(x)
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    match ctx:
        case None | RealContext():
            # use rounding primitives
            return real_trunc(x)
        case _:
            return ctx.with_rm(RoundingMode.RTZ).round_integer(x)

def nearbyint(x: Real, ctx: Optional[Context] = None):
    """
    Rounds `x` to a representable integer according to
    the rounding mode of this context.

    If the context supports overflow, the result may be infinite.
    """
    x = _real_to_float(x)
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    match ctx:
        case None | RealContext():
            raise RuntimeError('nearbyint() not supported in RealContext')
        case _:
            return ctx.round_integer(x)

def round(x: Real, ctx: Optional[Context] = None):
    """
    Rounds `x` to the nearest representable integer,
    rounding ties away from zero in halfway cases.

    If the context supports overflow, the result may be infinite.
    """
    x = _real_to_float(x)
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    match ctx:
        case None | RealContext():
            # use rounding primitives
            return real_round(x)
        case _:
            return ctx.with_rm(RoundingMode.RNA).round_integer(x)

#############################################################################
# Classification

def isnan(x: Real, ctx: Optional[Context] = None) -> bool:
    """Checks if `x` is NaN."""
    x = _real_to_float(x)
    return x.isnan

def isinf(x: Real, ctx: Optional[Context] = None) -> bool:
    """Checks if `x` is infinite."""
    x = _real_to_float(x)
    return x.isinf

def isfinite(x: Real, ctx: Optional[Context] = None) -> bool:
    """Checks if `x` is finite."""
    x = _real_to_float(x)
    return not x.is_nar()

def isnormal(x: Real, ctx: Optional[Context] = None) -> bool:
    """Checks if `x` is normal (not subnormal, zero, or NaN)."""
    x = _real_to_float(x)
    return x.is_normal()

def signbit(x: Real, ctx: Optional[Context] = None) -> bool:
    """Checks if the sign bit of `x` is set (i.e., `x` is negative)."""
    x = _real_to_float(x)
    # TODO: should all Floats have this property?
    return x.s

#############################################################################
# Tensor
# TODO: should these be implemented

def dim(x: list | tuple, ctx: Optional[Context] = None):
    """
    Returns the number of dimensions of the tensor `x`.

    Assumes that `x` is not a ragged tensor.
    """
    dim = 0
    while isinstance(x, (list, tuple)):
        dim += 1
        x = x[0]
    return Float.from_int(dim, ctx=ctx)

def size(x: list | tuple, dim: Real, ctx: Optional[Context] = None):
    """
    Returns the size of the dimension `dim` of the tensor `x`.

    Assumes that `x` is not a ragged tensor.
    """
    dim = _real_to_float(dim)
    for _ in range(int(dim)):
        x = x[0]
        if not isinstance(x, (list, tuple)):
            raise ValueError(f'dimension `{dim}` is out of bounds for the tensor `{x}`')
    return Float.from_int(len(x), ctx=ctx)

#############################################################################
# Constants

def digits(m: int, e: int, b: int, ctx: Optional[Context] = None) -> Float:
    """
    Creates a `Float` of the form `m * b**e`, where `m` is the
    significand, `e` is the exponent, and `b` is the base.

    The result is rounded under the given context.
    """
    match ctx:
        case None | RealContext():
            # real computation; no rounding
            x = digits_to_fraction(m, e, b)
            if not is_dyadic(x):
                raise ValueError(f'cannot evaluate exactly: digits(m={m}, e={e}, b={b})')
            m = x.numerator
            exp = 1 - x.denominator.bit_length()
            return Float(m=m, exp=exp)
        case Context():
            x = digits_to_fraction(m, e, b)
            return ctx.round(x)
        case _:
            raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for ctx={ctx}')

def hexfloat(s: str, ctx: Optional[Context] = None) -> Float:
    """
    Creates a `Float` from a hexadecimal floating-point string `s`.
    The result is rounded under the given context.
    """
    match ctx:
        case None | RealContext():
            # real computation; no rounding
            x = hexnum_to_fraction(s)
            if not is_dyadic(x):
                raise ValueError(f'cannot evaluate exactly: hexfloat(s={s})')
            m = x.numerator
            exp = 1 - x.denominator.bit_length()
            return Float(m=m, exp=exp)
        case Context():
            x = hexnum_to_fraction(s)
            return ctx.round(x)
        case _:
            raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for ctx={ctx}')

def rational(n: int, d: int, ctx: Optional[Context] = None) -> Float:
    """
    Creates a `Float` from a fraction with the given numerator and denominator.
    The result is rounded under the given context.
    """
    if d == 0:
        return Float(isnan=True, ctx=ctx)

    match ctx:
        case None | RealContext():
            # real computation; no rounding
            x = Fraction(n, d)
            if not is_dyadic(x):
                raise ValueError(f'cannot evaluate exactly: fraction(n={n}, d={d})')
            m = x.numerator
            exp = 1 - x.denominator.bit_length()
            return Float(m=m, exp=exp)
        case Context():
            x = Fraction(n, d)
            return ctx.round(x)
        case _:
            raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for ctx={ctx}')

def nan(ctx: Optional[Context] = None) -> Float:
    """
    Creates a `Float` representing NaN (Not a Number).
    The result is rounded under the given context.
    """
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return Float(isnan=True, ctx=ctx)

def inf(ctx: Optional[Context] = None) -> Float:
    """
    Creates a `Float` representing (positive) infinity.
    The result is rounded under the given context.
    """
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return Float(isinf=True, ctx=ctx)

def const_pi(ctx: Optional[Context] = None) -> Float:
    """
    Creates a `Float` representing π (pi).
    The result is rounded under the given context.
    """
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr_constant('PI', ctx=ctx)

def const_e(ctx: Optional[Context] = None) -> Float:
    """
    Creates a `Float` representing e (Euler's number).
    The result is rounded under the given context.
    """
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr_constant('E', ctx=ctx)

def const_log2e(ctx: Optional[Context] = None) -> Float:
    """
    Creates a `Float` representing log2(e) (the logarithm of e base 2).
    The result is rounded under the given context.
    """
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr_constant('LOG2E', ctx=ctx)

def const_log10e(ctx: Optional[Context] = None) -> Float:
    """
    Creates a `Float` representing log10(e) (the logarithm of e base 10).
    The result is rounded under the given context.
    """
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr_constant('LOG10E', ctx=ctx)

def const_ln2(ctx: Optional[Context] = None) -> Float:
    """
    Creates a `Float` representing ln(2) (the natural logarithm of 2).
    The result is rounded under the given context.
    """
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr_constant('LN2', ctx=ctx)

def const_pi_2(ctx: Optional[Context] = None) -> Float:
    """
    Creates a `Float` representing π/2 (pi divided by 2).
    The result is rounded under the given context.
    """
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr_constant('PI_2', ctx=ctx)

def const_pi_4(ctx: Optional[Context] = None) -> Float:
    """
    Creates a `Float` representing π/4 (pi divided by 4).
    The result is rounded under the given context.
    """
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr_constant('PI_4', ctx=ctx)

def const_1_pi(ctx: Optional[Context] = None) -> Float:
    """
    Creates a `Float` representing 1/π (one divided by pi).
    The result is rounded under the given context.
    """
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr_constant('M_1_PI', ctx=ctx)

def const_2_pi(ctx: Optional[Context] = None) -> Float:
    """
    Creates a `Float` representing 2/π (two divided by pi).
    The result is rounded under the given context.
    """
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr_constant('M_2_PI', ctx=ctx)

def const_2_sqrt_pi(ctx: Optional[Context] = None) -> Float:
    """
    Creates a `Float` representing 2/sqrt(π) (two divided by the square root of pi).
    The result is rounded under the given context.
    """
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr_constant('M_2_SQRTPI', ctx=ctx)

def const_sqrt2(ctx: Optional[Context] = None) -> Float:
    """
    Creates a `Float` representing sqrt(2) (the square root of 2).
    The result is rounded under the given context.
    """
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr_constant('SQRT2', ctx=ctx)

def const_sqrt1_2(ctx: Optional[Context] = None) -> Float:
    """
    Creates a `Float` representing sqrt(1/2) (the square root of 1/2).
    The result is rounded under the given context.
    """
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr_constant('SQRT1_2', ctx=ctx)
