"""
Mathematical functions under rounding contexts.
"""

from collections.abc import Callable
from fractions import Fraction

from .number import Context, Float, Real, REAL, RoundingMode
from .number.gmp import *
from .number.real import (
    RealContext,
    real_add, real_sub, real_mul, real_neg, real_abs,
    real_ceil, real_floor, real_trunc, real_roundint,
    real_fma
)

from .utils import (
    UNINIT,
    digits_to_fraction, hexnum_to_fraction, is_dyadic
)

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
    'round',
    'round_exact',
    'round_at',
    # Round-to-integer operations
    'ceil',
    'floor',
    'trunc',
    'nearbyint',
    'roundint',
    # Classification
    'isnan',
    'isinf',
    'isfinite',
    'isnormal',
    'signbit',
    # Tensor
    'empty',
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

################################################################################
# Type

def _cvt_to_real(x: Real) -> Float | Fraction:
    match x:
        case Float() | Fraction():
            return x
        case int():
            return Float.from_int(x)
        case float():
            return Float.from_float(x)
        case _:
            raise TypeError(f'Expected \'Float\' or \'Fraction\', got \'{type(x)}\' for x={x}')

def _cvt_to_float(x: Real) -> Float:
    match x:
        case Float():
            return x
        case int():
            return Float.from_int(x)
        case float():
            return Float.from_float(x)
        case Fraction():
            if is_dyadic(x):
                return Float.from_rational(x)
            raise ValueError(f'Cannot convert non-dyadic rational to Float: {x}')
        case _:
            raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')

#####################################################################
# Appliers

def _apply_mpfr_constant(name: Constant, ctx: Context = REAL) -> Float:
    """
    Applies an MPFR constant with the given context.

    The constant is computed with sufficient precision to be
    safely re-rounded under the given context.
    """
    p, n = ctx.round_params()
    match p, n:
        case int(), _:
            # floating-point style rounding
            x = mpfr_constant(name, prec=p)  # compute with round-to-odd (safe at p digits)
            return ctx.round(x)  # re-round under desired rounding mode
        case _, int():
            # fixed-point style rounding
            x = mpfr_constant(name, n=n)
            return ctx.round(x)  # re-round under desired rounding mode
        case _:
            # real computation
            raise ValueError(f'Cannot compute exactly: name={name}, ctx={ctx}')

def _apply_mpfr(fn: Callable[..., Float], *args: Real, ctx: Context = REAL) -> Float:
    """
    Applies an MPFR function with the given arguments and context.

    The function is expected to take a fixed number of `Float` arguments
    followed by either a precision `p` or a least absolute digit `n`.
    """
    p, n = ctx.round_params()
    fl_args = tuple(_cvt_to_float(x) for x in args)
    match p, n:
        case int(), _:
            # floating-point style rounding
            x = fn(*fl_args, prec=p)  # compute with round-to-odd (safe at p digits)
            return ctx.round(x)  # re-round under desired rounding mode
        case _, int():
            # fixed-point style rounding
            x = fn(*fl_args, n=n)
            return ctx.round(x)  # re-round under desired rounding mode
        case _:
            # real computation
            raise ValueError(f'Cannot compute exactly: fn={fn}, args={args}, ctx={ctx}')

def _apply_mpfr_or_real(
    mpfr_fn: Callable[..., Float],
    real_fn: Callable[..., Float | Fraction],
    *args: Real,
    ctx: Context
):
    """
    Applies either an MPFR function or a real function with
    the given arguments and context depending on the rounding
    requested by the context.

    The MPFR function is expected to take a fixed number of `Float` arguments
    followed by either a precision `p` or a least absolute digit `n`.
    The real function only takes `Float | Fraction` arguments.
    """
    p, n = ctx.round_params()
    if p is None and n is None or any(isinstance(x, Fraction) for x in args):
        # real computation
        r_args = tuple(_cvt_to_real(x) for x in args)
        x = real_fn(*r_args)
        if isinstance(x, Fraction) and isinstance(ctx, RealContext):
            return x  # exact rational number
        else:
            return ctx.round(x)
    else:
        return _apply_mpfr(mpfr_fn, *args, ctx=ctx)

################################################################################
# General operations

def acos(x: Real, ctx: Context = REAL):
    """Computes the inverse cosine of `x` rounded under `ctx`."""
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_acos, x, ctx=ctx)

def acosh(x: Real, ctx: Context = REAL):
    """Computes the inverse hyperbolic cosine of `x` rounded under `ctx`."""
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_acosh, x, ctx=ctx)

def add(x: Real, y: Real, ctx: Context = REAL):
    """Adds `x` and `y` rounded under `ctx`."""
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr_or_real(mpfr_add, real_add, x, y, ctx=ctx)

def asin(x: Real, ctx: Context = REAL):
    """Computes the inverse sine of `x` rounded under `ctx`."""
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_asin, x, ctx=ctx)

def asinh(x: Real, ctx: Context = REAL):
    """Computes the inverse hyperbolic sine of `x` rounded under `ctx`."""
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_asinh, x, ctx=ctx)

def atan(x: Real, ctx: Context = REAL):
    """Computes the inverse tangent of `x` rounded under `ctx`."""
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_atan, x, ctx=ctx)

def atan2(y: Real, x: Real, ctx: Context = REAL):
    """
    Computes `atan(y / x)` taking into account the correct quadrant
    that the point `(x, y)` resides in. The result is rounded under `ctx`.
    """
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_atan2, y, x, ctx=ctx)

def atanh(x: Real, ctx: Context = REAL):
    """Computes the inverse hyperbolic tangent of `x` under `ctx`."""
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_atanh, x, ctx=ctx)

def cbrt(x: Real, ctx: Context = REAL):
    """Computes the cube root of `x` rounded under `ctx`."""
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_cbrt, x, ctx=ctx)

def copysign(x: Real, y: Real, ctx: Context = REAL):
    """Returns `|x| * sign(y)` rounded under `ctx`."""
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_copysign, x, y, ctx=ctx)

def cos(x: Real, ctx: Context = REAL):
    """Computes the cosine of `x` rounded under `ctx`."""
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_cos, x, ctx=ctx)

def cosh(x: Real, ctx: Context = REAL):
    """Computes the hyperbolic cosine `x` rounded under `ctx`."""
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_cosh, x, ctx=ctx)

def div(x: Real, y: Real, ctx: Context = REAL):
    """Computes `x / y` rounded under `ctx`."""
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_div, x, y, ctx=ctx)

def erf(x: Real, ctx: Context = REAL):
    """Computes the error function of `x` rounded under `ctx`."""
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_erf, x, ctx=ctx)

def erfc(x: Real, ctx: Context = REAL):
    """Computes `1 - erf(x)` rounded under `ctx`."""
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_erfc, x, ctx=ctx)

def exp(x: Real, ctx: Context = REAL):
    """Computes `e ** x` rounded under `ctx`."""
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_exp, x, ctx=ctx)

def exp2(x: Real, ctx: Context = REAL):
    """Computes `2 ** x` rounded under `ctx`."""
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_exp2, x, ctx=ctx)

def exp10(x: Real, ctx: Context = REAL):
    """Computes `10 *** x` rounded under `ctx`."""
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_exp10, x, ctx=ctx)

def expm1(x: Real, ctx: Context = REAL):
    """Computes `exp(x) - 1` rounded under `ctx`."""
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_expm1, x, ctx=ctx)

def fabs(x: Real, ctx: Context = REAL):
    """Computes `|x|` rounded under `ctx`."""
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr_or_real(mpfr_fabs, real_abs, x, ctx=ctx)

def fdim(x: Real, y: Real, ctx: Context = REAL):
    """Computes `max(x - y, 0)` rounded under `ctx`."""
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_fdim, x, y, ctx=ctx)

def fma(x: Real, y: Real, z: Real, ctx: Context = REAL):
    """Computes `x * y + z` rounded under `ctx`."""
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr_or_real(mpfr_fma, real_fma, x, y, z, ctx=ctx)

def fmax(x: Real, y: Real, ctx: Context = REAL):
    """Computes `max(x, y)` rounded under `ctx`."""
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr_or_real(mpfr_fmax, max, x, y, ctx=ctx)

def fmin(x: Real, y: Real, ctx: Context = REAL):
    """Computes `min(x, y)` rounded under `ctx`."""
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr_or_real(mpfr_fmin, min, x, y, ctx=ctx)

def fmod(x: Real, y: Real, ctx: Context = REAL):
    """
    Computes the remainder of `x / y` rounded under this context.

    The remainder has the same sign as `x`; it is exactly `x - iquot * y`,
    where `iquot` is the `x / y` with its fractional part truncated.
    """
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_fmod, x, y, ctx=ctx)

def hypot(x: Real, y: Real, ctx: Context = REAL):
    """Computes `sqrt(x * x + y * y)` rounded under `ctx`."""
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_hypot, x, y, ctx=ctx)

def lgamma(x: Real, ctx: Context = REAL):
    """Computes the log-gamma of `x` rounded under `ctx`."""
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_lgamma, x, ctx=ctx)

def log(x: Real, ctx: Context = REAL):
    """Computes `log(x)` rounded under `ctx`."""
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_log, x, ctx=ctx)

def log10(x: Real, ctx: Context = REAL):
    """Computes `log10(x)` rounded under `ctx`."""
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_log10, x, ctx=ctx)

def log1p(x: Real, ctx: Context = REAL):
    """Computes `log1p(x)` rounded under `ctx`."""
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_log1p, x, ctx=ctx)

def log2(x: Real, ctx: Context = REAL):
    """Computes `log2(x)` rounded under `ctx`."""
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_log2, x, ctx=ctx)

def mul(x: Real, y: Real, ctx: Context = REAL):
    """Multiplies `x` and `y` rounded under `ctx`."""
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr_or_real(mpfr_mul, real_mul, x, y, ctx=ctx)

def neg(x: Real, ctx: Context = REAL):
    """Computes `-x` rounded under `ctx`."""
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for ctx={ctx}')
    return _apply_mpfr_or_real(mpfr_neg, real_neg, x, ctx=ctx)

def pow(x: Real, y: Real, ctx: Context = REAL):
    """Computes `x**y` rounded under `ctx`."""
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_pow, x, y, ctx=ctx)

def remainder(x: Real, y: Real, ctx: Context = REAL):
    """
    Computes the remainder of `x / y` rounded under `ctx`.

    The remainder is exactly `x - quo * y`, where `quo` is the
    integral value nearest the exact value of `x / y`.
    """
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_remainder, x, y, ctx=ctx)

def sin(x: Real, ctx: Context = REAL):
    """Computes the sine of `x` rounded under `ctx`."""
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_sin, x, ctx=ctx)

def sinh(x: Real, ctx: Context = REAL):
    """Computes the hyperbolic sine of `x` rounded under `ctx`."""
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_sinh, x, ctx=ctx)

def sqrt(x: Real, ctx: Context = REAL):
    """Computes square-root of `x` rounded under `ctx`."""
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_sqrt, x, ctx=ctx)

def sub(x: Real, y: Real, ctx: Context = REAL):
    """Subtracts `y` from `x` rounded under `ctx`."""
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr_or_real(mpfr_sub, real_sub, x, y, ctx=ctx)

def tan(x: Real, ctx: Context = REAL):
    """Computes the tangent of `x` rounded under `ctx`."""
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_tan, x, ctx=ctx)

def tanh(x: Real, ctx: Context = REAL):
    """Computes the hyperbolic tangent of `x` rounded under `ctx`."""
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_tanh, x, ctx=ctx)

def tgamma(x: Real, ctx: Context = REAL):
    """Computes gamma of `x` rounded under `ctx`."""
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr(mpfr_tgamma, x, ctx=ctx)

#############################################################################
# Rounding operations

def _round(x: Real, ctx: Context | None, exact: bool):
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    match ctx:
        case None | RealContext():
            # real computation; no rounding
            if isinstance(x, Fraction):
                return x  # exact rational number
            else:
                return REAL.round(x)
        case _:
            return ctx.round(x, exact=exact)

def round(x: Real, ctx: Context = REAL):
    """
    Rounds `x` under the given context `ctx`.

    If `ctx` is `None`, this operation is the identity operation.
    """
    return _round(x, ctx, exact=False)

def round_exact(x: Real, ctx: Context = REAL):
    """
    Rounds `x` under the given context `ctx`.

    If `ctx` is `None`, this operation is the identity operation.
    If the operation is not exact, it raises a ValueError.
    """
    return _round(x, ctx, exact=True)

def round_at(x: Real, n: Real, ctx: Context = REAL) -> Float:
    """
    Rounds `x` with least absolute digit `n`, the most significant digit
    that must definitely be rounded off. If `ctx` has bounded precision,
    the actual `n` use may be larger than the one specified.
    """
    n = _cvt_to_float(n)
    if not n.is_integer():
        raise ValueError(f'n={n} must be an integer')
    match ctx:
        case None | RealContext():
            raise ValueError(f'round_at() not supported for ctx={ctx}')
        case _:
            return ctx.round_at(x, int(n))

#############################################################################
# Round-to-integer operations

def ceil(x: Real, ctx: Context = REAL):
    """
    Computes the smallest integer greater than or equal to `x`
    that is representable under `ctx`.

    If the context supports overflow, the result may be infinite.
    """
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    match ctx:
        case None | RealContext():
            # use rounding primitives
            return real_ceil(_cvt_to_real(x))
        case _:
            return ctx.with_params(rm=RoundingMode.RTP).round_integer(x)

def floor(x: Real, ctx: Context = REAL):
    """
    Computes the largest integer less than or equal to `x`
    that is representable under `ctx`.

    If the context supports overflow, the result may be infinite.
    """
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    match ctx:
        case None | RealContext():
            # use rounding primitives
            return real_floor(_cvt_to_real(x))
        case _:
            return ctx.with_params(rm=RoundingMode.RTN).round_integer(x)

def trunc(x: Real, ctx: Context = REAL):
    """
    Computes the integer with the largest magnitude whose
    magnitude is less than or equal to the magnitude of `x`
    that is representable under `ctx`.

    If the context supports overflow, the result may be infinite.
    """
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    match ctx:
        case None | RealContext():
            # use rounding primitives
            return real_trunc(_cvt_to_real(x))
        case _:
            return ctx.with_params(rm=RoundingMode.RTZ).round_integer(x)

def nearbyint(x: Real, ctx: Context = REAL):
    """
    Rounds `x` to a representable integer according to
    the rounding mode of this context.

    If the context supports overflow, the result may be infinite.
    """
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    match ctx:
        case None | RealContext():
            raise RuntimeError('nearbyint() not supported in RealContext')
        case _:
            return ctx.round_integer(x)

def roundint(x: Real, ctx: Context = REAL):
    """
    Rounds `x` to the nearest representable integer,
    rounding ties away from zero in halfway cases.

    If the context supports overflow, the result may be infinite.
    """
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    match ctx:
        case None | RealContext():
            # use rounding primitives
            return real_roundint(_cvt_to_real(x))
        case _:
            return ctx.with_params(rm=RoundingMode.RNA).round_integer(x)

#############################################################################
# Classification

def isnan(x: Real, ctx: Context = REAL) -> bool:
    """Checks if `x` is NaN."""
    x = _cvt_to_float(x)
    return x.isnan

def isinf(x: Real, ctx: Context = REAL) -> bool:
    """Checks if `x` is infinite."""
    x = _cvt_to_float(x)
    return x.isinf

def isfinite(x: Real, ctx: Context = REAL) -> bool:
    """Checks if `x` is finite."""
    x = _cvt_to_real(x)
    return isinstance(x, Fraction) or not x.is_nar()

def isnormal(x: Real, ctx: Context = REAL) -> bool:
    """Checks if `x` is normal (not subnormal, zero, or NaN)."""
    # TODO: what if the argument is a Fraction?
    x = _cvt_to_real(x)
    return isinstance(x, Fraction) or x.is_normal()

def signbit(x: Real, ctx: Context = REAL) -> bool:
    """Checks if the sign bit of `x` is set (i.e., `x` is negative)."""
    # TODO: should all Floats have this property?
    x = _cvt_to_float(x)
    if isinstance(x, Fraction):
        return x < 0
    else:
        return x.s

#############################################################################
# Tensor

def empty(n: Real, ctx: Context = REAL) -> list:
    """
    Initializes an empty list of length `n`.
    """
    n = _cvt_to_float(n)
    if not n.is_integer() or n.is_negative():
        raise ValueError(f'Invalid list size: {n}')
    return [UNINIT for _ in range(int(n))]

def dim(x: list, ctx: Context = REAL):
    """
    Returns the number of dimensions of the tensor `x`.

    Assumes that `x` is not a ragged tensor.
    """
    dim = 0
    while True:
        if isinstance(x, list):
            dim += 1
            if x == []:
                break
            x = x[0]
        else:
            break

    if ctx is None:
        return Float.from_int(dim)
    else:
        return ctx.round(dim)

def size(x: list, dim: Real, ctx: Context = REAL):
    """
    Returns the size of the dimension `dim` of the tensor `x`.

    Assumes that `x` is not a ragged tensor.
    """
    dim = _cvt_to_float(dim)
    if dim.is_zero():
        # size(x, 0) = len(x)
        if ctx is None:
            return Float.from_int(len(x))
        else:
            return ctx.round(len(x))
    else:
        # size(x, n) = size(x[0], n - 1)
        for _ in range(int(dim)):
            x = x[0]
            if not isinstance(x, (list, tuple)):
                raise ValueError(f'dimension `{dim}` is out of bounds for the tensor `{x}`')
        if ctx is None:
            return Float.from_int(len(x))
        else:
            return ctx.round(len(x))

#############################################################################
# Constants

def digits(m: int, e: int, b: int, ctx: Context = REAL) -> Float:
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

def hexfloat(s: str, ctx: Context = REAL) -> Float:
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

def rational(n: int, d: int, ctx: Context = REAL) -> Float:
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

def nan(ctx: Context = REAL) -> Float:
    """
    Creates a `Float` representing NaN (Not a Number).
    The result is rounded under the given context.
    """
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return Float(isnan=True, ctx=ctx)

def inf(ctx: Context = REAL) -> Float:
    """
    Creates a `Float` representing (positive) infinity.
    The result is rounded under the given context.
    """
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return Float(isinf=True, ctx=ctx)

def const_pi(ctx: Context = REAL) -> Float:
    """
    Creates a `Float` representing π (pi).
    The result is rounded under the given context.
    """
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr_constant(Constant.PI, ctx=ctx)

def const_e(ctx: Context = REAL) -> Float:
    """
    Creates a `Float` representing e (Euler's number).
    The result is rounded under the given context.
    """
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr_constant(Constant.E, ctx=ctx)

def const_log2e(ctx: Context = REAL) -> Float:
    """
    Creates a `Float` representing log2(e) (the logarithm of e base 2).
    The result is rounded under the given context.
    """
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr_constant(Constant.LOG2E, ctx=ctx)

def const_log10e(ctx: Context = REAL) -> Float:
    """
    Creates a `Float` representing log10(e) (the logarithm of e base 10).
    The result is rounded under the given context.
    """
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr_constant(Constant.LOG10E, ctx=ctx)

def const_ln2(ctx: Context = REAL) -> Float:
    """
    Creates a `Float` representing ln(2) (the natural logarithm of 2).
    The result is rounded under the given context.
    """
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr_constant(Constant.LN2, ctx=ctx)

def const_pi_2(ctx: Context = REAL) -> Float:
    """
    Creates a `Float` representing π/2 (pi divided by 2).
    The result is rounded under the given context.
    """
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr_constant(Constant.PI_2, ctx=ctx)

def const_pi_4(ctx: Context = REAL) -> Float:
    """
    Creates a `Float` representing π/4 (pi divided by 4).
    The result is rounded under the given context.
    """
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr_constant(Constant.PI_4, ctx=ctx)

def const_1_pi(ctx: Context = REAL) -> Float:
    """
    Creates a `Float` representing 1/π (one divided by pi).
    The result is rounded under the given context.
    """
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr_constant(Constant.M_1_PI, ctx=ctx)

def const_2_pi(ctx: Context = REAL) -> Float:
    """
    Creates a `Float` representing 2/π (two divided by pi).
    The result is rounded under the given context.
    """
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr_constant(Constant.M_2_PI, ctx=ctx)

def const_2_sqrt_pi(ctx: Context = REAL) -> Float:
    """
    Creates a `Float` representing 2/sqrt(π) (two divided by the square root of pi).
    The result is rounded under the given context.
    """
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr_constant(Constant.M_2_SQRTPI, ctx=ctx)

def const_sqrt2(ctx: Context = REAL) -> Float:
    """
    Creates a `Float` representing sqrt(2) (the square root of 2).
    The result is rounded under the given context.
    """
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr_constant(Constant.SQRT2, ctx=ctx)

def const_sqrt1_2(ctx: Context = REAL) -> Float:
    """
    Creates a `Float` representing sqrt(1/2) (the square root of 1/2).
    The result is rounded under the given context.
    """
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')
    return _apply_mpfr_constant(Constant.SQRT1_2, ctx=ctx)
