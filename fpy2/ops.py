"""
Mathematical functions under rounding contexts.
"""

from collections.abc import Callable
from fractions import Fraction

from .number.engine import Engine, ENGINES
from .number import Context, Float, RealContext, RoundingMode, Real, REAL
from .number.gmp import *
from .number.real import (
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
    'mod',
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
    # utilities
    '_cvt_to_real',
    '_cvt_to_float',
]

################################################################################
# Utils

def _cvt_to_real(x: Real) -> Float | Fraction:
    """Converts `x` to `Float` or `Fraction`."""
    match x:
        case Float():
            return x
        case Fraction():
            return Float.from_rational(x) if is_dyadic(x) else x
        case int():
            return Float.from_int(x)
        case float():
            return Float.from_float(x)
        case _:
            raise TypeError(f'Expected \'Float\' or \'Fraction\', got \'{type(x)}\' for x={x}')

def _cvt_to_float(x: Real) -> Float:
    """Converts `x` to `Float`."""
    t = _cvt_to_real(x)
    match t:
        case Float():
            return t
        case Fraction():
            raise ValueError(f'Cannot convert non-dyadic rational {t} to Float')
        case _:
            raise TypeError(f'Expected \'Float\' or \'Fraction\', got \'{type(t)}\' for x={x}')

################################################################################
# General operations

def acos(x: Real, ctx: Context = REAL):
    """Computes the inverse cosine of `x` rounded under `ctx`."""
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')

    xr = _cvt_to_real(x)
    for engine in ENGINES:
        r = engine.acos(xr, ctx)
        if r is not None:
            return ctx.round(r)

    raise NotImplementedError(f'acos() not implemented for ctx={ctx}')

def acosh(x: Real, ctx: Context = REAL):
    """Computes the inverse hyperbolic cosine of `x` rounded under `ctx`."""
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')

    xr = _cvt_to_real(x)
    for engine in ENGINES:
        r = engine.acosh(xr, ctx)
        if r is not None:
            return ctx.round(r)

    raise NotImplementedError(f'acosh() not implemented for ctx={ctx}')

def add(x: Real, y: Real, ctx: Context = REAL):
    """Adds `x` and `y` rounded under `ctx`."""
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')

    xr = _cvt_to_real(x)
    yr = _cvt_to_real(y)
    for engine in ENGINES:
        r = engine.add(xr, yr, ctx)
        if r is not None:
            return ctx.round(r)

    raise NotImplementedError(f'add() not implemented for ctx={ctx}')

def asin(x: Real, ctx: Context = REAL):
    """Computes the inverse sine of `x` rounded under `ctx`."""
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')

    xr = _cvt_to_real(x)
    for engine in ENGINES:
        r = engine.asin(xr, ctx)
        if r is not None:
            return ctx.round(r)

    raise NotImplementedError(f'asin() not implemented for ctx={ctx}')

def asinh(x: Real, ctx: Context = REAL):
    """Computes the inverse hyperbolic sine of `x` rounded under `ctx`."""
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')

    xr = _cvt_to_real(x)
    for engine in ENGINES:
        r = engine.asinh(xr, ctx)
        if r is not None:
            return ctx.round(r)

    raise NotImplementedError(f'asinh() not implemented for ctx={ctx}')

def atan(x: Real, ctx: Context = REAL):
    """Computes the inverse tangent of `x` rounded under `ctx`."""
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')

    xr = _cvt_to_real(x)
    for engine in ENGINES:
        r = engine.atan(xr, ctx)
        if r is not None:
            return ctx.round(r)

    raise NotImplementedError(f'atan() not implemented for ctx={ctx}')

def atan2(y: Real, x: Real, ctx: Context = REAL):
    """
    Computes `atan(y / x)` taking into account the correct quadrant
    that the point `(x, y)` resides in. The result is rounded under `ctx`.
    """
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')

    yr = _cvt_to_real(y)
    xr = _cvt_to_real(x)
    for engine in ENGINES:
        r = engine.atan2(yr, xr, ctx)
        if r is not None:
            return ctx.round(r)

    raise NotImplementedError(f'atan2() not implemented for ctx={ctx}')

def atanh(x: Real, ctx: Context = REAL):
    """Computes the inverse hyperbolic tangent of `x` under `ctx`."""
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')

    xr = _cvt_to_real(x)
    for engine in ENGINES:
        r = engine.atanh(xr, ctx)
        if r is not None:
            return ctx.round(r)

    raise NotImplementedError(f'atanh() not implemented for ctx={ctx}')

def cbrt(x: Real, ctx: Context = REAL):
    """Computes the cube root of `x` rounded under `ctx`."""
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')

    xr = _cvt_to_real(x)
    for engine in ENGINES:
        r = engine.cbrt(xr, ctx)
        if r is not None:
            return ctx.round(r)

    raise NotImplementedError(f'cbrt() not implemented for ctx={ctx}')

def copysign(x: Real, y: Real, ctx: Context = REAL):
    """Returns `|x| * sign(y)` rounded under `ctx`."""
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')

    xr = _cvt_to_real(x)
    yr = _cvt_to_real(y)
    for engine in ENGINES:
        r = engine.copysign(xr, yr, ctx)
        if r is not None:
            return ctx.round(r)

    raise NotImplementedError(f'copysign() not implemented for ctx={ctx}')

def cos(x: Real, ctx: Context = REAL):
    """Computes the cosine of `x` rounded under `ctx`."""
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')

    xr = _cvt_to_real(x)
    for engine in ENGINES:
        r = engine.cos(xr, ctx)
        if r is not None:
            return ctx.round(r)

    raise NotImplementedError(f'cos() not implemented for ctx={ctx}')

def cosh(x: Real, ctx: Context = REAL):
    """Computes the hyperbolic cosine `x` rounded under `ctx`."""
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')

    xr = _cvt_to_real(x)
    for engine in ENGINES:
        r = engine.cosh(xr, ctx)
        if r is not None:
            return ctx.round(r)

    raise NotImplementedError(f'cosh() not implemented for ctx={ctx}')

def div(x: Real, y: Real, ctx: Context = REAL):
    """Computes `x / y` rounded under `ctx`."""
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')

    xr = _cvt_to_real(x)
    yr = _cvt_to_real(y)
    for engine in ENGINES:
        r = engine.div(xr, yr, ctx)
        if r is not None:
            return ctx.round(r)

    raise NotImplementedError(f'div() not implemented for ctx={ctx}')

def erf(x: Real, ctx: Context = REAL):
    """Computes the error function of `x` rounded under `ctx`."""
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')

    xr = _cvt_to_real(x)
    for engine in ENGINES:
        r = engine.erf(xr, ctx)
        if r is not None:
            return ctx.round(r)

    raise NotImplementedError(f'erf() not implemented for ctx={ctx}')

def erfc(x: Real, ctx: Context = REAL):
    """Computes `1 - erf(x)` rounded under `ctx`."""
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')

    xr = _cvt_to_real(x)
    for engine in ENGINES:
        r = engine.erfc(xr, ctx)
        if r is not None:
            return ctx.round(r)

    raise NotImplementedError(f'erfc() not implemented for ctx={ctx}')

def exp(x: Real, ctx: Context = REAL):
    """Computes `e ** x` rounded under `ctx`."""
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')

    xr = _cvt_to_real(x)
    for engine in ENGINES:
        r = engine.exp(xr, ctx)
        if r is not None:
            return ctx.round(r)

    raise NotImplementedError(f'exp() not implemented for ctx={ctx}')

def exp2(x: Real, ctx: Context = REAL):
    """Computes `2 ** x` rounded under `ctx`."""
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')

    xr = _cvt_to_real(x)
    for engine in ENGINES:
        r = engine.exp2(xr, ctx)
        if r is not None:
            return ctx.round(r)

    raise NotImplementedError(f'exp2() not implemented for ctx={ctx}')

def exp10(x: Real, ctx: Context = REAL):
    """Computes `10 *** x` rounded under `ctx`."""
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')

    xr = _cvt_to_real(x)
    for engine in ENGINES:
        r = engine.exp10(xr, ctx)
        if r is not None:
            return ctx.round(r)

    raise NotImplementedError(f'exp10() not implemented for ctx={ctx}')

def expm1(x: Real, ctx: Context = REAL):
    """Computes `exp(x) - 1` rounded under `ctx`."""
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')

    xr = _cvt_to_real(x)
    for engine in ENGINES:
        r = engine.expm1(xr, ctx)
        if r is not None:
            return ctx.round(r)

    raise NotImplementedError(f'expm1() not implemented for ctx={ctx}')

def fabs(x: Real, ctx: Context = REAL):
    """Computes `|x|` rounded under `ctx`."""
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')

    xr = _cvt_to_real(x)
    for engine in ENGINES:
        r = engine.fabs(xr, ctx)
        if r is not None:
            return ctx.round(r)

    raise NotImplementedError(f'fabs() not implemented for ctx={ctx}')

def fdim(x: Real, y: Real, ctx: Context = REAL):
    """Computes `max(x - y, 0)` rounded under `ctx`."""
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')

    xr = _cvt_to_real(x)
    yr = _cvt_to_real(y)
    for engine in ENGINES:
        r = engine.fdim(xr, yr, ctx)
        if r is not None:
            return ctx.round(r)

    raise NotImplementedError(f'fdim() not implemented for ctx={ctx}')

def fma(x: Real, y: Real, z: Real, ctx: Context = REAL):
    """Computes `x * y + z` rounded under `ctx`."""
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')

    xr = _cvt_to_real(x)
    yr = _cvt_to_real(y)
    zr = _cvt_to_real(z)
    for engine in ENGINES:
        r = engine.fma(xr, yr, zr, ctx)
        if r is not None:
            return ctx.round(r)

    raise NotImplementedError(f'fma() not implemented for ctx={ctx}')

def fmax(x: Real, y: Real, ctx: Context = REAL):
    """Computes `max(x, y)` rounded under `ctx`."""
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')

    xr = _cvt_to_real(x)
    yr = _cvt_to_real(y)
    for engine in ENGINES:
        r = engine.fmax(xr, yr, ctx)
        if r is not None:
            return ctx.round(r)

    raise NotImplementedError(f'fmax() not implemented for ctx={ctx}')

def fmin(x: Real, y: Real, ctx: Context = REAL):
    """Computes `min(x, y)` rounded under `ctx`."""
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')

    xr = _cvt_to_real(x)
    yr = _cvt_to_real(y)
    for engine in ENGINES:
        r = engine.fmin(xr, yr, ctx)
        if r is not None:
            return ctx.round(r)

    raise NotImplementedError(f'fmin() not implemented for ctx={ctx}')

def fmod(x: Real, y: Real, ctx: Context = REAL):
    """
    Computes the remainder of `x / y` rounded under this context.

    The remainder has the same sign as `x`; it is exactly `x - iquot * y`,
    where `iquot` is the `x / y` with its fractional part truncated.
    """
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')

    xr = _cvt_to_real(x)
    yr = _cvt_to_real(y)
    for engine in ENGINES:
        r = engine.fmod(xr, yr, ctx)
        if r is not None:
            return ctx.round(r)

    raise NotImplementedError(f'fmod() not implemented for ctx={ctx}')

def hypot(x: Real, y: Real, ctx: Context = REAL):
    """Computes `sqrt(x * x + y * y)` rounded under `ctx`."""
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')

    xr = _cvt_to_real(x)
    yr = _cvt_to_real(y)
    for engine in ENGINES:
        r = engine.hypot(xr, yr, ctx)
        if r is not None:
            return ctx.round(r)

    raise NotImplementedError(f'hypot() not implemented for ctx={ctx}')

def lgamma(x: Real, ctx: Context = REAL):
    """Computes the log-gamma of `x` rounded under `ctx`."""
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')

    xr = _cvt_to_real(x)
    for engine in ENGINES:
        r = engine.lgamma(xr, ctx)
        if r is not None:
            return ctx.round(r)

    raise NotImplementedError(f'lgamma() not implemented for ctx={ctx}')

def log(x: Real, ctx: Context = REAL):
    """Computes `log(x)` rounded under `ctx`."""
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')

    xr = _cvt_to_real(x)
    for engine in ENGINES:
        r = engine.log(xr, ctx)
        if r is not None:
            return ctx.round(r)

    raise NotImplementedError(f'log() not implemented for ctx={ctx}')

def log10(x: Real, ctx: Context = REAL):
    """Computes `log10(x)` rounded under `ctx`."""
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')

    xr = _cvt_to_real(x)
    for engine in ENGINES:
        r = engine.log10(xr, ctx)
        if r is not None:
            return ctx.round(r)

    raise NotImplementedError(f'log10() not implemented for ctx={ctx}')

def log1p(x: Real, ctx: Context = REAL):
    """Computes `log1p(x)` rounded under `ctx`."""
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')

    xr = _cvt_to_real(x)
    for engine in ENGINES:
        r = engine.log1p(xr, ctx)
        if r is not None:
            return ctx.round(r)

    raise NotImplementedError(f'log1p() not implemented for ctx={ctx}')

def log2(x: Real, ctx: Context = REAL):
    """Computes `log2(x)` rounded under `ctx`."""
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')

    xr = _cvt_to_real(x)
    for engine in ENGINES:
        r = engine.log2(xr, ctx)
        if r is not None:
            return ctx.round(r)

    raise NotImplementedError(f'log2() not implemented for ctx={ctx}')

def mod(x: Real, y: Real, ctx: Context = REAL):
    """
    Computes `x % y` rounded under `ctx`.

    Implements Python's modulus operator, defined as:

        x % y = x - floor(x / y) * y
    """
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')

    xr = _cvt_to_real(x)
    yr = _cvt_to_real(y)
    for engine in ENGINES:
        r = engine.mod(xr, yr, ctx)
        if r is not None:
            return ctx.round(r)

    raise NotImplementedError(f'mod() not implemented for ctx={ctx}')

def mul(x: Real, y: Real, ctx: Context = REAL):
    """Multiplies `x` and `y` rounded under `ctx`."""
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')

    xr = _cvt_to_real(x)
    yr = _cvt_to_real(y)
    for engine in ENGINES:
        r = engine.mul(xr, yr, ctx)
        if r is not None:
            return ctx.round(r)

    raise NotImplementedError(f'mul() not implemented for ctx={ctx}')

def neg(x: Real, ctx: Context = REAL):
    """Computes `-x` rounded under `ctx`."""
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for ctx={ctx}')

    xr = _cvt_to_real(x)
    for engine in ENGINES:
        r = engine.neg(xr, ctx)
        if r is not None:
            return ctx.round(r)

    raise NotImplementedError(f'neg() not implemented for ctx={ctx}')

def pow(x: Real, y: Real, ctx: Context = REAL):
    """Computes `x**y` rounded under `ctx`."""
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')

    xr = _cvt_to_real(x)
    yr = _cvt_to_real(y)
    for engine in ENGINES:
        r = engine.pow(xr, yr, ctx)
        if r is not None:
            return ctx.round(r)

    raise NotImplementedError(f'pow() not implemented for ctx={ctx}')

def remainder(x: Real, y: Real, ctx: Context = REAL):
    """
    Computes the remainder of `x / y` rounded under `ctx`.

    The remainder is exactly `x - quo * y`, where `quo` is the
    integral value nearest the exact value of `x / y`.
    """
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')

    xr = _cvt_to_real(x)
    yr = _cvt_to_real(y)
    for engine in ENGINES:
        r = engine.remainder(xr, yr, ctx)
        if r is not None:
            return ctx.round(r)

    raise NotImplementedError(f'remainder() not implemented for ctx={ctx}')

def sin(x: Real, ctx: Context = REAL):
    """Computes the sine of `x` rounded under `ctx`."""
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')

    xr = _cvt_to_real(x)
    for engine in ENGINES:
        r = engine.sin(xr, ctx)
        if r is not None:
            return ctx.round(r)

    raise NotImplementedError(f'sin() not implemented for ctx={ctx}')

def sinh(x: Real, ctx: Context = REAL):
    """Computes the hyperbolic sine of `x` rounded under `ctx`."""
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')

    xr = _cvt_to_real(x)
    for engine in ENGINES:
        r = engine.sinh(xr, ctx)
        if r is not None:
            return ctx.round(r)

    raise NotImplementedError(f'sinh() not implemented for ctx={ctx}')

def sqrt(x: Real, ctx: Context = REAL):
    """Computes square-root of `x` rounded under `ctx`."""
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')

    xr = _cvt_to_real(x)
    for engine in ENGINES:
        r = engine.sqrt(xr, ctx)
        if r is not None:
            return ctx.round(r)

    raise NotImplementedError(f'sqrt() not implemented for ctx={ctx}')

def sub(x: Real, y: Real, ctx: Context = REAL):
    """Subtracts `y` from `x` rounded under `ctx`."""
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')

    xr = _cvt_to_real(x)
    yr = _cvt_to_real(y)
    for engine in ENGINES:
        r = engine.sub(xr, yr, ctx)
        if r is not None:
            return ctx.round(r)

    raise NotImplementedError(f'sub() not implemented for ctx={ctx}')

def tan(x: Real, ctx: Context = REAL):
    """Computes the tangent of `x` rounded under `ctx`."""
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')

    xr = _cvt_to_real(x)
    for engine in ENGINES:
        r = engine.tan(xr, ctx)
        if r is not None:
            return ctx.round(r)

    raise NotImplementedError(f'tan() not implemented for ctx={ctx}')

def tanh(x: Real, ctx: Context = REAL):
    """Computes the hyperbolic tangent of `x` rounded under `ctx`."""
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')

    xr = _cvt_to_real(x)
    for engine in ENGINES:
        r = engine.tanh(xr, ctx)
        if r is not None:
            return ctx.round(r)

    raise NotImplementedError(f'tanh() not implemented for ctx={ctx}')

def tgamma(x: Real, ctx: Context = REAL):
    """Computes gamma of `x` rounded under `ctx`."""
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')

    xr = _cvt_to_real(x)
    for engine in ENGINES:
        r = engine.tgamma(xr, ctx)
        if r is not None:
            return ctx.round(r)

    raise NotImplementedError(f'tgamma() not implemented for ctx={ctx}')

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

    for engine in ENGINES:
        r = engine.const_pi(ctx)
        if r is not None:
            return ctx.round(r)

    raise NotImplementedError(f'const_pi() not implemented for ctx={ctx}')

def const_e(ctx: Context = REAL) -> Float:
    """
    Creates a `Float` representing e (Euler's number).
    The result is rounded under the given context.
    """
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')

    for engine in ENGINES:
        r = engine.const_e(ctx)
        if r is not None:
            return ctx.round(r)

    raise NotImplementedError(f'const_e() not implemented for ctx={ctx}')

def const_log2e(ctx: Context = REAL) -> Float:
    """
    Creates a `Float` representing log2(e) (the logarithm of e base 2).
    The result is rounded under the given context.
    """
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')

    for engine in ENGINES:
        r = engine.const_log2e(ctx)
        if r is not None:
            return ctx.round(r)

    raise NotImplementedError(f'const_log2e() not implemented for ctx={ctx}')

def const_log10e(ctx: Context = REAL) -> Float:
    """
    Creates a `Float` representing log10(e) (the logarithm of e base 10).
    The result is rounded under the given context.
    """
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')

    for engine in ENGINES:
        r = engine.const_log10e(ctx)
        if r is not None:
            return ctx.round(r)

    raise NotImplementedError(f'const_log10e() not implemented for ctx={ctx}')

def const_ln2(ctx: Context = REAL) -> Float:
    """
    Creates a `Float` representing ln(2) (the natural logarithm of 2).
    The result is rounded under the given context.
    """
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')

    for engine in ENGINES:
        r = engine.const_ln2(ctx)
        if r is not None:
            return ctx.round(r)

    raise NotImplementedError(f'const_ln2() not implemented for ctx={ctx}')

def const_pi_2(ctx: Context = REAL) -> Float:
    """
    Creates a `Float` representing π/2 (pi divided by 2).
    The result is rounded under the given context.
    """
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')

    for engine in ENGINES:
        r = engine.const_pi_2(ctx)
        if r is not None:
            return ctx.round(r)

    raise NotImplementedError(f'const_pi_2() not implemented for ctx={ctx}')

def const_pi_4(ctx: Context = REAL) -> Float:
    """
    Creates a `Float` representing π/4 (pi divided by 4).
    The result is rounded under the given context.
    """
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')

    for engine in ENGINES:
        r = engine.const_pi_4(ctx)
        if r is not None:
            return ctx.round(r)

    raise NotImplementedError(f'const_pi_4() not implemented for ctx={ctx}')

def const_1_pi(ctx: Context = REAL) -> Float:
    """
    Creates a `Float` representing 1/π (one divided by pi).
    The result is rounded under the given context.
    """
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')

    for engine in ENGINES:
        r = engine.const_1_pi(ctx)
        if r is not None:
            return ctx.round(r)

    raise NotImplementedError(f'const_1_pi() not implemented for ctx={ctx}')

def const_2_pi(ctx: Context = REAL) -> Float:
    """
    Creates a `Float` representing 2/π (two divided by pi).
    The result is rounded under the given context.
    """
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')

    for engine in ENGINES:
        r = engine.const_2_pi(ctx)
        if r is not None:
            return ctx.round(r)

    raise NotImplementedError(f'const_2_pi() not implemented for ctx={ctx}')

def const_2_sqrt_pi(ctx: Context = REAL) -> Float:
    """
    Creates a `Float` representing 2/sqrt(π) (two divided by the square root of pi).
    The result is rounded under the given context.
    """
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')

    for engine in ENGINES:
        r = engine.const_2_sqrtpi(ctx)
        if r is not None:
            return ctx.round(r)

    raise NotImplementedError(f'const_2_sqrt_pi() not implemented for ctx={ctx}')

def const_sqrt2(ctx: Context = REAL) -> Float:
    """
    Creates a `Float` representing sqrt(2) (the square root of 2).
    The result is rounded under the given context.
    """
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')

    for engine in ENGINES:
        r = engine.const_sqrt2(ctx)
        if r is not None:
            return ctx.round(r)

    raise NotImplementedError(f'const_sqrt2() not implemented for ctx={ctx}')

def const_sqrt1_2(ctx: Context = REAL) -> Float:
    """
    Creates a `Float` representing sqrt(1/2) (the square root of 1/2).
    The result is rounded under the given context.
    """
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for x={ctx}')

    for engine in ENGINES:
        r = engine.const_sqrt1_2(ctx)
        if r is not None:
            return ctx.round(r)

    raise NotImplementedError(f'const_sqrt1_2() not implemented for ctx={ctx}')
