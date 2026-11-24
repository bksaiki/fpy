"""
Mathematical functions under rounding contexts.
"""

from collections.abc import Callable, Sequence
from fractions import Fraction
from typing import overload

from .number import REAL, RM, Context, Float, Real, RealFloat
from .number.engine import ENGINES, Engine
from .number.engine.engine import EngineRes
from .utils import UNINIT, digits_to_fraction, hexnum_to_fraction, is_dyadic

__all__ = [
    '_cvt_to_float',
    # utilities
    '_cvt_to_real',
    # General operations
    'acos',
    'acosh',
    'add',
    'asin',
    'asinh',
    'atan',
    'atan2',
    'atanh',
    'cast',
    'cbrt',
    # Round-to-integer operations
    'ceil',
    'const_1_pi',
    'const_2_pi',
    'const_2_sqrt_pi',
    'const_e',
    'const_ln2',
    'const_log2e',
    'const_log10e',
    'const_pi',
    'const_pi_2',
    'const_pi_4',
    'const_sqrt1_2',
    'const_sqrt2',
    'copysign',
    'cos',
    'cosh',
    # Constants
    'digits',
    'dim',
    'div',
    # Tensor
    'empty',
    'erf',
    'erfc',
    'exp',
    'exp2',
    'exp10',
    'expm1',
    'fabs',
    'fdim',
    'floor',
    'fma',
    'fmax',
    'fmin',
    'fmod',
    # Tuple
    'fst',
    'hexfloat',
    'hypot',
    'inf',
    'isfinite',
    'isinf',
    # Classification
    'isnan',
    'isnormal',
    'lgamma',
    'log',
    'log1p',
    'log2',
    'log10',
    'logb',
    'mod',
    'mul',
    'nan',
    'nearbyint',
    'neg',
    'pow',
    'rational',
    'remainder',
    # Rounding operations
    'round',
    'round_at',
    'round_exact',
    'roundint',
    # Numerical data
    'signbit',
    'sin',
    'sinh',
    'size',
    'snd',
    'sqrt',
    'sub',
    'tan',
    'tanh',
    'tgamma',
    'trunc',
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

def _normalize(x: Float | Fraction, ctx: Context, args: tuple[Float | Fraction, ...] = ()) -> Float | Fraction:
    if ctx is REAL and isinstance(x, Fraction):
        return x
    else:
        result = ctx.round(x)
        if args and isinstance(result, Float):
            if result.isnan:
                if not any(isinstance(a, Float) and a.isnan for a in args):
                    result._real._flags._set_invalid(True)
            elif result.isinf and not result.inexact:
                if all(isinstance(a, Fraction) or not a.is_nar() for a in args):
                    result._real._flags._set_divzero(True)
        return result


def _check_ctx(ctx: object) -> None:
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for ctx={ctx}')

type _Op = Callable[[Engine], Callable[..., EngineRes]]

def _dispatch(op: _Op, ctx: Context, *args: Real) -> tuple[Float | Fraction, tuple[Float | Fraction, ...]]:
    """The first engine's result for *op* over *args*, and *args* as reals."""
    _check_ctx(ctx)
    xs = tuple(_cvt_to_real(x) for x in args)
    for engine in ENGINES:
        r = op(engine)(*xs, ctx)
        if r is not None:
            return r, xs
    raise NotImplementedError(f'{op(Engine()).__name__}() not implemented for ctx={ctx}')

def _eval(op: _Op, ctx: Context, *args: Real) -> Float | Fraction:
    """*op* over *args*, rounded under *ctx*."""
    r, xs = _dispatch(op, ctx, *args)
    return _normalize(r, ctx, xs)

def _const(op: _Op, ctx: Context) -> Float:
    """The constant *op* rounded under *ctx*; no engine computes one exactly."""
    r = _eval(op, ctx)
    assert isinstance(r, Float)
    return r


def _round_integer(op: _Op, ctx: Context, x: Real) -> Float | Fraction:
    """The integer *op* rounds *x* to under *ctx*, inexact where it differs from *x*."""
    r, _ = _dispatch(op, ctx, x)
    y = _normalize(r, ctx)
    if isinstance(y, Float) and y.is_finite() and y != x:
        y._real._flags._set_inexact(True)
    return y

def _rational(x: Float | Fraction) -> Fraction:
    return x if isinstance(x, Fraction) else x.as_rational()


def _zero_sum(r: Float | Fraction, ctx: Context, terms: tuple[Float | Fraction, ...],
              zero_signs: tuple[bool, ...]) -> Float | Fraction:
    """*r*, the sum of *terms*, with IEEE 754's sign when that sum is exactly
    zero (Sec. 6.3): its terms' sign when all are zeros of that one sign
    (*zero_signs*, a zero term's sign), else -0 rounding toward negative and
    +0 otherwise.  Only the toward-negative case differs from the engines'."""
    if (ctx.rounding_mode() is not RM.RTN or not isinstance(r, Float)
            or not r.is_zero() or r.s):
        return r
    if len(zero_signs) == len(terms) and len(set(zero_signs)) == 1:
        return r
    if any(isinstance(t, Float) and t.is_nar() for t in terms) or sum(map(_rational, terms)) != 0:
        return r
    return ctx.round(Float.from_float(-0.0))


def _signbit_of(x: Float | Fraction) -> bool:
    return x.s if isinstance(x, Float) else x < 0


def _zero_sign(x: Float | Fraction) -> tuple[bool, ...]:
    """`(sign,)` for a zero *x*, else `()`."""
    return (x.s,) if isinstance(x, Float) and x.is_zero() else ()


def _empty(dims_list: list[int]) -> list:
    if len(dims_list) == 1:
        return [UNINIT for _ in range(dims_list[0])]
    else:
        return [_empty(dims_list[1:]) for _ in range(dims_list[0])]

def _flatten(x: list) -> list:
    """Flattens a nested list of scalars into a flat list."""
    result = []
    for item in x:
        if isinstance(item, list):
            result.extend(_flatten(item))
        else:
            result.append(item)
    return result

################################################################################
# General operations

def acos(x: Real, ctx: Context = REAL) -> Float | Fraction:
    """Computes the inverse cosine of `x` rounded under `ctx`."""
    return _eval(lambda e: e.acos, ctx, x)

def acosh(x: Real, ctx: Context = REAL) -> Float | Fraction:
    """Computes the inverse hyperbolic cosine of `x` rounded under `ctx`."""
    return _eval(lambda e: e.acosh, ctx, x)

def add(x: Real, y: Real, ctx: Context = REAL) -> Float | Fraction:
    """Adds `x` and `y` rounded under `ctx`."""
    r, (xr, yr) = _dispatch(lambda e: e.add, ctx, x, y)
    return _zero_sum(_normalize(r, ctx, (xr, yr)), ctx, (xr, yr),
                     _zero_sign(xr) + _zero_sign(yr))

def asin(x: Real, ctx: Context = REAL) -> Float | Fraction:
    """Computes the inverse sine of `x` rounded under `ctx`."""
    return _eval(lambda e: e.asin, ctx, x)

def asinh(x: Real, ctx: Context = REAL) -> Float | Fraction:
    """Computes the inverse hyperbolic sine of `x` rounded under `ctx`."""
    return _eval(lambda e: e.asinh, ctx, x)

def atan(x: Real, ctx: Context = REAL) -> Float | Fraction:
    """Computes the inverse tangent of `x` rounded under `ctx`."""
    return _eval(lambda e: e.atan, ctx, x)

def atan2(y: Real, x: Real, ctx: Context = REAL) -> Float | Fraction:
    """
    Computes `atan(y / x)` taking into account the correct quadrant
    that the point `(x, y)` resides in. The result is rounded under `ctx`.
    """
    return _eval(lambda e: e.atan2, ctx, y, x)

def atanh(x: Real, ctx: Context = REAL) -> Float | Fraction:
    """Computes the inverse hyperbolic tangent of `x` under `ctx`."""
    return _eval(lambda e: e.atanh, ctx, x)

def cbrt(x: Real, ctx: Context = REAL) -> Float | Fraction:
    """Computes the cube root of `x` rounded under `ctx`."""
    return _eval(lambda e: e.cbrt, ctx, x)

def copysign(x: Real, y: Real, ctx: Context = REAL) -> Float | Fraction:
    """Returns `|x| * sign(y)` rounded under `ctx`."""
    return _eval(lambda e: e.copysign, ctx, x, y)

def cos(x: Real, ctx: Context = REAL) -> Float | Fraction:
    """Computes the cosine of `x` rounded under `ctx`."""
    return _eval(lambda e: e.cos, ctx, x)

def cosh(x: Real, ctx: Context = REAL) -> Float | Fraction:
    """Computes the hyperbolic cosine `x` rounded under `ctx`."""
    return _eval(lambda e: e.cosh, ctx, x)

def div(x: Real, y: Real, ctx: Context = REAL) -> Float | Fraction:
    """Computes `x / y` rounded under `ctx`."""
    return _eval(lambda e: e.div, ctx, x, y)

def erf(x: Real, ctx: Context = REAL) -> Float | Fraction:
    """Computes the error function of `x` rounded under `ctx`."""
    return _eval(lambda e: e.erf, ctx, x)

def erfc(x: Real, ctx: Context = REAL) -> Float | Fraction:
    """Computes `1 - erf(x)` rounded under `ctx`."""
    return _eval(lambda e: e.erfc, ctx, x)

def exp(x: Real, ctx: Context = REAL) -> Float | Fraction:
    """Computes `e ** x` rounded under `ctx`."""
    return _eval(lambda e: e.exp, ctx, x)

def exp2(x: Real, ctx: Context = REAL) -> Float | Fraction:
    """Computes `2 ** x` rounded under `ctx`."""
    return _eval(lambda e: e.exp2, ctx, x)

def exp10(x: Real, ctx: Context = REAL) -> Float | Fraction:
    """Computes `10 *** x` rounded under `ctx`."""
    return _eval(lambda e: e.exp10, ctx, x)

def expm1(x: Real, ctx: Context = REAL) -> Float | Fraction:
    """Computes `exp(x) - 1` rounded under `ctx`."""
    return _eval(lambda e: e.expm1, ctx, x)

def fabs(x: Real, ctx: Context = REAL) -> Float | Fraction:
    """Computes `|x|` rounded under `ctx`."""
    return _eval(lambda e: e.fabs, ctx, x)

def fdim(x: Real, y: Real, ctx: Context = REAL) -> Float | Fraction:
    """Computes `max(x - y, 0)` rounded under `ctx`."""
    return _eval(lambda e: e.fdim, ctx, x, y)

def fma(x: Real, y: Real, z: Real, ctx: Context = REAL) -> Float | Fraction:
    """Computes `x * y + z` rounded under `ctx`."""
    r, (xr, yr, zr) = _dispatch(lambda e: e.fma, ctx, x, y, z)
    out = _normalize(r, ctx, (xr, yr, zr))
    if any(isinstance(t, Float) and t.is_nar() for t in (xr, yr)):
        return out
    # the exact product's sign, when it is zero
    p = _rational(xr) * _rational(yr)
    ps = (_signbit_of(xr) != _signbit_of(yr),) if p == 0 else ()
    return _zero_sum(out, ctx, (p, zr), ps + _zero_sign(zr))

def fmax(x: Real, y: Real, ctx: Context = REAL) -> Float | Fraction:
    """Computes `max(x, y)` rounded under `ctx`."""
    return _eval(lambda e: e.fmax, ctx, x, y)

def fmin(x: Real, y: Real, ctx: Context = REAL) -> Float | Fraction:
    """Computes `min(x, y)` rounded under `ctx`."""
    return _eval(lambda e: e.fmin, ctx, x, y)

def fmod(x: Real, y: Real, ctx: Context = REAL) -> Float | Fraction:
    """
    Computes the remainder of `x / y` rounded under this context.

    The remainder has the same sign as `x`; it is exactly `x - iquot * y`,
    where `iquot` is the `x / y` with its fractional part truncated.
    """
    return _eval(lambda e: e.fmod, ctx, x, y)

def hypot(x: Real, y: Real, ctx: Context = REAL) -> Float | Fraction:
    """Computes `sqrt(x * x + y * y)` rounded under `ctx`."""
    return _eval(lambda e: e.hypot, ctx, x, y)

def lgamma(x: Real, ctx: Context = REAL) -> Float | Fraction:
    """Computes the log-gamma of `x` rounded under `ctx`."""
    return _eval(lambda e: e.lgamma, ctx, x)

def log(x: Real, ctx: Context = REAL) -> Float | Fraction:
    """Computes `log(x)` rounded under `ctx`."""
    return _eval(lambda e: e.log, ctx, x)

def log10(x: Real, ctx: Context = REAL) -> Float | Fraction:
    """Computes `log10(x)` rounded under `ctx`."""
    return _eval(lambda e: e.log10, ctx, x)

def log1p(x: Real, ctx: Context = REAL) -> Float | Fraction:
    """Computes `log1p(x)` rounded under `ctx`."""
    return _eval(lambda e: e.log1p, ctx, x)

def log2(x: Real, ctx: Context = REAL) -> Float | Fraction:
    """Computes `log2(x)` rounded under `ctx`."""
    return _eval(lambda e: e.log2, ctx, x)

def mod(x: Real, y: Real, ctx: Context = REAL) -> Float | Fraction:
    """
    Computes `x % y` rounded under `ctx`.

    Implements Python's modulus operator, defined as:

        x % y = x - floor(x / y) * y
    """
    return _eval(lambda e: e.mod, ctx, x, y)

def mul(x: Real, y: Real, ctx: Context = REAL) -> Float | Fraction:
    """Multiplies `x` and `y` rounded under `ctx`."""
    return _eval(lambda e: e.mul, ctx, x, y)

def neg(x: Real, ctx: Context = REAL) -> Float | Fraction:
    """Computes `-x` rounded under `ctx`."""
    if ctx is not None and not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\' or \'None\', got \'{type(ctx)}\' for ctx={ctx}')

    xr = _cvt_to_real(x)
    for engine in ENGINES:
        r = engine.neg(xr, ctx)
        if r is not None:
            return _normalize(r, ctx, (xr,))

    raise NotImplementedError(f'neg() not implemented for ctx={ctx}')

def pow(x: Real, y: Real, ctx: Context = REAL) -> Float | Fraction:
    """Computes `x**y` rounded under `ctx`."""
    return _eval(lambda e: e.pow, ctx, x, y)

def remainder(x: Real, y: Real, ctx: Context = REAL) -> Float | Fraction:
    """
    Computes the remainder of `x / y` rounded under `ctx`.

    The remainder is exactly `x - quo * y`, where `quo` is the
    integral value nearest the exact value of `x / y`.
    """
    return _eval(lambda e: e.remainder, ctx, x, y)

def sin(x: Real, ctx: Context = REAL) -> Float | Fraction:
    """Computes the sine of `x` rounded under `ctx`."""
    return _eval(lambda e: e.sin, ctx, x)

def sinh(x: Real, ctx: Context = REAL) -> Float | Fraction:
    """Computes the hyperbolic sine of `x` rounded under `ctx`."""
    return _eval(lambda e: e.sinh, ctx, x)

def sqrt(x: Real, ctx: Context = REAL) -> Float | Fraction:
    """Computes square-root of `x` rounded under `ctx`."""
    return _eval(lambda e: e.sqrt, ctx, x)

def sub(x: Real, y: Real, ctx: Context = REAL) -> Float | Fraction:
    """Subtracts `y` from `x` rounded under `ctx`."""
    r, (xr, yr) = _dispatch(lambda e: e.sub, ctx, x, y)
    ny = -_rational(yr) if not (isinstance(yr, Float) and yr.is_nar()) else yr
    return _zero_sum(_normalize(r, ctx, (xr, yr)), ctx, (xr, ny),
                     _zero_sign(xr) + tuple(not s for s in _zero_sign(yr)))

def tan(x: Real, ctx: Context = REAL) -> Float | Fraction:
    """Computes the tangent of `x` rounded under `ctx`."""
    return _eval(lambda e: e.tan, ctx, x)

def tanh(x: Real, ctx: Context = REAL) -> Float | Fraction:
    """Computes the hyperbolic tangent of `x` rounded under `ctx`."""
    return _eval(lambda e: e.tanh, ctx, x)

def tgamma(x: Real, ctx: Context = REAL) -> Float | Fraction:
    """Computes gamma of `x` rounded under `ctx`."""
    return _eval(lambda e: e.tgamma, ctx, x)

#############################################################################
# Rounding operations

def _round(x: Real, ctx: Context | None, exact: bool) -> Float | Fraction:
    _check_ctx(ctx)
    if ctx is None or ctx is REAL:
        # real computation; no rounding
        if isinstance(x, Fraction):
            return x  # exact rational number
        else:
            return REAL.round(x)
    else:
        return ctx.round(x, exact=exact)

@overload
def round(x: float | Float, ctx: Context = REAL) -> Float: ...
@overload
def round(x: Real, ctx: Context = REAL) -> Float | Fraction: ...
def round(x: Real, ctx: Context = REAL) -> Float | Fraction:
    """
    Rounds `x` under the given context `ctx`.

    If `ctx` is `None`, this operation is the identity operation.
    """
    return _round(x, ctx, exact=False)

def round_exact(x: Real, ctx: Context = REAL) -> Float | Fraction:
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
    if ctx is None or ctx is REAL:
        raise ValueError(f'round_at() not supported for ctx={ctx}')
    else:
        return ctx.round_at(x, int(n))

def cast(x: Real, ctx: Context = REAL) -> Real:
    """
    Identity operation that checks if `x` is representable in the context `ctx`.
    
    Returns `x` if it can be represented in `ctx`, otherwise raises an error.
    This is useful for asserting that a value fits within a specific rounding context.
    """
    if ctx is None or ctx is REAL:
        # real computation; always representable
        return x
    else:
        # Check if x can be exactly represented in ctx
        rounded = ctx.round(x, exact=True)
        return rounded

#############################################################################
# Round-to-integer operations

def ceil(x: Real, ctx: Context = REAL) -> Float | Fraction:
    """
    Computes the smallest integer greater than or equal to `x`
    that is representable under `ctx`.

    If the context supports overflow, the result may be infinite.
    """
    return _round_integer(lambda e: e.ceil, ctx, x)

def floor(x: Real, ctx: Context = REAL) -> Float | Fraction:
    """
    Computes the largest integer less than or equal to `x`
    that is representable under `ctx`.

    If the context supports overflow, the result may be infinite.
    """
    return _round_integer(lambda e: e.floor, ctx, x)

def trunc(x: Real, ctx: Context = REAL) -> Float | Fraction:
    """
    Computes the integer with the largest magnitude whose
    magnitude is less than or equal to the magnitude of `x`
    that is representable under `ctx`.

    If the context supports overflow, the result may be infinite.
    """
    return _round_integer(lambda e: e.trunc, ctx, x)

def nearbyint(x: Real, ctx: Context = REAL) -> Float:
    """
    Rounds `x` to a representable integer according to
    the rounding mode of this context.

    If the context supports overflow, the result may be infinite.
    """
    _check_ctx(ctx)
    if ctx is None or ctx is REAL:
        raise RuntimeError('nearbyint() not supported in RealContext')
    else:
        return ctx.round_integer(x)

def roundint(x: Real, ctx: Context = REAL) -> Float | Fraction:
    """
    Rounds `x` to the nearest representable integer,
    rounding ties away from zero in halfway cases.

    If the context supports overflow, the result may be infinite.
    """
    return _round_integer(lambda e: e.roundint, ctx, x)

#############################################################################
# Classification

def isnan(x: Real, ctx: Context = REAL) -> bool:
    """Checks if `x` is NaN."""
    x = _cvt_to_real(x)
    return isinstance(x, Float) and x.isnan

def isinf(x: Real, ctx: Context = REAL) -> bool:
    """Checks if `x` is infinite."""
    x = _cvt_to_real(x)
    return isinstance(x, Float) and x.isinf

def isfinite(x: Real, ctx: Context = REAL) -> bool:
    """Checks if `x` is finite."""
    x = _cvt_to_real(x)
    return isinstance(x, Fraction) or not x.is_nar()

def isnormal(x: Real, ctx: Context = REAL) -> bool:
    """Checks if `x` is normal (not subnormal, zero, or NaN)."""
    # TODO: what if the argument is a Fraction?
    x = _cvt_to_real(x)
    return isinstance(x, Fraction) or x.is_normal()

#############################################################################
# Numerical data

def signbit(x: Real, ctx: Context = REAL) -> bool:
    """Checks if the sign bit of `x` is set (i.e., `x` is negative)."""
    # TODO: should all Floats have this property?
    if isinstance(x, Fraction):
        return x < 0
    else:
        return _cvt_to_float(x).s

def logb(x: Real, ctx: Context = REAL) -> Float:
    """
    Extracts the (normalized) exponent of `x`, rounded under `ctx`.

    - If the argument is 0, returns -∞.
    - If the argument is +/-∞, returns +∞.
    - If the argument is NaN, returns NaN.
    - If the argument is `(-1) * (1 + f) * b^e`, returns `e`.

    For non-zero arguments, `logb(x) = floor(log_{b}(abs(x)))`, where
    `b` is the base of the floating-point representation
    """
    # TODO: compute `logb` for a non-dyadic fraction
    x = _cvt_to_float(x)
    if x.is_nonzero():
        # finite, non-zero => floor(log2(abs(x)))
        return ctx.round(RealFloat.from_int(x.e))
    elif x.isinf:
        # infinite => +inf
        return ctx.round(Float(isinf=True))
    elif x.isnan:
        # NaN => NaN
        return ctx.round(Float(isnan=True))
    else:
        # 0 => -inf
        return ctx.round(Float(s=True, isinf=True))


#############################################################################
# Tensor

def empty(*dims: Real, ctx: Context = REAL) -> list:
    """
    Initializes an empty nested list with dimensions specified by `dims`.

    Each argument specifies the size of a dimension from outer to inner.
    For example:
    - empty(5) creates a 1D list of size 5
    - empty(3, 4) creates a 3x4 2D list (3 outer lists, each with 4 elements)
    - empty(2, 3, 4) creates a 2x3x4 3D list
    """
    if not dims:
        raise ValueError('At least one dimension must be specified')

    # Convert all dimensions to integers and validate
    int_dims = []
    for d in dims:
        d_float = _cvt_to_float(d)
        if not d_float.is_integer() or d_float.is_negative():
            raise ValueError(f'Invalid list size: {d_float}')
        int_dims.append(int(d_float))

    return _empty(int_dims)

def dim(x: list, ctx: Context = REAL) -> Float:
    """
    Returns the number of dimensions of the tensor `x`.

    Assumes that `x` is not a ragged tensor.

    A shape query: the answer is an exact integer, so `ctx` is accepted for
    uniformity and ignored (as it is by the other queries and projections).
    """
    # `dim` is list-only in FPy (matches `TypeInfer`, which unifies `Dim`'s
    # argument with a list type); reject a non-list operand rather than
    # silently returning 0.
    if not isinstance(x, list):
        raise TypeError(f'dim() expects a list, got {x}')
    dim = 0
    while True:
        if isinstance(x, list):
            dim += 1
            if x == []:
                break
            x = x[0]
        else:
            break

    return Float.from_int(dim, checked=False)

def size(x: list, dim: Real, ctx: Context = REAL) -> Float:
    """
    Returns the size of the dimension `dim` of the tensor `x`.

    Assumes that `x` is not a ragged tensor.

    A shape query: the answer is an exact integer, so `ctx` is accepted for
    uniformity and ignored (as it is by the other queries and projections).
    """
    dim = _cvt_to_float(dim)
    # size(x, n) = len(x[0]...[0]), n indices deep
    for _ in range(int(dim)):
        x = x[0]
    return Float.from_int(len(x), checked=False)

#############################################################################
# Tuple

def fst[T](x: Sequence[T], ctx: Context = REAL) -> T:
    """
    Returns the first element `x` of a pair `(x, y)`.
    """
    if len(x) != 2:
        raise ValueError(f'fst requires a pair, got `{x}`')
    return x[0]

def snd[T](x: Sequence[T], ctx: Context = REAL) -> T:
    """
    Returns the second element `y` of a pair `(x, y)`.
    """
    if len(x) != 2:
        raise ValueError(f'snd requires a pair, got `{x}`')
    return x[1]

#############################################################################
# Constants

def digits(m: int, e: int, b: int, ctx: Context = REAL) -> Float:
    """
    Creates a `Float` of the form `m * b**e`, where `m` is the
    significand, `e` is the exponent, and `b` is the base.

    The result is rounded under the given context.
    """
    if ctx is None or ctx is REAL:
        # real computation; no rounding
        x = digits_to_fraction(m, e, b)
        if not is_dyadic(x):
            raise ValueError(f'cannot evaluate exactly: digits(m={m}, e={e}, b={b})')
        m = x.numerator
        exp = 1 - x.denominator.bit_length()
        return Float(m=m, exp=exp)
    else:
        x = digits_to_fraction(m, e, b)
        return ctx.round(x)

def hexfloat(s: str, ctx: Context = REAL) -> Float:
    """
    Creates a `Float` from a hexadecimal floating-point string `s`.
    The result is rounded under the given context.
    """
    if ctx is None or ctx is REAL:
        # real computation; no rounding
        x = hexnum_to_fraction(s)
        if not is_dyadic(x):
            raise ValueError(f'cannot evaluate exactly: hexfloat(s={s})')
        m = x.numerator
        exp = 1 - x.denominator.bit_length()
        return Float(m=m, exp=exp)
    else:
        x = hexnum_to_fraction(s)
        return ctx.round(x)

def rational(n: int, d: int, ctx: Context = REAL) -> Float:
    """
    Creates a `Float` from a fraction with the given numerator and denominator.
    The result is rounded under the given context.
    """
    if d == 0:
        return Float(isnan=True, ctx=ctx)

    if ctx is None or ctx is REAL:
        # real computation; no rounding
        x = Fraction(n, d)
        if not is_dyadic(x):
            raise ValueError(f'cannot evaluate exactly: rational(n={n}, d={d})')
        m = x.numerator
        exp = 1 - x.denominator.bit_length()
        return Float(m=m, exp=exp)
    else:
        x = Fraction(n, d)
        return ctx.round(x)

def nan(ctx: Context = REAL) -> Float:
    """NaN (Not a Number) rounded under `ctx`."""
    _check_ctx(ctx)
    return ctx.round(Float(isnan=True))

def inf(ctx: Context = REAL) -> Float:
    """Positive infinity rounded under `ctx`."""
    _check_ctx(ctx)
    return ctx.round(Float(isinf=True))

def const_pi(ctx: Context = REAL) -> Float:
    """
    Creates a `Float` representing π (pi).
    The result is rounded under the given context.
    """
    return _const(lambda e: e.const_pi, ctx)

def const_e(ctx: Context = REAL) -> Float:
    """
    Creates a `Float` representing e (Euler's number).
    The result is rounded under the given context.
    """
    return _const(lambda e: e.const_e, ctx)

def const_log2e(ctx: Context = REAL) -> Float:
    """
    Creates a `Float` representing log2(e) (the logarithm of e base 2).
    The result is rounded under the given context.
    """
    return _const(lambda e: e.const_log2e, ctx)

def const_log10e(ctx: Context = REAL) -> Float:
    """
    Creates a `Float` representing log10(e) (the logarithm of e base 10).
    The result is rounded under the given context.
    """
    return _const(lambda e: e.const_log10e, ctx)

def const_ln2(ctx: Context = REAL) -> Float:
    """
    Creates a `Float` representing ln(2) (the natural logarithm of 2).
    The result is rounded under the given context.
    """
    return _const(lambda e: e.const_ln2, ctx)

def const_pi_2(ctx: Context = REAL) -> Float:
    """
    Creates a `Float` representing π/2 (pi divided by 2).
    The result is rounded under the given context.
    """
    return _const(lambda e: e.const_pi_2, ctx)

def const_pi_4(ctx: Context = REAL) -> Float:
    """
    Creates a `Float` representing π/4 (pi divided by 4).
    The result is rounded under the given context.
    """
    return _const(lambda e: e.const_pi_4, ctx)

def const_1_pi(ctx: Context = REAL) -> Float:
    """
    Creates a `Float` representing 1/π (one divided by pi).
    The result is rounded under the given context.
    """
    return _const(lambda e: e.const_1_pi, ctx)

def const_2_pi(ctx: Context = REAL) -> Float:
    """
    Creates a `Float` representing 2/π (two divided by pi).
    The result is rounded under the given context.
    """
    return _const(lambda e: e.const_2_pi, ctx)

def const_2_sqrt_pi(ctx: Context = REAL) -> Float:
    """
    Creates a `Float` representing 2/sqrt(π) (two divided by the square root of pi).
    The result is rounded under the given context.
    """
    return _const(lambda e: e.const_2_sqrt_pi, ctx)

def const_sqrt2(ctx: Context = REAL) -> Float:
    """
    Creates a `Float` representing sqrt(2) (the square root of 2).
    The result is rounded under the given context.
    """
    return _const(lambda e: e.const_sqrt2, ctx)

def const_sqrt1_2(ctx: Context = REAL) -> Float:
    """
    Creates a `Float` representing sqrt(1/2) (the square root of 1/2).
    The result is rounded under the given context.
    """
    return _const(lambda e: e.const_sqrt1_2, ctx)
