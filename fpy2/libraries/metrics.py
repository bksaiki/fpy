"""
Common metrics:
- error
- condition numbers
"""

from fractions import Fraction

from . import base as fp

__all__ = [
    # relative error
    'absolute_error',
    'relative_error',
    'scaled_error',
    'ordinal_error',
    # condition numbers
    'sum_condition',
]


@fp.fpy
def absolute_error(x: fp.Real, y: fp.Real):
    """
    Computes the absolute error between `x` and `y`, i.e., `|x - y|`,
    rounding under the current context.
    """
    with fp.REAL:
        t = abs(x - y)
    return fp.round(t)

@fp.fpy
def scaled_error(x: fp.Real, y: fp.Real, s: fp.Real):
    """
    Computes the scaled error between `x` and `y`, scaled by `s`,
    i.e, `|x - y| / |s|`, rounding under the current context.

    When `s = y`, this is equivalent to `relative_error(x, y)`.
    """
    with fp.REAL:
        n = absolute_error(x, y)
        d = abs(s)
    return n / d

@fp.fpy
def relative_error(x: fp.Real, y: fp.Real):
    """
    Computes the relative error between `x` and `y`, i.e., `|x - y| / |y|`,
    rounding under the current context.
    """
    with fp.REAL:
        n = absolute_error(x, y)
        d = abs(y)
    return n / d

@fp.fpy_primitive
def ordinal_error(x: fp.Float, y: fp.Float, ctx: fp.Context) -> Fraction:
    """
    Computes the ordinal error between `x` and `y`, i.e., the number of
    floating-point numbers between `x` and `y`.

    This is equivalent to `|int(x) - int(y)|`, where `int` is the
    conversion from `Float` to `int`.
    """
    if not isinstance(x, fp.Float | Fraction):
        raise TypeError(f'Expected \'Float\', got `{x}`')
    if not isinstance(y, fp.Float | Fraction):
        raise TypeError(f'Expected \'Float\', got {y}')
    if not isinstance(ctx, fp.OrdinalContext):
        raise TypeError(f'Expected \'OrdinalContext\', got {ctx}')

    a = fp._cvt_to_float(x)
    b = fp._cvt_to_float(y)
    a_ord = ctx.to_fractional_ordinal(a)
    b_ord = ctx.to_fractional_ordinal(b)

    err = abs(a_ord - b_ord)
    return fp.Float.from_rational(err)

@fp.fpy
def sum_condition(xs: list[fp.Real]):
    """
    Computes the condition number of summation over `xs`, i.e.,
    `|sum(|x_i|)| / |sum(x_i)|`, rounding under the current context.
    """
    with fp.REAL:
        sum_abs = sum([abs(x) for x in xs])
        abs_sum = abs(sum(xs))
    return sum_abs / abs_sum
