"""
Common error metrics.
"""

from .number import Context, Float, REAL
from .ops import fabs, sub, div

__all__ = [
    'absolute_error',
    'relative_error',
    'scaled_error'
]

def absolute_error(x: Float, y: Float, ctx: Context = REAL):
    """
    Computes the absolute error between `x` and `y`, i.e., `|x - y|`,
    rounded under the context `ctx`.
    """
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got `{x}`')
    if not isinstance(y, Float):
        raise TypeError(f'Expected \'Float\', got {y}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got {ctx}')

    return ctx.round(fabs(sub(x, y)))

def scaled_error(x: Float, y: Float, s: Float, ctx: Context):
    """
    Computes the scaled error between `x` and `y`, scaled by `s`,
    i.e, `|x - y| / |s|`, rounded under the context `ctx`.

    When `s = y`, this is equivalent to `relative_error(x, y, ctx)`.
    """
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got `{x}`')
    if not isinstance(y, Float):
        raise TypeError(f'Expected \'Float\', got {y}')
    if not isinstance(s, Float):
        raise TypeError(f'Expected \'Float\', got {s}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got {ctx}')

    return div(absolute_error(x, y), fabs(s), ctx)

def relative_error(x: Float, y: Float, ctx: Context):
    """
    Computes the relative error between `x` and `y`, i.e., `|x - y| / |y|`,
    rounded under the context `ctx`.
    """
    if not isinstance(x, Float):
        raise TypeError(f'Expected \'Float\', got `{x}`')
    if not isinstance(y, Float):
        raise TypeError(f'Expected \'Float\', got {y}')
    if not isinstance(ctx, Context):
        raise TypeError(f'Expected \'Context\', got {ctx}')
    
    return div(absolute_error(x, y), fabs(y), ctx)
