"""
Scheduling language: floating-point to fixed-point rounding
"""

from ..function import Function
from ..transform import FloatToFixed


def float_to_fixed(func: Function) -> Function:
    """
    Express floating-point rounding in `func` as fixed-point rounding.

    A float format rounds at a digit position that depends on the value, since
    its grid coarsens with magnitude.  Once that position is computed, the
    rounding is a fixed-point one: for a format with precision ``P``, subnormal
    position ``EXP``, largest exponent ``EMAX``, and bound ``B``,

    ``round_F(x) = round_A(inf, n, B)(x)``, where
    ``n = clamp(logb(x) - P + 1, EXP, EMAX - P + 1)``.

    The rewrite computes that position under ``fp.INTEGER`` and turns the round
    into an ``fp.round_at`` under a fixed-point context on the format's finest
    grid.  The value is never scaled and the bound is the format's own.  NaN,
    infinities, and zeros take their own branches, since ``logb`` is undefined
    for them.

    Applies to an :class:`fpy2.IEEEContext` that overflows to infinity and
    rounds deterministically; other contexts are left unchanged.  Only blocks
    whose body is entirely ``x = fp.round(v)`` (or a returned round) are
    rewritten.

    Run :func:`fpy2.strategies.rescale_fixed` afterwards to shift the resulting
    fixed-point rounding to digit position zero, where its values are integers.

    Parameters
    ----------
    func : Function
        The function to transform.

    Returns
    -------
    Function
        The transformed function.

    Examples
    --------
    A quantizer that rounds into a float format::

        @fp.fpy(ctx=fp.REAL)
        def quantize(x):
            with fp.FP16:
                y = fp.round(x)
            return y

    ``float_to_fixed(quantize)`` replaces the float rounding with a
    fixed-point one at a computed position::

        @fp.fpy(
            ctx=fp.REAL,
        )
        def quantize(x):
            if fp.isnan(x):
                y = fp.nan()
            elif (fp.isinf(x) or x == 0):
                y = x
            else:
                with fp.INTEGER:
                    _n = (min(max((fp.logb(x) - 10), -24), 5) - 1)
                with MPBFixedContext(nmin=-25, ...):
                    y = fp.round_at(x, _n)
            return y
    """
    if not isinstance(func, Function):
        raise TypeError(f"Expected a \'Function\', got {func}")

    ast = FloatToFixed.apply(func.ast)
    return func.with_ast(ast)
