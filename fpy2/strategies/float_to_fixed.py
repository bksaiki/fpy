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

    The rewrite computes that position under ``fp.REAL`` and rounds under a
    fixed-point context built at it.  The value is never scaled and the bound
    is the format's own.  Values below ``emin`` get their own branch, where the
    format is fixed-point already and the context is a constant; NaN,
    infinities, and zeros get another, since ``logb`` is undefined for them.

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
            if fp.isnan(x) or fp.isinf(x) or x == 0:
                with fp.MPBFixedContext(-25, 65504, overflow=..., enable_nan=True, enable_inf=True):
                    y = fp.round(x)
            else:
                with fp.REAL:
                    e = fp.logb(x)
                if e < -14:
                    with fp.MPBFixedContext(-25, 65504, overflow=..., enable_nan=True, enable_inf=True):
                        y = fp.round(x)
                else:
                    with fp.REAL:
                        exp = min((e - 10), 5)
                    with fp.MPBFixedContext((exp - 1), 65504, overflow=..., enable_nan=True, enable_inf=True):
                        y = fp.round(x)
            return y
    """
    if not isinstance(func, Function):
        raise TypeError(f"Expected a \'Function\', got {func}")

    ast = FloatToFixed.apply(func.ast)
    return func.with_ast(ast)
