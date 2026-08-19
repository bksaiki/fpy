"""
Scheduling language: floating-point to fixed-point rounding
"""

from ..function import Function
from ..transform import FloatToFixed
from ..transform.utils.cursor import Block, Cursor


def float_to_fixed(func: Function, where: int | Cursor | Block | None = None) -> Function:
    """
    Express floating-point rounding in `func` as fixed-point rounding.

    A float format rounds at a digit position that depends on the value, since
    its values thin out as the magnitude grows.  Once that position is
    computed, the rounding is a fixed-point one: for a format with precision
    ``P``, subnormal position ``EXP``, largest exponent ``EMAX``, and bound
    ``B``,

    ``round_F(x) = round_A(inf, n, B)(x)``, where
    ``n = clamp(logb(x) - P + 1, EXP, EMAX - P + 1)``.

    The rewrite computes that position under ``fp.REAL`` and rounds under a
    fixed-point context built at it.  The value is never scaled and the bound
    is the format's own.  Values below ``emin`` get their own branch, where the
    format is fixed-point already and the context is a constant; NaN,
    infinities, and zeros get another, since ``logb`` is undefined for them —
    each of those is a constant, so the branch assigns what the format makes
    of it.

    Applies to a float format that rounds deterministically and whose overflow
    a fixed-point round can reproduce; other contexts are left unchanged.  Only
    blocks whose body is entirely ``x = fp.round(v)`` (or a returned round) are
    rewritten.

    Run :func:`fpy2.strategies.rescale_fixed` afterwards to shift the resulting
    fixed-point rounding to digit position zero, where its values are integers.

    Parameters
    ----------
    func : Function
        The function to transform.
    where : int | Cursor | Block | None
        Which block to lower: an index counting candidate blocks (the
        structurally-matching rounding blocks, whether or not they verify)
        in visit order, outermost-first; or a
        :class:`fpy2.strategies.Cursor` / :class:`fpy2.strategies.Block`
        naming a program point, which takes every candidate *at or
        beneath* it -- so the statement an earlier rewrite left behind
        names the rounding now nested inside it. A cursor or region from
        an earlier program is forwarded to this one first. If `None`,
        lower every candidate that verifies and skip the rest.

    Returns
    -------
    Function
        The transformed function.

    Raises
    ------
    TransformDeclined
        If an explicit `where` names a candidate this rewrite refuses, or a
        region whose every candidate it refuses; the message says why.
    TransformReferenceError
        If an explicit `where` names no candidate block, or a cursor of a
        program this one was not derived from.

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
            with fp.REAL:
                if fp.isnan(x):
                    y = fp.nan()
                elif fp.isinf(x):
                    y = (-fp.inf() if fp.signbit(x) else fp.inf())
                elif x == 0:
                    y = (-0.0 if fp.signbit(x) else 0)
                else:
                    e = fp.logb(x)
                    if e < -14:
                        with fp.MPBFixedContext(-25, 65504, overflow=fp.OverflowMode.OVERFLOW, enable_inf=True):
                            y = fp.round(x)
                    else:
                        exp = min(max((e - 10), -24), 5)
                        with fp.MPBFixedContext((exp - 1), 65504, overflow=fp.OverflowMode.OVERFLOW, enable_inf=True):
                            y = fp.round(x)
            return y
    """
    if not isinstance(func, Function):
        raise TypeError(f"Expected a \'Function\', got {func}")

    return func.with_edits(FloatToFixed.apply_with_edits(
        func.ast, where=func.rebase(where)
    ))
