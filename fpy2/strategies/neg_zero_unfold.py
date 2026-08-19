"""
Scheduling language: the sign of zero as program text
"""

from ..function import Function
from ..transform import UnfoldNegZero


def unfold_neg_zero(func: Function, where: int | None = None) -> Function:
    """
    Take the signed zero out of `func`'s rounding contexts and state it as
    program text.

    A fixed-point format with ``enable_neg_zero`` keeps the sign of a value
    that rounds to zero: rounding ``-1e-30`` gives ``-0.0``, not ``+0.0``.
    Stated as program text, that rule is the same rounding without the flag,
    plus a sign restoration from the operand::

        with C_:                  # C with enable_neg_zero=False
            t = fp.round(x)
        if t == 0:
            r = fp.copysign(t, x)
        else:
            r = t

    C++ has no integer type with a signed zero, so this one flag decides
    whether a rounding reaches integer storage; anything targeting integer
    arithmetic needs the sign out of the format.

    Applies to :class:`fpy2.MPFixedContext` and :class:`fpy2.MPBFixedContext`
    roundings whose format keeps a signed zero and whose behavior the emitted
    program reproduces exactly; a format it cannot reproduce is left
    unchanged.  Wrapping overflow is the common decliner — it wraps by ordinal
    over the full signed range, so a negative operand can land on ``+0.0``,
    which no sign restoration from the operand gives back.
    :class:`fpy2.SMFixedContext` has its signed zero by construction, so it is
    rebuilt as the :class:`fpy2.MPBFixedContext` it derives from;
    :class:`fpy2.FixedContext` (two's complement) already has a single zero
    and is never a candidate.  Only blocks whose body is entirely
    ``x = fp.round(v)`` (or a returned round) are rewritten.

    Run :func:`fpy2.strategies.unfold_overflow` afterwards: with the sign rule
    out of the context, the bound is the only edge rule left to state.

    Parameters
    ----------
    func : Function
        The function to transform.
    where : int | None
        The index of the block to rewrite, counting candidate blocks (the
        structurally-matching rounding blocks, whether or not they verify)
        in visit order, outermost-first. If `None`, rewrite every candidate
        that verifies and skip the rest.

    Returns
    -------
    Function
        The transformed function.

    Raises
    ------
    TransformDeclined
        If an explicit `where` names a candidate this rewrite refuses;
        the message says why.
    TransformReferenceError
        If an explicit `where` names no candidate block.

    Examples
    --------
    A quantizer that rounds into a fixed-point format with a signed zero::

        @fp.fpy(ctx=fp.REAL)
        def quantize(x):
            with fp.MPFixedContext(-8):
                y = fp.round(x)
            return y

    ``unfold_neg_zero(quantize)`` rounds without the signed zero and restores
    the sign::

        @fp.fpy(
            ctx=fp.REAL,
        )
        def quantize(x):
            with fp.REAL:
                with fp.MPFixedContext(-8, enable_neg_zero=False):
                    t = fp.round(x)
                if t == 0:
                    y = fp.copysign(t, x)
                else:
                    y = t
            return y
    """
    if not isinstance(func, Function):
        raise TypeError(f"Expected a \'Function\', got {func}")

    ast = UnfoldNegZero.apply(func.ast, where=where)
    return func.with_ast(ast)
