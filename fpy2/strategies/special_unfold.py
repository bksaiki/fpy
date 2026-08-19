"""
Scheduling language: special values as program text
"""

from ..function import Function
from ..transform import UnfoldSpecial


def unfold_special(func: Function, where: int | None = None) -> Function:
    """
    Take the special values out of `func`'s rounding contexts and state them
    as program text.

    A format's answer for NaN, an infinity and a zero is a constant, since a
    special operand is the only way a special reaches the rounding.  So each
    becomes a branch on the operand::

        if fp.isnan(x):
            r = v                 # what the format made of NaN
        elif fp.isinf(x):
            r = w
        elif x == 0:
            r = -0.0 if fp.signbit(x) else 0
        else:
            with C_:              # C, less any rule the branches took over
                r = fp.round(x)

    The surviving rounding is then left an operand that is finite *and*
    non-zero — structure a value-class analysis reads to discharge the guards
    below it, and a format free of NaN is one an integer type can store.

    Stating a special and *shedding* its rule from the format are separate.
    Stating one needs only a statically-known context, since the branch
    assigns what the rounding would have returned.  Shedding needs a format
    that states the rule as a parameter (``enable_nan``/``enable_inf``,
    ``nan_value``/``inf_value``) *and* an agreement check against the source,
    so a format whose overflow *produces* an infinity keeps that rule while
    its branch is still emitted, and a float format is stated but never shed.
    A refusal is neither: a branch cannot refuse a value, and leaving it to
    the rounding refuses it identically — so a format that refuses both
    specials is left unchanged, as is ``REAL``, which rounds exactly.

    Which branches appear is decided per operand, so a class the operand
    cannot hold gets none — which also makes the rewrite idempotent.  Only
    blocks whose body is entirely ``x = fp.round(v)`` or ``x = fp.cast(v)``
    (or a returned round) are rewritten; a cast substitutes a special exactly
    as a round does, and a stochastic rounding takes the branches too, since
    a special never reaches the random draw.

    Run :func:`fpy2.strategies.rescale_fixed` afterwards: a substituted
    constant does not commute with scaling, so the rescale declines a format
    whose ``nan_value``/``inf_value`` is finite — unless this pass has
    already taken those out of the context.

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
    A quantizer that rounds into a fixed-point format that represents NaN
    and the infinities::

        @fp.fpy(ctx=fp.REAL)
        def quantize(x):
            with fp.MPFixedContext(-8, enable_nan=True, enable_inf=True):
                y = fp.round(x)
            return y

    ``unfold_special(quantize)`` states each rule as a branch::

        @fp.fpy(
            ctx=fp.REAL,
        )
        def quantize(x):
            with fp.REAL:
                if fp.isnan(x):
                    y = (-fp.nan() if fp.signbit(x) else fp.nan())
                elif fp.isinf(x):
                    y = (-fp.inf() if fp.signbit(x) else fp.inf())
                elif x == 0:
                    y = (-0.0 if fp.signbit(x) else 0)
                else:
                    with fp.MPFixedContext(-8):
                        y = fp.round(x)
            return y
    """
    if not isinstance(func, Function):
        raise TypeError(f"Expected a \'Function\', got {func}")

    return func.with_edits(UnfoldSpecial.apply_with_edits(func.ast, where=where))
