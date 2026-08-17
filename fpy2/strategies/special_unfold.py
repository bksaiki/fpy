"""
Scheduling language: special values as program text
"""

from ..function import Function
from ..transform import UnfoldSpecial


def unfold_special(func: Function, where: int | None = None) -> Function:
    """
    Take the special values out of `func`'s rounding contexts and state them
    as program text.

    A fixed-point format states what NaN and the infinities become — a
    representable value of their own (``enable_nan``/``enable_inf``), a
    substituted constant (``nan_value``/``inf_value``), or a refusal.  Each
    rule that names a value becomes a branch on the operand, since a special
    operand is the only way a special reaches the rounding::

        if fp.isnan(x):
            r = v                 # what the format made of NaN
        elif fp.isinf(x):
            r = w
        elif x == 0:
            r = -0.0 if fp.signbit(x) else 0
        else:
            with C_:              # C with the stated rules removed
                r = fp.round(x)

    The zero branch removes nothing from the format, but with it the
    surviving rounding's operand is finite *and* non-zero — structure a
    value-class analysis can use to discharge the format's remaining guards,
    and a format free of NaN is one an integer type can store.

    The two sides come out independently: a format whose overflow *produces*
    an infinity keeps that side (finite operands past the bound land there,
    which the branches never see) while its NaN rule still comes out.  A
    refusal also stays — a branch can only assign a value, not refuse one —
    so a format with no stated special of its own is left unchanged, as is a
    float format, which cannot shed its specials.  Only blocks whose body is
    entirely ``x = fp.round(v)`` (or a returned round) are rewritten.

    Run :func:`fpy2.strategies.rescale_fixed` afterwards: a substituted
    constant does not commute with scaling, so the rescale declines a format
    whose ``nan_value``/``inf_value`` is finite — unless this pass has
    already taken those out of the context.

    Parameters
    ----------
    func : Function
        The function to transform.
    where : int | None
        The index of the block to rewrite, counting candidate blocks (those
        this rewrite could apply to) in visit order, outermost-first. If
        `None`, rewrite every candidate.

    Returns
    -------
    Function
        The transformed function.

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

    ast = UnfoldSpecial.apply(func.ast, where=where)
    return func.with_ast(ast)
