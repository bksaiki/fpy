"""
Scheduling language: overflow as program text
"""

from ..function import Function
from ..transform import UnfoldOverflow


def unfold_overflow(
    func: Function, where: int | None = None, *, early_check: bool = False
) -> Function:
    """
    Take the bound out of `func`'s rounding contexts and state it as program text.

    A bounded format decides two things at once: where its grid lies, and what
    becomes of a value too large for it.  IEEE 754 defines the second in terms
    of the first — round with an unbounded exponent range, then see whether the
    result fits — so for a bounded context ``C`` with unbounded counterpart
    ``U``, and a finite ``x``,

    ``round_C(x) = overflow(sign x)`` when ``|round_U(x)| > maxval``, and
    ``round_U(x)`` otherwise.

    The rewrite rounds under ``U`` and turns the bound into a comparison.  What
    overflow produces, and what the format makes of NaN and the infinities, are
    asked of the source context rather than assumed; a special value gets a
    branch only where the rewrite would otherwise disagree with it.

    A bounded *fixed-point* format unfolds the same way, its counterpart
    being :class:`fpy2.MPFixedContext` at the same digit position.

    Applies to a bounded format that rounds deterministically and whose
    overflow is a constant of its own: wrapping gives a different answer at
    every magnitude, and an unsigned format states no bound below zero, so
    neither is rewritten.  Other contexts are left unchanged.  Only blocks whose
    body is entirely ``x = fp.round(v)`` (or a returned round) are rewritten.

    Run :func:`fpy2.strategies.float_to_fixed` afterwards: with no bound left in
    the context, it lowers the rounding through its unbounded path.

    Parameters
    ----------
    func : Function
        The function to transform.
    where : int | None
        The index of the block to rewrite, counting candidate blocks (those
        this rewrite could apply to) in visit order, outermost-first. If
        `None`, rewrite every candidate.
    early_check : bool
        Also test the operand before rounding it, so nothing certain to
        overflow is rounded at all. The threshold is the format's ``infval``,
        the next value above ``maxval`` — not ``maxval`` itself, which a value
        may exceed and still round back onto the grid. This test is sound but
        not complete, so the one after the rounding stays either way.

    Returns
    -------
    Function
        The transformed function.

    Examples
    --------
    A quantizer that rounds into a bounded float format::

        @fp.fpy(ctx=fp.REAL)
        def quantize(x):
            with fp.FP16:
                y = fp.round(x)
            return y

    ``unfold_overflow(quantize)`` rounds under the unbounded format and
    compares::

        @fp.fpy(
            ctx=fp.REAL,
        )
        def quantize(x):
            with fp.REAL:
                with fp.MPSFloatContext(11, -14):
                    t = fp.round(x)
                if t > 65504:
                    y = fp.inf()
                elif t < -65504:
                    y = -fp.inf()
                else:
                    y = t
            return y

    With ``early_check=True``, a check on ``x`` precedes all of that::

        if x >= 65536:
            y = fp.inf()
        elif x <= -65536:
            y = -fp.inf()
        else:
            ...
    """
    if not isinstance(func, Function):
        raise TypeError(f"Expected a \'Function\', got {func}")

    ast = UnfoldOverflow.apply(func.ast, where=where, early_check=early_check)
    return func.with_ast(ast)
