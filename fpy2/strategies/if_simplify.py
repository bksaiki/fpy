"""
Scheduling language: simplify_if
"""

from ..function import Function
from ..transform import Cursor, SimplifyIf


def simplify_if(
    func: Function,
    where: int | Cursor | None = None,
    *,
    strict: bool = False,
) -> Function:
    """:class:`fpy2.transform.SimplifyIf` over *func*: every `if` statement
    becomes an `if` expression, with both branch bodies hoisted and each merged
    variable made explicit.

    Hoisting makes a branch body unconditional, so a construct that could
    change whether -- or which -- value the function produces is declined under
    every mode, raising :class:`~fpy2.transform.TransformDeclined`: `return`,
    `assert`, an effect, a list write, `while`, `for`, `fp.cast`, a call to
    another FPy function, any operation under an `ASSERT` overflow context,
    and, under a context that cannot hold an infinity or NaN, an operation
    that could produce one -- a guard excluding the bad input is load-bearing
    there.  That is either an operation with a pole at a finite operand
    (`fp.logb(0)`, `fp.sqrt(-1)`, `fp.acos(2)`), or, where the format is
    bounded and rounds an overflow to infinity, any operation at all.  The
    `ASSERT` case is keyed on whether an operation consults the rounding
    context, not on its node class, so `x * x` overflows there exactly as
    `fp.round(x)` would.

    ``where`` names one site: an index counting `if` statements in visit
    order, or a cursor or region, which takes the sites at or beneath it.
    ``None`` rewrites every one.  A nested `if` left behind is sound: it
    becomes unconditional, but a branch body is effect-free by this pass's own
    refusals, so the value it computes is discarded by the enclosing
    `IfExpr`.

    ``strict`` governs what is left: operations whose value is preserved but
    whose observable effects cannot be shown to be.  The default hoists them --
    an out-of-range subscript is behavior FPy already leaves undefined, and a
    function with no ``ctx=`` inherits its caller's context, so no operation in
    it can be shown not to overflow.  ``strict=True`` declines them.  It is
    therefore conservative in an unannotated function and most useful after
    :func:`monomorphize`, which makes contexts concrete.

    Cursors forward across this pass.  A rewritten `if` forwards to the region
    that replaced it; a cursor naming a statement *inside* a branch does not,
    since that subtree was rebuilt and renamed.

    Examples
    --------
    ::

        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            if x > 0:
                y = x * 2
            else:
                y = -x
            return y

    ``simplify_if(f)`` yields::

        @fp.fpy
        def f(x):
            cond = x > 0
            y2 = (x * 2)
            y3 = -x
            y = (y2 if cond else y3)
            return y
    """
    if not isinstance(func, Function):
        raise TypeError(f"Expected a \'Function\', got {func}")

    return func.with_edits(
        SimplifyIf.apply_with_edits(func.ast, func.rebase(where), strict=strict)
    )
