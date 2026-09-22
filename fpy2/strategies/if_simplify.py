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

    An arm whose statements are all plain assignments is reduced to one
    expression per name and placed *inside* the `IfExpr`, which is lazy, so it
    keeps its guard.  Only an arm that cannot be reduced -- a loop, a list
    write, a nested `if` left unrewritten by ``where`` -- is hoisted, and only
    a hoisted arm can be refused.

    Hoisting makes a branch body unconditional, so a construct that could
    change which value the function produces, or that aborts where the program
    asked to, is declined under every mode, raising
    :class:`~fpy2.transform.TransformDeclined`: `return`, `assert`, an effect,
    a list write, `while`, `for`, `fp.cast`, a call to another FPy function,
    and any operation under an `ASSERT` overflow context.  That last is keyed
    on whether an operation consults the rounding context, not on its node
    class, so `x * x` overflows there exactly as `fp.round(x)` would.

    ``where`` names one site: an index counting `if` statements in visit
    order, or a cursor or region, which takes the sites at or beneath it.
    ``None`` rewrites every one.  A nested `if` left behind is sound: it
    becomes unconditional, but a branch body is effect-free by this pass's own
    refusals, so the value it computes is discarded by the enclosing
    `IfExpr`.

    ``strict`` governs what is left: an observable effect that cannot be shown
    to be preserved, and a trap the program did not ask for.  The default
    hoists them -- an out-of-range subscript is behavior FPy already leaves
    undefined; a function with no ``ctx=`` inherits its caller's context, so no
    operation in it can be shown not to overflow; and a context that cannot
    hold an infinity raises where a wider one would return, which is the
    format's limit rather than a requested abort.  That last covers an
    operation with a pole at a finite operand (`fp.logb(0)`, `fp.sqrt(-1)`,
    `fp.acos(2)`) and, where the format is bounded and rounds an overflow to
    infinity, any operation at all.  ``strict=True`` declines them.  It is
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
