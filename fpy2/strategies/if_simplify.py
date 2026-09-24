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
    becomes an `if` expression, with each merged variable made explicit.

    An arm whose statements are all plain assignments is reduced to one
    expression per name and placed *inside* the `IfExpr`, which is lazy, so it
    keeps its guard.  An arm that cannot be reduced -- a loop, a list write, a
    nested `if` left unrewritten by ``where`` -- is hoisted into the enclosing
    block and runs unconditionally.

    Refusals are judged on the arm as written, before it is known to inline, so
    an arm that would have inlined can still be declined.  A construct that
    could change which value the function produces, or that aborts where the
    program asked to, is declined under every mode, raising
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

    ``strict`` also declines what cannot be *shown* to be preserved: an
    out-of-range subscript, an operation under an unresolved context, and one
    whose context cannot hold an infinity or NaN it might produce
    (`fp.logb(0)` under `fp.INTEGER`).  The default admits all three, so
    ``strict`` is conservative in an unannotated function and most useful
    after :func:`monomorphize`, which makes contexts concrete.

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
            y = ((x * 2) if cond else -x)
            return y
    """
    if not isinstance(func, Function):
        raise TypeError(f"Expected a \'Function\', got {func}")

    return func.with_edits(
        SimplifyIf.apply_with_edits(func.ast, func.rebase(where), strict=strict)
    )
