"""
Scheduling language: simplify_if
"""

from ..function import Function
from ..transform import SimplifyIf


def simplify_if(func: Function, *, strict: bool = False) -> Function:
    """:class:`fpy2.transform.SimplifyIf` over *func*: every `if` statement
    becomes an `if` expression, with both branch bodies hoisted and each merged
    variable made explicit.

    Hoisting makes a branch body unconditional, so a construct that could
    change whether -- or which -- value the function produces is declined under
    every mode, raising :class:`~fpy2.transform.TransformDeclined`: `return`,
    `assert`, an effect, a list write, `while`, `for`, `fp.cast`, and a
    rounding under an `ASSERT` overflow context.

    ``strict`` governs what is left: operations whose value is preserved but
    whose observable effects cannot be shown to be.  The default hoists them --
    an out-of-range subscript is behavior FPy already leaves undefined, and a
    rounding under an unresolved context cannot be shown to overflow.
    ``strict=True`` declines them, making the rewrite observationally
    equivalent; it is most useful after :func:`monomorphize`, which makes
    contexts concrete.

    Cursors do not forward across this pass: it rewrites at sites it does
    not report.

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

    return func.with_ast(SimplifyIf.apply(func.ast, strict=strict))
