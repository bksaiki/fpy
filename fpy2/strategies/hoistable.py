"""
Scheduling language: hoistable form
"""

from ..function import Function
from ..transform import Hoistable


def to_hoistable(func: Function) -> Function:
    """
    Rewrite `func` so that a statement may be inserted above any expression:
    every expression node ends up evaluated exactly once, unconditionally,
    whenever its enclosing statement is reached.

    Takes no `where`: normal form is not a per-site decision, so this rewrites
    the whole function.

    **Why a schedule wants it.**  A rewrite needing to emit a statement has
    nowhere to put one inside an expression, so it declines to enter a
    conditionally- or repeatedly-evaluated position:
    :func:`fpy2.strategies.elim_round` and :func:`fpy2.strategies.insert_round`
    both leave a ternary arm alone.  This pass gives each such position a
    statement slot -- a ternary becomes an ``if``/``else``, an ``and``/``or``
    tail becomes guarded statements, and a ``while`` loop is rotated so its
    condition is evaluated once before the loop and once at the end of the body.
    :func:`fpy2.strategies.comp_to_loop` is the same move for a comprehension,
    and must run *first*: the loop body it generates is the slot a
    comprehension's element lacked.

    **Weaker than** :func:`fpy2.strategies.to_anf`, which requires this and
    additionally binds every nameable subexpression to a name -- what a backend
    needs and a rewrite does not.  This names an expression only where a lowering
    to its right would otherwise be hoisted above it and change the order of
    evaluation.

    **What it leaves.**  A comprehension, which nothing raises about;
    :meth:`fpy2.transform.Hoistable.refusals` reports each one, and is empty
    exactly when the whole function is hoistable.

    One thing to know: a rotated condition exists in *two* places, so a later
    rewrite aimed at one copy must be aimed at the other too.

    Cursors do not forward across this pass.

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
    ::

        @fp.fpy
        def f(a: fp.Real, b: fp.Real, c: bool) -> fp.Real:
            with fp.FP64:
                y = (a * b) + (a - b)
                return y if c else fp.sqrt(y)

    ``to_hoistable(f)`` yields::

        @fp.fpy
        def f(a, b, c):
            with fp.FP64:
                y = ((a * b) + (a - b))
                if c:
                    t = y
                else:
                    t = fp.sqrt(y)
                return t

    The ternary is lowered because its second arm had nowhere to put a
    statement; ``y``'s right-hand side is left exactly as written, where
    :func:`fpy2.strategies.to_anf` would name both products.
    """
    if not isinstance(func, Function):
        raise TypeError(f"Expected a \'Function\', got {func}")

    return func.with_ast(Hoistable.apply(func.ast))
