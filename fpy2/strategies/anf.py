"""
Scheduling language: administrative normal form
"""

from ..function import Function
from ..transform import ANF


def to_anf(func: Function) -> Function:
    """
    Rewrite `func` into administrative normal form: every proper subexpression
    of a statement becomes a name, a literal or a nullary constant.

    Takes no `where`.  Every other strategy aims at one site and leaves the rest
    alone, which is what makes a schedule a sequence of decisions; normal form is
    not a decision of that kind -- a program flattened in one nest and not
    another is in no state a consumer wants -- so this rewrites the whole
    function.

    **Why a schedule wants this.**  A rewrite that needs to emit a statement has
    nowhere to put one inside an expression, so it declines to enter a
    conditionally- or repeatedly-evaluated position:
    :func:`fpy2.strategies.elim_round` and :func:`fpy2.strategies.insert_round`
    both leave a ternary arm, a short-circuited operand and a comprehension's
    element alone.  This pass lowers the first two into statements -- a ternary
    into an ``if``/``else`` over one name, a bool chain into guarded assignments
    that keep its short circuit -- so the code in them becomes reachable.
    :func:`fpy2.strategies.comp_to_loop` is the same move for the third.

    A ``while`` condition is rotated instead, evaluated once before the loop and
    once at the end of the body, which is the loop's own order.  That duplicates
    the condition, so it is done only where the condition holds something that
    needs a statement at all; a condition of names, literals and arithmetic is
    left as it is.

    **What it leaves.**  Only scalars are bound: naming a list or a tuple would
    give it a place of its own, and a second place is what decides whether the
    C++ backend can drop a list's shared handle.  So a chain of subscripts is
    named at its outermost *scalar* and the aggregate spine stays inline.  A
    position with no statement slot at all -- a comprehension's element and its
    iterables -- keeps whatever nesting it had; nothing raises, and
    :meth:`fpy2.transform.ANF.refusals` reports each one with the reason.

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
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                return fp.sqrt(x * x) + 1.0

    ``to_anf(f)`` yields::

        @fp.fpy
        def f(x):
            with fp.FP64:
                t = (x * x)
                t2 = fp.sqrt(t)
                return (t2 + 1)
    """
    if not isinstance(func, Function):
        raise TypeError(f"Expected a \'Function\', got {func}")

    return func.with_ast(ANF.apply(func.ast))
