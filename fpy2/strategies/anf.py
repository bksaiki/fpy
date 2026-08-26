"""
Scheduling language: administrative normal form
"""

from ..function import Function
from ..transform import ANF


def to_anf(func: Function) -> Function:
    """
    Rewrite `func` into administrative normal form: every proper subexpression
    of a statement becomes a name, a literal or a nullary constant.

    Takes no `where`: normal form is not a per-site decision, so this rewrites
    the whole function.

    **Why a schedule wants it.**  A rewrite needing to emit a statement has
    nowhere to put one inside an expression, so it declines to enter a
    conditionally- or repeatedly-evaluated position:
    :func:`fpy2.strategies.elim_round` and :func:`fpy2.strategies.insert_round`
    both leave a ternary arm alone.  Lowering it to an ``if``/``else`` makes the
    code in it reachable; :func:`fpy2.strategies.comp_to_loop` is the same move
    for a comprehension.

    A ``while`` condition is rotated instead -- evaluated once before the loop
    and once at the end of the body -- and only where it holds something needing
    a statement, since that duplicates it.

    **What it leaves.**  Only scalars are bound: a name holding a list is a
    second *place*, which decides whether the C++ backend can drop the list's
    shared handle.  So a chain of subscripts is named at its outermost scalar
    and the aggregate spine stays inline.  A comprehension's element and
    iterables keep whatever nesting they had; nothing raises, and
    :meth:`fpy2.transform.ANF.refusals` reports each with the reason.

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
