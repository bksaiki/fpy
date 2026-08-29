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

    **Requires hoistable form.**  A name has to go in a statement slot running
    exactly as often, and under exactly the same condition, as the expression it
    replaces.  A ternary arm, an ``and``/``or`` tail and a ``while`` condition
    have no such slot, so where one of those holds something needing a statement
    this raises :class:`~fpy2.strategies.TransformError` rather than return a
    program that looks normalized and is not.
    :func:`fpy2.strategies.to_hoistable` is what gives those positions a slot;
    run it first.  The gate is narrow -- it asks what *this* pass would have to
    name -- so a ternary over pure arithmetic is accepted and left nested, where
    ``to_hoistable`` would flatten it.

    **What it leaves.**  Only scalars are bound: a name holding a list is a
    second *place*, which decides whether the C++ backend can drop the list's
    shared handle.  So a chain of subscripts is named at its outermost scalar
    and the aggregate spine stays inline.  A comprehension's element and
    iterables keep whatever nesting they had -- that is not a precondition
    failure, and :meth:`fpy2.transform.ANF.refusals` reports each with the
    reason.

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
