"""
Scheduling language: single exit
"""

from ..function import Function
from ..transform import SingleExit


def single_exit(func: Function) -> Function:
    """
    Rewrite `func` to end in exactly one `return`.

    An early return becomes an assignment to one result name, and the
    statements that would have followed move into the branch that falls
    through.  Nothing is duplicated: an arm that returns cannot reach what
    follows it.

    Takes no `where`: a function has one exit structure, not one per return, so
    this is not a per-site decision.

    **Why a schedule wants it.**  Three consumers cannot express an early exit.
    :class:`fpy2.backend.FPCoreCompiler` rejects multiple returns outright;
    :func:`fpy2.strategies.inline` refuses a callee with more than one, so a
    caller cannot be flattened; and :func:`fpy2.strategies.simplify_if` refuses
    a `return` inside a branch, so the function cannot be put in expression
    form.  Running this first removes the shape all three decline.

    **What it refuses.**  A `return` inside a loop -- rewriting one needs a flag
    suppressing the rest of the body and every later iteration, since FPy has no
    `break`.  Such a loop unrolls away where its trip count is known
    (:func:`fpy2.strategies.unroll_for`), which is the shape that occurs in
    practice.  A `return` under a `with` whose body only *sometimes* returns is
    refused too: moving the continuation inside would change its rounding
    context.

    Cursors do not forward across this pass.

    Parameters
    ----------
    func : Function
        The function to transform.

    Returns
    -------
    Function
        The transformed function.

    Raises
    ------
    TransformDeclined
        If a `return` cannot be moved.

    Examples
    --------
    ::

        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            if x > 0:
                return 1.0
            y = x * 2
            return y

    ``single_exit(f)`` yields::

        @fp.fpy
        def f(x):
            if x > 0:
                r = 1
            else:
                y = (x * 2)
                r = y
            return r
    """
    if not isinstance(func, Function):
        raise TypeError(f"Expected a \'Function\', got {func}")

    return func.with_ast(SingleExit.apply(func.ast))
