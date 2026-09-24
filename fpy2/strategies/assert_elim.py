"""
Scheduling language: dropping assertions
"""

from ..function import Function
from ..transform import AssertElim


def drop_asserts(func: Function) -> Function:
    """
    Remove every ``assert`` from `func`.

    **This changes what the program does.** The other strategies preserve a
    function's result; this one does not. An input that made `func` abort
    makes the result return whatever the unasserted code computes for it.

    It is for a target that cannot spell an assertion, where the choice is
    between declining the program and compiling the rest of it.

    FPy admits no empty block, so a statement whose body held only assertions
    is removed with them, and a two-armed ``if`` that loses one arm becomes a
    one-armed one over the other.

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
        def recip(x: fp.Real) -> fp.Real:
            assert x != 0, 'nonzero'
            return 1 / x

    ``drop_asserts(recip)`` yields::

        @fp.fpy
        def recip(x):
            return (1 / x)

    where ``recip(0)`` raised before and is now the target's own answer for
    dividing by zero.
    """
    if not isinstance(func, Function):
        raise TypeError(f"Expected a 'Function', got {func}")

    return func.with_ast(AssertElim.apply(func.ast))
