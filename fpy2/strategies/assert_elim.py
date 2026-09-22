"""
Scheduling language: dropping assertions
"""

from ..function import Function
from ..transform import AssertElim


def drop_asserts(func: Function) -> Function:
    """
    Remove every ``assert`` from `func`.

    **This changes what the program does.** The other strategies rewrite a
    function while preserving its result; this one does not. An input that
    made `func` abort makes the result return a value instead — whatever the
    unasserted code computes for it, which the assertion existed to say was
    not meaningful. Nothing checks that the condition held.

    It exists for targets that cannot spell an assertion at all. A Triton
    kernel cannot raise, so an ``assert`` has no lowering either way, and the
    only question is which answer the caller wants: a compiler that declines
    the program, or one that compiles the rest of it. Applying this first is
    also what stops an assertion constraining passes that would otherwise
    have rewritten around it — :func:`simplify_if` declines to hoist a branch
    arm holding one, because hoisting would make it run unconditionally.

    Prefer removing the assertion from the source where you control it. Reach
    for this where you do not, or where the assertion is meaningful to another
    backend and merely unspellable in this one.

    There is no partial mode: an assertion is either in the program or it is
    not, so there is nothing for a flag to select.

    FPy admits no empty block, so a statement whose body held only assertions
    is removed with them — a loop that only checked its elements, a ``with``
    that only checked its rounding. A two-armed ``if`` that loses one arm
    becomes a one-armed one over the other.

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
