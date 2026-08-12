"""
Scheduling language: context lifting
"""

from ..function import Function
from ..transform import LiftContext


def lift_context(func: Function) -> Function:
    """
    Lift statically-known context expressions in `func` to the
    top level.

    Every expression that provably evaluates to a rounding context
    (e.g. a ``fp.IEEEContext(11, 64)`` constructor call) is hoisted to
    a leading ``ctxN = ...`` assignment and its use sites become
    variable references. In particular, a context constructed inside a
    loop body is built once, before the loop, instead of once per
    iteration.

    Context expressions that cannot be statically evaluated (e.g. built
    from an argument) are left in place. The pass is idempotent.

    Parameters
    ----------
    func : Function
        The function to transform.

    Returns
    -------
    Function
        The transformed function.
    """
    if not isinstance(func, Function):
        raise TypeError(f"Expected a \'Function\', got {func}")

    ast = LiftContext.apply(func.ast)
    return func.with_ast(ast)
