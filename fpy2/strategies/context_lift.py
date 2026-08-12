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

    Examples
    --------
    ::

        @fp.fpy
        def accum(xs: list[fp.Real]) -> fp.Real:
            acc = 0.0
            for x in xs:
                with fp.IEEEContext(8, 32):
                    acc = acc + x
            return acc

    ``lift_context(accum)`` builds the context once, before the loop::

        @fp.fpy
        def accum(xs):
            ctx = fp.IEEEContext(8, 32)
            acc = 0
            for x in xs:
                with ctx:
                    acc = (acc + x)
            return acc
    """
    if not isinstance(func, Function):
        raise TypeError(f"Expected a \'Function\', got {func}")

    ast = LiftContext.apply(func.ast)
    return func.with_ast(ast)
