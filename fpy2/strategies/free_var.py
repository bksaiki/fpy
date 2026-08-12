"""
Scheduling language: closing over free variables
"""

from ..function import Function
from ..transform import FreeVarElim


def close(func: Function) -> Function:
    """
    Close `func` over its captured *data* values, materializing each as
    a leading assignment ``x = <value>``.

    Only values with an FPy literal form — numbers, booleans, and
    tuples/lists thereof — are bound. Captures with no literal form
    (a called function, a module, a rounding context) are left free
    for their own resolving machinery.

    Baking the captured values in makes the result self-contained and
    visible to later passes; the bound values are those at the time of
    the call, so subsequent changes to the captured globals no longer
    affect the function.

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

        SCALE = 2.0

        @fp.fpy
        def scaled(x: fp.Real) -> fp.Real:
            return SCALE * x

    ``close(scaled)`` yields::

        @fp.fpy
        def scaled(x):
            SCALE = 2
            return (SCALE * x)
    """
    if not isinstance(func, Function):
        raise TypeError(f"Expected a \'Function\', got {func}")

    ast = FreeVarElim.apply(func.ast)
    return func.with_ast(ast)
