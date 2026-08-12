"""
Scheduling language: reduction fusion
"""

from ..function import Function
from ..transform import ReduceFusion


def fuse(func: Function) -> Function:
    """
    Fuse ``any`` / ``all`` reductions over list comprehensions in `func`
    into single loops, eliminating the intermediate list.

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
        def any_small(xs: list[fp.Real]) -> bool:
            return any([abs(x) < 1e-6 for x in xs])

    ``fuse(any_small)`` yields::

        @fp.fpy
        def any_small(xs):
            acc = False
            for x in xs:
                b = abs(x) < 1e-06
                acc = acc or b
            return acc
    """
    if not isinstance(func, Function):
        raise TypeError(f"Expected a \'Function\', got {func}")

    ast = ReduceFusion.apply(func.ast)
    return func.with_ast(ast)
