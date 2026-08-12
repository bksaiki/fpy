"""
Scheduling language: reduction fusion
"""

from ..function import Function
from ..transform import ReduceFusion


def fuse(func: Function) -> Function:
    """
    Fuse ``any`` / ``all`` reductions over list comprehensions in `func`
    into single loops, eliminating the intermediate ``list[bool]``::

        r = any([e for x in xs])

    becomes::

        acc = False
        for x in xs:
            t = e
            acc = acc or t
        r = acc

    (``all`` seeds with ``True`` and folds with ``and``.) Reductions
    over anything other than a list comprehension, and reductions in
    positions with no statement to hoist into (e.g. an ``if``
    expression's branches), are left unchanged.

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

    ast = ReduceFusion.apply(func.ast)
    return func.with_ast(ast)
