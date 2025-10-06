"""
Scheduling language: simplify
"""

from ..function import Function
from ..transform import (
    ConstFold,
    CopyPropagate, ConstPropagate,
    DeadCodeEliminate,
)

def simplify(func: Function) -> Function:
    """
    Applies simplifying transformations to the function:

    - constant folding
    - constant propagation
    - copy propagation
    - dead code elimination
    """
    if not isinstance(func, Function):
        raise TypeError(f"Expected a \'Function\', got {func}")
    ast = func.ast

    # continually simplify until no more can be done
    eliminated = True
    while eliminated:
        ast = ConstFold.apply(ast)
        ast = ConstPropagate.apply(ast)
        ast = CopyPropagate.apply(ast)
        ast, eliminated = DeadCodeEliminate.apply_with_status(ast)

    return func.with_ast(ast)
