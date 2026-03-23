"""
Scheduling language: simplify
"""

from ..function import Function
from ..transform import (
    ConstFold,
    CopyPropagate, ConstPropagate,
    DeadCodeEliminate,
)

def simplify(
    func: Function, *,
    enable_const_fold: bool = True,
    enable_const_prop: bool = True,
    enable_copy_prop: bool = True,
    enable_dead_code_elim: bool = True
) -> Function:
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
        if enable_const_fold:
            ast = ConstFold.apply(ast)
        if enable_const_prop:
            ast = ConstPropagate.apply(ast)
        if enable_copy_prop:
            ast = CopyPropagate.apply(ast)
        if enable_dead_code_elim:
            ast, eliminated = DeadCodeEliminate.apply_with_status(ast)
        else:
            eliminated = False

    return func.with_ast(ast)
