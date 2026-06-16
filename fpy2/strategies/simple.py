"""
Scheduling language: simplify
"""

from ..function import Function
from ..transform import (
    ConstFold,
    CopyPropagate,
    DeadCodeEliminate,
)

def simplify(
    func: Function, *,
    enable_const_fold: bool = True,
    enable_copy_prop: bool = True,
    enable_dead_code_elim: bool = True
) -> Function:
    """Apply :class:`ConstFold` + :class:`CopyPropagate` +
    :class:`DeadCodeEliminate` to *func* until fixpoint."""
    if not isinstance(func, Function):
        raise TypeError(f"Expected a \'Function\', got {func}")
    ast = func.ast

    eliminated = True
    while eliminated:
        if enable_const_fold:
            ast = ConstFold.apply(ast)
        if enable_copy_prop:
            ast = CopyPropagate.apply(ast)
        if enable_dead_code_elim:
            ast, eliminated = DeadCodeEliminate.apply_with_status(ast)
        else:
            eliminated = False

    return func.with_ast(ast)
