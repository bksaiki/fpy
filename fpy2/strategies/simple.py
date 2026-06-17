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
    :class:`DeadCodeEliminate` to *func* until none of the enabled
    passes report a change."""
    if not isinstance(func, Function):
        raise TypeError(f"Expected a \'Function\', got {func}")
    ast = func.ast

    while True:
        changed = False
        if enable_const_fold:
            ast, c = ConstFold.apply_with_status(ast)
            changed |= c
        if enable_copy_prop:
            ast, c = CopyPropagate.apply_with_status(ast)
            changed |= c
        if enable_dead_code_elim:
            ast, c = DeadCodeEliminate.apply_with_status(ast)
            changed |= c
        if not changed:
            break

    return func.with_ast(ast)
