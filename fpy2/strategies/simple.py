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
    enable_const_fold_context: bool = True,
    enable_const_fold_op: bool = True,
    enable_copy_prop: bool = True,
    enable_dead_code_elim: bool = True
) -> Function:
    """Apply :class:`ConstFold` + :class:`CopyPropagate` +
    :class:`DeadCodeEliminate` to *func* until none of the enabled
    passes report a change.

    The two ``enable_const_fold_*`` flags forward to
    :class:`ConstFold`'s ``enable_context`` / ``enable_op`` knobs
    (ignored when ``enable_const_fold=False``).  A common use is
    ``enable_const_fold_context=False`` to simplify boolean / numeric
    expressions while leaving ``with``-block contexts untouched.

    A cursor does not cross this pass: dead-code elimination deletes statements
    it cannot attribute to any one site. It runs last in a schedule, so pin what
    you need before it.

    Examples
    --------
    ::

        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                a = 1.0 + 2.0
                y = x
                if False:
                    return -1.0
                return a * y

    ``simplify(f)`` yields::

        @fp.fpy
        def f(x):
            with fp.FP64:
                return (3 * x)
    """
    if not isinstance(func, Function):
        raise TypeError(f"Expected a \'Function\', got {func}")
    ast = func.ast

    while True:
        changed = False
        if enable_const_fold:
            ast, c = ConstFold.apply_with_status(
                ast,
                enable_context=enable_const_fold_context,
                enable_op=enable_const_fold_op,
            )
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
