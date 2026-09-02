"""
Scheduling language: simplify
"""

from ..function import Function
from ..transform import Simplify


def simplify(
    func: Function, *,
    enable_const_fold: bool = True,
    enable_const_fold_context: bool = False,
    enable_const_fold_op: bool = True,
    enable_copy_prop: bool = True,
    enable_dead_code_elim: bool = True
) -> Function:
    """:class:`fpy2.transform.Simplify` over *func*: constant folding, copy
    propagation and dead-code elimination to a fixpoint.

    The two ``enable_const_fold_*`` flags forward to
    :class:`ConstFold`'s ``enable_context`` / ``enable_op`` knobs
    (ignored when ``enable_const_fold=False``).  Context folding is off by
    default: it replaces a ``with``-block's expression with the resolved
    :class:`~fpy2.Context` object, whose ``repr`` is not FPy source for
    anything but the named formats, so the result no longer re-parses.  Pass
    ``enable_const_fold_context=True`` where an opaque context is wanted --
    a consumer that needs a resolved value folds it itself, as the FPCore
    backend does.

    Cursors do not forward across this pass: it rewrites at sites it does
    not report.

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

    return func.with_ast(Simplify.apply(
        func.ast,
        enable_const_fold=enable_const_fold,
        enable_const_fold_context=enable_const_fold_context,
        enable_const_fold_op=enable_const_fold_op,
        enable_copy_prop=enable_copy_prop,
        enable_dead_code_elim=enable_dead_code_elim,
    ))
