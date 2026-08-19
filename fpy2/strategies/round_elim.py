"""
Scheduling language: rounding elimination
"""

from ..function import Function
from ..transform import RoundElim


def elim_round(func: Function) -> Function:
    """
    Eliminate rounding operations in `func` that are provably identities
    under the active rounding context.

    Arithmetic operations whose unrounded result already fits the active
    context are hoisted into ``with fp.REAL:`` blocks; explicit rounds
    and casts whose argument already fits the target context collapse to
    their argument. Expressions are left unchanged when identity cannot
    be proven.

    Typically run after :func:`fpy2.strategies.monomorphize` pins a
    context and concrete argument formats, which is what makes rounding
    provably redundant; run :func:`fpy2.strategies.simplify` afterwards
    to clean up the temporaries the rewrite introduces.

    A cursor does not cross this pass: hoisting an operation into its own
    ``with fp.REAL:`` block inserts statements ahead of the one that held it,
    at sites the pass does not report. Aim what you need before it, or
    re-list the sites after.

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
    With ``prod3`` monomorphized to FP32 arguments under an FP64
    context::

        pinned = monomorphize(prod3, fp.FP64, [RealType(fp.FP32)] * 3)

    ``elim_round(pinned)`` proves the inner multiply exact (FP32 * FP32
    fits in FP64) and unrounds it::

        @fp.fpy(
            ctx=fp.FP64,
        )
        def prod3(x, y, z):
            with fp.REAL:
                _t = (x * y)
            return (_t * z)
    """
    if not isinstance(func, Function):
        raise TypeError(f"Expected a \'Function\', got {func}")

    ast = RoundElim.apply(func.ast)
    return func.with_ast(ast)
