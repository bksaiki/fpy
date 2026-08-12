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

    This is the payoff step of a precision schedule: after
    :func:`fpy2.strategies.monomorphize` pins a context and concrete
    argument formats, this pass deletes the roundings the pinned formats
    made redundant. Run :func:`fpy2.strategies.simplify` afterwards to
    clean up the temporaries the rewrite introduces.

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

    ast = RoundElim.apply(func.ast)
    return func.with_ast(ast)
