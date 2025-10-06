"""
Scheduling language: loop unroll
"""

from ..ast import NamedId
from ..function import Function
from ..transform import ForUnroll, ForUnrollStrategy, WhileUnroll


def unroll_while(func: Function, where: int | None = None, times: int = 1) -> Function:
    """
    Unroll `while` loops in the function.

    Parameters
    ----------
    func : Function
        The function to transform.
    where : int | None
        The index of the `while` loop to unroll. If `None`, unroll all
        `while` loops.
    times : int
        The number of times to unroll the loop.

    Returns
    -------
    Function
        The transformed function.
    """
    if not isinstance(func, Function):
            raise TypeError(f"Expected a \'Function\', got {func}")
    if not isinstance(times, int):
        raise TypeError(f"Expected an \'int\' for times, got {times}")
    if times < 1:
        raise ValueError(f"Expected a positive integer for times, got {times}")

    ast = WhileUnroll.apply(func.ast, where, times)
    return func.with_ast(ast)

def unroll_for(
    func: Function,
    where: int | None = None,
    times: int = 1,
    *,
    strategy: ForUnrollStrategy = ForUnrollStrategy.STRICT,
    temp_id: str = 't',
    len_id: str = 'n',
    idx_id: str = 'i'
) -> Function:
    """
    Unroll `for` loops in the function.

    Parameters
    ----------
    where : int | None
        The index of the `for` loop to unroll. If `None`, unroll all
        `for` loops.
    times : int
        The number of times to unroll the loop.
    """
    if not isinstance(func, Function):
            raise TypeError(f"Expected a \'Function\', got {func}")
    if not isinstance(times, int):
        raise TypeError(f"Expected an \'int\' for times, got {times}")
    if times < 1:
        raise ValueError(f"Expected a positive integer for times, got {times}")

    ast = ForUnroll.apply(
         func.ast, where, times, strategy,
         temp_id=NamedId(temp_id), len_id=NamedId(len_id), idx_id=NamedId(idx_id)
    )

    return func.with_ast(ast)

