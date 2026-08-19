"""
Scheduling language: loop unroll
"""

from ..ast import NamedId
from ..function import Function
from ..transform import ForUnroll, ForUnrollStrategy, WhileUnroll
from ..transform.utils.cursor import Cursor


def unroll_while(
    func: Function, where: int | Cursor | None = None, times: int = 1
) -> Function:
    """
    Unroll `while` loops in the function.

    Parameters
    ----------
    func : Function
        The function to transform.
    where : int | Cursor | None
        Which `while` loop to unroll: an index counting `while` loops in visit
        order, outermost-first; or a :class:`fpy2.strategies.StmtCursor` /
        :class:`fpy2.strategies.BlockCursor` naming a program point, which takes
        every loop at or beneath it. A cursor or region from an earlier
        program is forwarded to this one first. If `None`, unroll every
        `while` loop.
    times : int
        The number of times to unroll the loop.

    Returns
    -------
    Function
        The transformed function.

    Raises
    ------
    ValueError
        If `times` is not positive.
    TransformReferenceError
        If `where` does not correspond to a `while` loop.

    Examples
    --------
    ::

        @fp.fpy
        def countdown(x: fp.Real) -> fp.Real:
            while x > 0.0:
                x = x - 1.0
            return x

    ``unroll_while(countdown, times=1)`` yields::

        @fp.fpy
        def countdown(x):
            if x > 0:
                x = (x - 1)
                while x > 0:
                    x = (x - 1)
            return x
    """
    if not isinstance(func, Function):
            raise TypeError(f"Expected a \'Function\', got {func}")
    if not isinstance(times, int):
        raise TypeError(f"Expected an \'int\' for times, got {times}")
    if times < 1:
        raise ValueError(f"Expected a positive integer for times, got {times}")

    return func.with_edits(
        WhileUnroll.apply_with_edits(func.ast, func.rebase(where), times)
    )

def unroll_for(
    func: Function,
    where: int | Cursor | None = None,
    times: int = 1,
    *,
    strategy: ForUnrollStrategy = ForUnrollStrategy.PEEL,
    temp_id: str = 't',
    len_id: str = 'n',
    idx_id: str = 'i'
) -> Function:
    """
    Unroll `for` loops in the function.

    Parameters
    ----------
    where : int | Cursor | None
        Which `for` loop to unroll: an index counting `for` loops in visit
        order, outermost-first; or a :class:`fpy2.strategies.StmtCursor` /
        :class:`fpy2.strategies.BlockCursor` naming a program point, which takes
        every loop at or beneath it. A cursor or region from an earlier
        program is forwarded to this one first. If `None`, unroll every
        `for` loop.
    times : int
        The number of times to unroll the loop; the rewritten loop
        consumes ``times + 1`` consecutive elements per iteration.

    Raises
    ------
    ValueError
        If `times` is not positive.
    TransformReferenceError
        If `where` does not correspond to a `for` loop.

    Examples
    --------
    ::

        @fp.fpy
        def total(xs: list[fp.Real]) -> fp.Real:
            acc = 0.0
            for x in xs:
                acc = acc + x
            return acc

    ``unroll_for(total, times=1)`` yields (the default ``PEEL`` strategy
    runs any odd remainder in a residual loop)::

        @fp.fpy
        def total(xs):
            acc = 0
            t = xs
            with fp.INTEGER:
                n = len(t)
                m = (n - fp.fmod(n, 2))
            for i in range(0, m, 2):
                with fp.INTEGER:
                    i3 = (i + 1)
                x = t[i]
                acc = (acc + x)
                x = t[i3]
                acc = (acc + x)
            for i4 in range(m, n, 1):
                x = t[i4]
                acc = (acc + x)
            return acc
    """
    if not isinstance(func, Function):
            raise TypeError(f"Expected a \'Function\', got {func}")
    if not isinstance(times, int):
        raise TypeError(f"Expected an \'int\' for times, got {times}")
    if times < 1:
        raise ValueError(f"Expected a positive integer for times, got {times}")

    log = ForUnroll.apply_with_edits(
         func.ast, func.rebase(where), times, strategy,
         temp_id=NamedId(temp_id), len_id=NamedId(len_id), idx_id=NamedId(idx_id)
    )

    return func.with_edits(log)

