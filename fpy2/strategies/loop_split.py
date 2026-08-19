"""
Scheduling language: loop split
"""

from ..ast import Expr, Integer, NamedId, Var
from ..function import Function
from ..transform import SplitLoop, SplitLoopStrategy
from ..transform.utils.cursor import Cursor


def split(
    func: Function,
    factor: int | str,
    where: int | Cursor | None = None,
    *,
    strategy: SplitLoopStrategy = SplitLoopStrategy.PEEL,
    temp_id: str = 't',
    outer_id: str = 'i',
    inner_id: str = 'j'
) -> Function:
    """
    Split ``for`` loops in `func` into nested loops over chunks of
    `factor` elements (Halide's ``split``).

    Parameters
    ----------
    func : Function
        The function to transform.
    factor : int | str
        The chunk size — a positive constant, or the name of a free
        variable of the function holding it (a variable factor is
        guarded by a runtime ``assert factor >= 1``).
    where : int | Cursor | None
        Which `for` loop to split: an index counting `for` loops in visit
        order, outermost-first, or a cursor or region, which takes every loop at
        or beneath it. If `None`, split every `for` loop.
    strategy : SplitLoopStrategy
        How to handle a length that is not a multiple of `factor`.
        Defaults to ``PEEL``, which runs the remainder in a residual
        loop and is correct for any length; ``STRICT`` instead requires
        divisibility — rejected at transform time when provably
        violated, else asserted at runtime.

    Returns
    -------
    Function
        The transformed function.

    Raises
    ------
    ValueError
        If `factor` is a non-positive constant, or ``STRICT`` is used
        on an iterable whose statically-known length is not a multiple
        of a constant `factor`.
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

    ``split(total, 2)`` yields (the default ``PEEL`` strategy runs any
    remainder in a residual loop)::

        @fp.fpy
        def total(xs):
            acc = 0
            t = xs
            with fp.INTEGER:
                t3 = 2
                assert t3 >= 1
                t4 = len(t)
                t5 = (t4 - fp.fmod(t4, t3))
            for i in range(0, t5, t3):
                with fp.INTEGER:
                    t6 = (i + t3)
                for j in range(i, t6, 1):
                    x = t[j]
                    acc = (acc + x)
            for j7 in range(t5, t4, 1):
                x = t[j7]
                acc = (acc + x)
            return acc
    """
    if not isinstance(func, Function):
        raise TypeError(f"Expected a \'Function\', got {func}")

    if isinstance(factor, int):
        if factor < 1:
            raise ValueError(f"Expected a positive integer for factor, got {factor}")
        factor_e: Expr = Integer(factor, None)
    elif isinstance(factor, str):
        factor_e = Var(NamedId(factor), None)
    else:
        raise TypeError(f"Expected an \'int\' or \'str\' for factor, got {factor}")

    log = SplitLoop.apply_with_edits(
        func.ast, factor_e, func.rebase(where), strategy,
        temp_id=NamedId(temp_id), outer_id=NamedId(outer_id), inner_id=NamedId(inner_id)
    )

    return func.with_edits(log)
