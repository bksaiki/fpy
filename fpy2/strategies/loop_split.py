"""
Scheduling language: loop split
"""

from ..ast import Expr, Integer, NamedId, Var
from ..function import Function
from ..transform import SplitLoop, SplitLoopStrategy


def split(
    func: Function,
    factor: int | str,
    where: int | None = None,
    *,
    strategy: SplitLoopStrategy = SplitLoopStrategy.STRICT,
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
        The chunk size — a positive constant, or the name of a
        variable in scope holding it.
    where : int | None
        The index of the `for` loop to split. If `None`, split all
        `for` loops.
    strategy : SplitLoopStrategy
        How to handle a length that is not a multiple of `factor`;
        ``STRICT`` (the only strategy) asserts divisibility at runtime.

    Returns
    -------
    Function
        The transformed function.

    Raises
    ------
    ValueError
        If `factor` is a non-positive constant, or `where` does not
        correspond to a `for` loop.

    Examples
    --------
    ::

        @fp.fpy
        def total(xs: list[fp.Real]) -> fp.Real:
            acc = 0.0
            for x in xs:
                acc = acc + x
            return acc

    ``split(total, 2)`` yields::

        @fp.fpy
        def total(xs):
            acc = 0
            t = xs
            with fp.INTEGER:
                t3 = 2
                t4 = len(t)
                assert fp.fmod(t4, t3) == 0
            for i in range(0, t4, t3):
                with fp.INTEGER:
                    t5 = (i + t3)
                for j in range(i, t5, 1):
                    x = t[j]
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

    ast = SplitLoop.apply(
        func.ast, factor_e, where, strategy,
        tmp_id=NamedId(temp_id), outer_id=NamedId(outer_id), inner_id=NamedId(inner_id)
    )

    return func.with_ast(ast)
