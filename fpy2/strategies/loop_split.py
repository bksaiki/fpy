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
    inner_id: str = 'i'
):
    """
    Split ``for`` loops in `func` into nested loops over chunks of
    `factor` elements (Halide's ``split``).

    Parameters
    ----------
    func : Function
        The function to transform.
    factor : int | str
        The chunk size — a constant, or the name of a variable in
        scope holding it.
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
            t0 = 2
            with fp.INTEGER:
                t1 = len(t)
                assert fp.fmod(t1, t0) == 0
            for i in range(0, t1, t0):
                with fp.INTEGER:
                    i2 = t[i:(i + t0)]
                for x in i2:
                    acc = (acc + x)
            return acc
    """
    if not isinstance(func, Function):
        raise TypeError(f"Expected a \'Function\', got {func}")
    if not isinstance(factor, (int, str)):
        raise TypeError(f"Expected an \'int\' or \'str\' for factor, got {factor}")
    
    if isinstance(factor, int):
        factor_e: Expr = Integer(factor, None)
    else:
        factor_e = Var(NamedId(factor), None)

    ast = SplitLoop.apply(
        func.ast, factor_e, where, strategy,
        tmp_id=NamedId(temp_id), outer_id=NamedId(outer_id), inner_id=NamedId(inner_id)
    )

    return func.with_ast(ast)
