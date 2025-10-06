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
