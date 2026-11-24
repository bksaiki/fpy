"""
FPy backend abstraction.
"""

from ..ast.fpyast import FuncDef
from ..transform import HoistInvariant, HoistScale, TransformDeclined


class CompileError(Exception):
    """
    Base class for compilation errors.
    """

class Backend:
    """
    Base class for FPy backends.
    """


def hoist(ast: FuncDef) -> FuncDef:
    """*ast* with loop-invariant work and scale factors hoisted, where any."""
    for apply in (HoistInvariant.apply, HoistScale.apply):
        try:
            ast = apply(ast)
        except TransformDeclined:
            pass
    return ast
