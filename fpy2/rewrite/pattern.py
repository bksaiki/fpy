"""
This module defines a pattern of the FPy AST.
"""

from abc import ABC, abstractmethod

from ..ast import Expr, FuncDef, EffectStmt, StmtBlock
from ..utils import NamedId, default_repr


class Pattern(ABC):
    """
    Abstract base class for FPy IR patterns.
    """

    @abstractmethod
    def vars(self) -> set[NamedId]:
        """Returns the set of pattern variables."""
        ...

    @abstractmethod
    def format(self) -> str:
        """Returns a string representation of the pattern."""
        ...


@default_repr
class ExprPattern(Pattern):
    """Expression pattern"""

    expr: Expr
    """syntax of the underlying pattern"""

    def __init__(self, func: FuncDef):
        if not isinstance(func, FuncDef):
            raise TypeError(f'Expected \'FuncDef\', got {type(func)} for {func}')

        stmts = func.body.stmts
        if len(stmts) != 1 or not isinstance(stmts[0], EffectStmt):
            raise TypeError(f'Expected a effectful statement, got {stmts[0]}')

        self.expr = stmts[0].expr


    def vars(self) -> set[NamedId]:
        """Returns the set of pattern variables."""
        raise NotImplementedError

    def format(self) -> str:
        """Returns a string representation of the pattern."""
        return '@pattern\n' + self.expr.format()


@default_repr
class StmtPattern(Pattern):
    """Statement pattern"""

    block: StmtBlock
    """syntax of the underlying pattern"""

    def __init__(self, syntax: FuncDef):
        if not isinstance(syntax, FuncDef):
            raise TypeError(f'Expected \'FuncDef\', got {type(syntax)}')
        self.block = syntax.body

    def vars(self) -> set[NamedId]:
        """Returns the set of pattern variables."""
        raise NotImplementedError

    def format(self) -> str:
        """Returns a string representation of the pattern."""
        return '@pattern\n' + self.block.format()
