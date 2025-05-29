"""
This module defines a pattern of the FPy AST.
"""

from abc import ABC, abstractmethod

from ..analysis import DefineUse, LiveVars
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
    """expression of the underlying pattern"""

    _func: FuncDef
    """syntax of the underlying pattern"""

    _vars: set[NamedId]
    """set of pattern variables"""

    def __init__(self, func: FuncDef):
        if not isinstance(func, FuncDef):
            raise TypeError(f'Expected \'FuncDef\', got {type(func)} for {func}')

        stmts = func.body.stmts
        if len(stmts) != 1 or not isinstance(stmts[0], EffectStmt):
            raise TypeError(f'Expected a effectful statement, got {stmts[0]}')

        self.expr = stmts[0].expr
        self._func = func
        self._vars = LiveVars.analyze(self.expr)

    def vars(self) -> set[NamedId]:
        """Returns the set of pattern variables."""
        return set(self._vars)

    def format(self) -> str:
        """Returns a string representation of the pattern."""
        return '@pattern\n' + self.expr.format()

    def to_ast(self) -> FuncDef:
        return self._func


@default_repr
class StmtPattern(Pattern):
    """Statement pattern"""

    block: StmtBlock
    """syntax of the underlying pattern"""

    _func: FuncDef
    """syntax of the underlying pattern"""

    _vars: set[NamedId]
    """set of pattern variables"""

    def __init__(self, func: FuncDef):
        if not isinstance(func, FuncDef):
            raise TypeError(f'Expected \'FuncDef\', got {type(func)}')

        # set of targets (LHS of assignments)
        # TODO: which analysis should be run?
        def_use = DefineUse.analyze(func)
        targets = set(def_use.defs.keys())

        self.block = func.body
        self._func = func
        self._vars = LiveVars.analyze(func) | targets

    def vars(self) -> set[NamedId]:
        """Returns the set of pattern variables."""
        return self._vars

    def format(self) -> str:
        """Returns a string representation of the pattern."""
        return '@pattern\n' + self.block.format()

    def to_ast(self):
        return self._func
