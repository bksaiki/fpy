"""Comparison operators"""

import operator

from enum import Enum
from typing import Any, Callable

class CompareOp(Enum):
    """Comparison operators as an enumeration"""
    LT = 0
    LE = 1
    GE = 2
    GT = 3
    EQ = 4
    NE = 5

    def symbol(self):
        """Get the symbol for the operator."""
        return _symbol_table[self]

    def invert(self):
        """Assuming `a op b`, returns `op` such that `b op a`."""
        match self:
            case CompareOp.LT:
                return CompareOp.GT
            case CompareOp.LE:
                return CompareOp.GE
            case CompareOp.GE:
                return CompareOp.LE
            case CompareOp.GT:
                return CompareOp.LT
            case CompareOp.EQ:
                return CompareOp.EQ
            case CompareOp.NE:
                return CompareOp.NE
            case _:
                raise RuntimeError('unreachable')

    def to_operator(self):
        return _op_table[self]


_symbol_table = {
    CompareOp.LT: '<',
    CompareOp.LE: '<=',
    CompareOp.GE: '>=',
    CompareOp.GT: '>',
    CompareOp.EQ: '==',
    CompareOp.NE: '!='
}

_op_table: dict[CompareOp, Callable[[Any, Any], bool]] = {
    CompareOp.LT: operator.__lt__,
    CompareOp.LE: operator.__le__,
    CompareOp.GE: operator.__ge__,
    CompareOp.GT: operator.__gt__,
    CompareOp.EQ: operator.__eq__,
    CompareOp.NE: operator.__ne__
}
